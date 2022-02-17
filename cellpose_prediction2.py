#!/usr/bin/env python3
"""
cellpose_prediction.py
predict cell positions using cellpose

"""

from os import path
from glob import glob

import warnings

from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import h5py
import z5py
import fire
from dask import bag as db
from dask import dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar

from IPython.display import display

from matplotlib import pyplot as plt
from skimage import transform, io, filters, morphology, measure

from tqdm import tqdm
from skimage.io import imsave
from collections.abc import Iterable

from cellpose import models,plot
import torch
#import mxnet as mx

warnings.simplefilter("error",FutureWarning)
warnings.simplefilter("ignore",pd.errors.PerformanceWarning)


def row_to_indices(dimension_order,**pos):
    nonspecified_keys=[k for k in dimension_order if not k in pos.keys()]
    to_int=lambda x : int(x) if not isinstance(x,Iterable) else np.array(x).astype(np.int32)
    indices=tuple([to_int(pos[k]) if k in pos.keys() else slice(None)
                  for k in dimension_order])
    return indices,nonspecified_keys
    
def get_channel_ind(c_name,channels):
    ind=[j for j,c in enumerate(channels) if c_name in str(c)]
    if not len(ind)==1:
        print(c_name,channels)
        raise AssertionError()
    return ind[0]
def specify_channel(channel_ind,imgs,nonspecified_keys):
    return imgs[tuple([channel_ind if k=="c" else slice(None) 
                            for k in nonspecified_keys])]
def max_min_norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

#    print(inds,channel_ind,nonspecified_keys)
    return imgs[inds]

def execute_prediction(
    analyzed_dir,
    cellpose_model_path=None,
    suffix="",
    diameter=15,
    brightfield_channel="Phase",
    nucleus_channels=None,
    use_nucleus=True,
    rescaling_method="divide",
    gaussian_sigma=0,
    test_ind_ratio=0.5,
    cellprob_threshold=0.,
    flow_threshold=0.4,
    cellpose_normalize=True,
    th_quantiles=[0.005,1-0.0001],
    gpu_count=4,
    ):

    print(locals())
    params_dict=locals()
    
    series_df=pd.read_hdf(path.join(analyzed_dir,"image_props.hdf5"),"series")
    print(series_df[series_df["is_valid"]].index)
    valid_series=np.array(series_df[series_df["is_valid"]].index)
    
    image_h5f_path=path.join(analyzed_dir,"rescaled_image.hdf5")
    assert path.isfile(image_h5f_path)
    planes_df=pd.read_hdf(image_h5f_path,"planes_df")

    with h5py.File(image_h5f_path,"r") as h5f:
        ds_image=h5f[f'rescaled_image_{rescaling_method}']
        dimension_order=ds_image.attrs["dimension_order"]
        channels=ds_image.attrs["channels"]

        bf_channel_ind=get_channel_ind(brightfield_channel,channels)
        try:
            if nucleus_channels is None: n_channel_inds=None
            else: n_channel_inds=list(map(
                lambda x:get_channel_ind(x,channels),nucleus_channels))
        except AssertionError:
            n_channel_inds=None
        
        bf_valid_image=np.array(ds_image[row_to_indices(dimension_order,
                                 s=valid_series,
                                 c=bf_channel_ind)[0]])
        bf_quantiles=np.quantile(bf_valid_image,th_quantiles)
        if not n_channel_inds is None:
            n_quantiless=[]
            for n_channel_ind in n_channel_inds:
                n_valid_image=np.array(ds_image[row_to_indices(dimension_order,
                                         s=valid_series,
                                         c=n_channel_ind)[0]])
                n_quantiless.append(np.quantile(n_valid_image,th_quantiles))
        else:
            n_quantiless=None
            
        params_dict.update({
            "bf_channel_ind":bf_channel_ind,
            "n_channel_inds":n_channel_inds,
            "bf_quantiles":bf_quantiles,
            "n_quantiless":n_quantiless,
        })

        shape_except_c=[n for n,k in zip(ds_image.shape,dimension_order) if k!="c"]
        assert len(ds_image.shape)==len(dimension_order)

    z5f=z5py.File(path.join(analyzed_dir,f"cellpose_predicted_{suffix}.zr"),"w")
    chunks=[n if k in "xy" else 1 for n,k in zip(shape_except_c,dimension_order.replace("c",""))]
    ds_mask=z5f.create_dataset("mask",shape=shape_except_c,chunks=chunks,dtype=np.int32)
    ds_flow=z5f.create_dataset("flow_hsv",shape=shape_except_c+[3],chunks=chunks+[3],dtype=np.int32)
    ds_prob=z5f.create_dataset("cell_prob",shape=shape_except_c,chunks=chunks,dtype=np.float32)
    nonspecified_keys=[k for k in dimension_order if not k in "stz"]
        
    for k, v in params_dict.items():
        print(k,v)
        try:
            z5f.attrs[k] = v
        except TypeError:
            z5f.attrs[k] = list(map(str,v))

    planes_grouped=list(enumerate(planes_df.groupby(["image","T_index","Z_index"])))

    normalize_bf=lambda img: max_min_norm(
                                 np.clip(filters.gaussian(img,
                                         sigma=gaussian_sigma,
                                         preserve_range=True),
                                     *bf_quantiles))
    if not n_channel_inds is None:
        normalize_ns=[]
        for n_quantiles in n_quantiless:
            normalize_n=lambda img: max_min_norm(
                                        np.clip(filters.gaussian(img,
                                                sigma=gaussian_sigma,
                                                preserve_range=True),
                                            *n_quantiles))
            normalize_ns.append(normalize_n)
        
    def predict(args):
        jj,((s,t,z), _)=args
        indices,_=row_to_indices(dimension_order,s=s,t=t,z=z)
        indices_except_channel=tuple([j for j,k in zip(indices,dimension_order) if k!="c"])
        assert len(indices)==len(dimension_order)
        
        with h5py.File(image_h5f_path,"r") as h5f:
            ds_image=h5f[f'rescaled_image_{rescaling_method}']
            imgs=np.array(ds_image[indices])
        assert "c" in nonspecified_keys

        brightfield_img=specify_channel(bf_channel_ind,imgs,nonspecified_keys)
        brightfield_img=normalize_bf(brightfield_img)
        if n_channel_inds is not None:
            nucleus_img=np.mean([normalize_n(
                specify_channel(ii,imgs,nonspecified_keys))
                for ii,normalize_n in zip(n_channel_inds,normalize_ns)],axis=0)
            if use_nucleus:
                prediction_img=np.array([brightfield_img,nucleus_img])
                prediction_channels=[1,2]
            else:
                prediction_img=np.array(brightfield_img)
                prediction_channels=[0,0]
        else:
            prediction_img=np.array(brightfield_img)
            prediction_channels=[0,0]
        print(jj,gpu_count,"gpu: ",jj%gpu_count)
        if cellpose_model_path is None:
            model = models.Cellpose(gpu=True,torch=True,
                                   device=torch.device(jj%gpu_count))
            masks, flow, _, _ = model.eval(\
                   [prediction_img], channels=prediction_channels)
        else:
            model = models.CellposeModel(
                   gpu=True,
                   torch=True,
                   device=torch.device(jj%gpu_count),
                   pretrained_model=cellpose_model_path, 
               )
            masks, flow, _ = model.eval(
                   [prediction_img],rescale=[1.], 
                   channels=prediction_channels,
                   cellprob_threshold=cellprob_threshold,
                   diameter=diameter,
                   normalize=cellpose_normalize,
                   flow_threshold=flow_threshold)

        ds_mask[indices_except_channel]=masks[0]
        ds_flow[indices_except_channel]=flow[0][0]
        ds_prob[indices_except_channel]=flow[0][2]
        
        if jj==int(test_ind_ratio*len(planes_grouped)):
            fig = plt.figure(figsize=(40,10))
            plot.show_segmentation(
                fig, 
                prediction_img, 
                masks[0], 
                flow[0][0],
                channels=prediction_channels)
            fig.savefig(path.join(analyzed_dir,f"cellpose_example_{suffix}.pdf"))
            img_prefix=f"cellpose_example_{path.basename(analyzed_dir)}_s{s}_t{t}_z{z}_{suffix}"
            imsave(path.join(analyzed_dir,f"{img_prefix}_brightfield.tiff"),brightfield_img)
            if n_channel_inds is not None:
                imsave(path.join(analyzed_dir,f"{img_prefix}_nucleus.tiff"),nucleus_img)
            imsave(path.join(analyzed_dir,f"{img_prefix}_mask.tiff"),masks[0])
        #return masks,flow,styles
    with ProgressBar(): 
        db.from_sequence(planes_grouped).map(predict).compute(num_workers=gpu_count)
        
    print(analyzed_dir + " finished")

if __name__ == "__main__":
    fire.Fire(execute_prediction)
