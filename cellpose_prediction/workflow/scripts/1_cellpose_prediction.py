#!/usr/bin/env python3
"""
cellpose_prediction.py
predict cell positions using cellpose

"""

from os import path
import os
import warnings

import numpy as np
import pandas as pd
import zarr
import click

from tqdm import tqdm
from skimage.io import imsave
from time import sleep
import yaml
import cellpose
from matplotlib import pyplot as plt
from os import path
import torch
import sys
from time import sleep
import GPUtil

@click.command()
@click.argument("zarr_path", type=click.Path(exists=True))
@click.argument("metadata_yaml", type=click.Path(exists=True))
@click.argument("report_path", type=click.Path())
def main(
    zarr_path,
    metadata_yaml,
    report_path,
    cellpose_model_path=None,
    suffix="",
    diameter=0,
    cyto_channel="Phase",
    nucleus_channels=None,
    show_segmentation_ind_ratio=False,
    cellprob_threshold=0.,
    flow_threshold=0.4,
    cellpose_normalize=True,
    gpu_count=4,
    gpu_offset=0,
    roi=None,
    ):

    zarr_file=zarr.open(zarr_path,"r+")
    with open(metadata_yaml, "r") as f:
        metadata=yaml.safe_load(f)
    channel_names=metadata["channel_names"]

    example_dir=path.join(report_path,"cellpose_examples")
    os.makedirs(example_dir,exist_ok=True)
    log_dir=path.join(report_path,"cellpose_log")
    os.makedirs(log_dir,exist_ok=True)

    ds_image=zarr_file["image"] #assume TCZYX
    sizeT=ds_image.shape[0]
    sizeZ=ds_image.shape[2]
    shape_mask=[ds_image.shape[0],*ds_image.shape[2:]]
    chunks=[1,1,2048,2048]
    ds_mask=zarr_file.create_dataset(f"mask{suffix}",
                shape=shape_mask,
                chunks=chunks,
                dtype=np.int32,
                overwrite=True)
    ds_flow=zarr_file.create_dataset(f"flow_hsv{suffix}",
                shape=[*shape_mask,3],
                chunks=[*chunks,3],
                dtype=np.int32,
                overwrite=True)
    ds_prob=zarr_file.create_dataset(f"cell_prob{suffix}",
                shape=shape_mask,
                chunks=chunks,
                dtype=np.float32,
                overwrite=True)

    c_channel_ind=channel_names.index(cyto_channel)
    try:
        if nucleus_channels is None: n_channel_inds=None
        else: n_channel_inds=list(map(
            lambda x:channel_names.index(x),nucleus_channels))
    except AssertionError:
        n_channel_inds=None
    
    def predict(args):
        jj,(t,z)=args
        # to avoid logger corruption
        import logging
        log_file=path.join(log_dir,f"cellpose_{jj}.log")
        cellpose.logger.handlers=[]
        cellpose.logger.addHandler(logging.FileHandler(log_file))
        cellpose.logger.addHandler(logging.StreamHandler(sys.stdout))

        print(jj)
        cyto_img=ds_image[t,c_channel_ind,z,...]
        if n_channel_inds is not None:
            def normalize(image):
                q1,q2=np.percentile(image,[0.1,99.9])
                return np.clip((image-q1)/(q2-q1),0,1)
            nucleus_img=np.mean([
                normalize(ds_image[t,n_channel_ind,z,...])
                for n_channel_ind in n_channel_inds],axis=0)
            prediction_img=np.array([cyto_img,nucleus_img])
            prediction_channels=[1,2]
        else:
            prediction_img=np.array(cyto_img)
            prediction_channels=[0,0]

        if not roi is None:
            roi_slices=(slice(roi[0],roi[1]),slice(roi[2],roi[3]))
        else:
            roi_slices=(slice(None),slice(None))
        prediction_img=prediction_img[...,roi_slices[0],roi_slices[1]]

        #get a device with sufficient memory
        all_deviceIDs=GPUtil.getAvailable("id",limit=10,maxMemory=1.0,maxLoad=1.0)
        excludeID=[j for j in all_deviceIDs 
                   if not j in np.arange(gpu_offset,gpu_offset+gpu_count)]
        sleep(jj%gpu_count*5)
        while True:
            deviceIDs=GPUtil.getAvailable(
                "memory",maxMemory=1.0,
                excludeID=excludeID)
            if len(deviceIDs)>0:
                gpu_index=deviceIDs[0]
                break
            sleep(1)
        print(jj,gpu_count,"using gpu: ",gpu_index)

        if cellpose_model_path is None:
            model = cellpose.models.Cellpose(gpu=True,torch=True,
                                   device=torch.device(gpu_index))
            masks, flow, _, _ = model.eval(\
                   [prediction_img], channels=prediction_channels)
        else:
            model = cellpose.models.CellposeModel(
                   gpu=True,
                   torch=True,
                   device=torch.device(gpu_index),
                   pretrained_model=cellpose_model_path, 
               )
            masks, flow, _ = model.eval(
                   [prediction_img],rescale=[1.], 
                   channels=prediction_channels,
                   cellprob_threshold=cellprob_threshold,
                   diameter=diameter,
                   normalize=cellpose_normalize,
                   flow_threshold=flow_threshold)

        ds_mask[t,z,roi_slices[0],roi_slices[1]]=masks[0]
        ds_flow[t,z,roi_slices[0],roi_slices[1]]=flow[0][0]
        ds_prob[t,z,roi_slices[0],roi_slices[1]]=flow[0][2]
        torch.cuda.empty_cache() 
        
        if show_segmentation_ind_ratio and jj==int(show_segmentation_ind_ratio*sizeT*sizeZ):
            fig = plt.figure(figsize=(40,10))
            cellpose.plot.show_segmentation(
                fig, 
                prediction_img, 
                masks[0], 
                flow[0][0],
                channels=prediction_channels)
            
            fig.savefig(path.join(example_dir,f"cellpose_example_{suffix}.pdf"))
            img_prefix=f"cellpose_output_{image_id}_t{t}_z{z}_{suffix}"
            imsave(path.join(example_dir,f"{img_prefix}_brightfield.tiff"),cyto_img)
            if n_channel_inds is not None:
                imsave(path.join(example_dir,f"{img_prefix}_nucleus.tiff"),nucleus_img)
            imsave(path.join(example_dir,f"{img_prefix}_mask.tiff"),masks[0])

    args=list(enumerate(np.ndindex(sizeT,sizeZ)))
      

if __name__ == "__main__":
    main()
