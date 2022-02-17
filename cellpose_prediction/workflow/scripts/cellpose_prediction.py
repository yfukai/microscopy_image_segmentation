#!/usr/bin/env python3
"""
cellpose_prediction.py
predict cell positions using cellpose

"""

from os import path
import os
import signal
from subprocess import Popen, PIPE
import warnings

import numpy as np
import pandas as pd
import zarr
import fire

from tqdm import tqdm
from skimage.io import imsave
import ipyparallel as ipp
from time import sleep
import yaml
import random

#import sys
#sys.path.append("../")
#print(os.listdir("../"))
#from utils import with_ipcluster

def with_ipcluster(func):
    def wrapped(*args, **kwargs):
        if "ipcluster_execute" in kwargs.keys() \
            and not kwargs["ipcluster_execute"]:
            return func(*args, **kwargs)
        if "ipcluster_nproc" in kwargs.keys():
            nproc = kwargs["ipcluster_nproc"]
        else:
            nproc = 1
        if "ipcluster_timeout" in kwargs.keys():
            timeout = kwargs["ipcluster_timeout"]
        else:
            timeout = 100

        hash = random.getrandbits(128)
        profile_name="default_%032x" % hash
        command = ["ipcluster", "start", "--profile", profile_name, "--n", str(nproc)]
        try:
            print("starting ipcluster...")
            proc = Popen(command, stdout=PIPE, stderr=PIPE)
            i = 0
            while True:
                sleep(1)
                outs = proc.stderr.readline().decode("ascii")
                print(outs.replace("\n", ""))
                if "successfully" in outs:
                    break
                if i > timeout:
                    raise TimeoutError("ipcluster timeout")
                i = i + 1
            print("started.")
            res = func(*args, **kwargs, ipcluster_profile=profile_name)
        finally:
            print("terminating ipcluster...")
            os.kill(proc.pid, signal.SIGINT)
    return wrapped


warnings.simplefilter("error",FutureWarning)
warnings.simplefilter("ignore",pd.errors.PerformanceWarning)

def get_channel_ind(c_name,channels):
    ind=[j for j,c in enumerate(channels) if c_name in str(c)]
    if not len(ind)==1:
        print(c_name,channels)
        raise AssertionError()
    return ind[0]

@with_ipcluster
def cellpose_prediction(
    zarr_path,
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
    ipcluster_nproc=4,
    roi=None,
    ipcluster_execute=True,
    ipcluster_profile="default",
    ):

    print(locals())
    params_dict=locals()

  
    assert path.isdir(zarr_path)
    zarr_file=zarr.open(zarr_path,"r+")

    output_dir=zarr_path+"_cellpose_examples"    
    os.makedirs(output_dir, exist_ok=True)
    image_id=path.basename(zarr_path.rstrip(os.sep))

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

    channel_names=ds_image.attrs["channel_names"]
    c_channel_ind=get_channel_ind(cyto_channel,channel_names)
    try:
        if nucleus_channels is None: n_channel_inds=None
        else: n_channel_inds=list(map(
            lambda x:get_channel_ind(x,channel_names),nucleus_channels))
    except AssertionError:
        n_channel_inds=None
    
    if ipcluster_execute:
        cli = ipp.Client(profile=ipcluster_profile)
    #    cli = ipp.Client(profile="default")
        dview = cli[:]
        dview.clear()
        bview = cli.load_balanced_view()

        dview.push(dict(
            ds_image=ds_image,
            ds_mask=ds_mask,
            ds_flow=ds_flow,
            ds_prob=ds_prob,
            n_channel_inds=n_channel_inds,
            c_channel_ind=c_channel_ind,
            output_dir=output_dir,
            gpu_count=gpu_count,
            gpu_offset=gpu_offset,
            cellpose_model_path=cellpose_model_path,
            cellprob_threshold=cellprob_threshold,
            diameter=diameter,
            cellpose_normalize=cellpose_normalize,
            flow_threshold=flow_threshold,
            show_segmentation_ind_ratio=show_segmentation_ind_ratio,
            sizeT=sizeT,
            sizeZ=sizeZ,
            image_id=image_id,
            roi=roi,
        ),block=True)
        sleep(2)

    #@profile
    def predict(args):
        import numpy as np
        from cellpose import models,plot
        from matplotlib import pyplot as plt
        from os import path
        import torch
        import sys
        from time import sleep
        import GPUtil

        jj,(t,z)=args
        # to avoid logger corruption
        import cellpose
        import logging
        log_file=path.join(output_dir,f"cellpose_{jj}.log")
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
        excludeID=[j for j in all_deviceIDs if not j in np.arange(gpu_offset,gpu_offset+gpu_count)]
        sleep(jj%gpu_count*5)
        while True:
            deviceIDs=GPUtil.getAvailable(
                "memory",maxMemory=1.0,
                excludeID=excludeID)
            if len(deviceIDs)>0:
                gpu_index=deviceIDs[0]
                break
            sleep(1)
        print(jj,gpu_count,"gpu: ",gpu_index)

        if cellpose_model_path is None:
            model = models.Cellpose(gpu=True,torch=True,
                                   device=torch.device(gpu_index))
            masks, flow, _, _ = model.eval(\
                   [prediction_img], channels=prediction_channels)
        else:
            model = models.CellposeModel(
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
            plot.show_segmentation(
                fig, 
                prediction_img, 
                masks[0], 
                flow[0][0],
                channels=prediction_channels)
            
            fig.savefig(path.join(output_dir,f"cellpose_example_{suffix}.pdf"))
            img_prefix=f"cellpose_output_{image_id}_t{t}_z{z}_{suffix}"
            imsave(path.join(output_dir,f"{img_prefix}_brightfield.tiff"),cyto_img)
            if n_channel_inds is not None:
                imsave(path.join(output_dir,f"{img_prefix}_nucleus.tiff"),nucleus_img)
            imsave(path.join(output_dir,f"{img_prefix}_mask.tiff"),masks[0])

    args=list(enumerate(np.ndindex(sizeT,sizeZ)))
    print(args)
    if ipcluster_execute:
        res = bview.map_async(predict,args)
        res.wait_interactive()
        _ = res.get()
    else:
        for arg in tqdm(args):
            predict(arg)
       
    print(zarr_path + " finished")

    print(params_dict)
    params_path = path.join(output_dir, "cellpose_prediction_params.yaml")
    with open(params_path, "w") as f:
        yaml.dump(params_dict, f)


if __name__ == "__main__":
    try:
        cellpose_prediction(
            snakemake.input["zarr_path"], # type: ignore
            **snakemake.config["cellpose_prediction"], # type: ignore
            ipcluster_nproc=snakemake.threads, #type: ignore
        )
    except NameError as e:
        if not "snakemake" in str(e):
            raise e
        fire.Fire(cellpose_prediction)
