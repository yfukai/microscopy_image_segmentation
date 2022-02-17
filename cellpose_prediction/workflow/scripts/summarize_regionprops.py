#!/usr/bin/env python3
"""
cellpose_prediction.py
predict cell positions using cellpose

"""

import os
import signal
from subprocess import Popen, PIPE
import warnings

import numpy as np
import pandas as pd
import zarr
import fire

from tqdm import tqdm
import ipyparallel as ipp
from time import sleep
import cupy as cp

warnings.simplefilter("error",FutureWarning)
warnings.simplefilter("ignore",pd.errors.PerformanceWarning)

import random
hash = random.getrandbits(128)
profile_name="default_%032x" % hash
print(profile_name)


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
            res = func(*args, **kwargs)
        finally:
            print("terminating ipcluster...")
            os.kill(proc.pid, signal.SIGINT)
    return wrapped

@with_ipcluster
def relabel_mask(
    zarr_path,
    suffix="",
    min_area=5,
    gpu_count=4,
    ipcluster_nproc=4,
    ipcluster_execute=True,
    ):
    structure=cp.ones((3,3))
    zarr_file=zarr.open(zarr_path,"r+")
    ds_mask=zarr_file[f"mask{suffix}"]
    sizeT,sizeZ = ds_mask.shape[:2]

    if ipcluster_execute:
        cli = ipp.Client(profile=profile_name)
        dview = cli[:]
        dview.clear()
        bview = cli.load_balanced_view()

        dview.push(dict(
            ds_mask=ds_mask,
            gpu_count=gpu_count,
            min_area=min_area,
        ),block=True)
        sleep(2)
 
    def _relabel(arg):
        import cupy as cp
        from tqdm import tqdm
        from cupyx.scipy import ndimage as ndi
        j,(t,z)=arg
        with cp.cuda.Device(j%gpu_count):
            pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(pool.malloc)
            mask=cp.array(ds_mask[t,z])
            indices=cp.unique(mask[mask!=0])
            new_mask_val=int(cp.max(indices))+1
            for i in tqdm(indices):
                maskarr, num_features = ndi.label(mask==i,structure=structure)
                if num_features!=1:
                    for j in range(2,num_features+1):
                        ind=(maskarr==j)
                        assert cp.any(ind)
                        if cp.sum(ind)>min_area:
                            mask[ind]=new_mask_val
                            new_mask_val=new_mask_val+1
                        else:
                            mask[ind]=0
                ds_mask[t,z]=mask.get()
            del mask
            pool.free_all_blocks()
    

    args=enumerate(np.ndindex(sizeT,sizeZ))
    if ipcluster_execute:
        res=bview.map_async(_relabel,args)
        res.wait_interactive()
        _=res.get()
    else:
        for arg in tqdm(args):
            _relabel(arg)

if __name__ == "__main__":
    try:
        relabel_mask(
            snakemake.input["zarr_path"], # type: ignore
            **snakemake.config["relabel_mask"], # type: ignore
            ipcluster_nproc=snakemake.threads, #type: ignore
        )
    except NameError as e:
        if not "snakemake" in str(e):
            raise e
        fire.Fire(relabel_mask)
