#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fire
from subprocess import call, check_output
import os
from os import path
import shutil
import yaml

SCRIPT_PATH = path.abspath(__file__)
HOME_PATH = path.expanduser("~")
CACHE_PATH = path.join(HOME_PATH, ".cellpose_prediction_cache")
CONFIG_NAME = "cellpose_prediction_run_config.yaml"
SNAKEMAKE_CONFIG_NAME = "cellpose_prediction_snakemake_config.yaml"


def main(
    n_cores,
    working_directory,
    zarr_name_pattern="stitched_image*",
    conda=True,
    docker=True,
    config="config/config.yaml",
    extra_args="",
    errorfail=False,
    iscache=False,
):
    os.makedirs(CACHE_PATH, exist_ok=True)
    os.environ["SNAKEMAKE_OUTPUT_CACHE"] = CACHE_PATH

    os.chdir(path.dirname(SCRIPT_PATH))
    working_directory = path.abspath(working_directory)
    command = (
        f'snakemake -j{n_cores} -d "{working_directory}" '
        + ('--use-conda ' if conda else '')
        + ('--use-singularity --singularity-args="--nv" ' if docker else '')
        + ('-k --restart-times 5 ' if not errorfail else '')
        + f'--config name_pattern={zarr_name_pattern} '
        + f"--configfile {config} {extra_args}"
    )
    if iscache:
        command = command + "--cache"

    shutil.copy(
        path.join(path.dirname(SCRIPT_PATH), config),
        path.join(working_directory, SNAKEMAKE_CONFIG_NAME),
    )
    git_description = str(check_output(["git", "describe", "--always"]).strip())
    print(git_description)
    with open(path.join(working_directory, CONFIG_NAME), "w") as f:
        yaml.dump(
            {
                "command": command,
                "git_description": git_description,
            },
            f,
        )
    print(command)
    call(command, shell=True)


if __name__ == "__main__":
    fire.Fire(main)
