#!/usr/bin/env python3

""" Run a batch of models on multiple GPUs.

    This script will queue up models to run on multiple GPUs.

    Remarks for running on server:
    1a) Put process to background by using "&"
    e.g.: python3 quick_run_papermill.py -f models/batch_run/batch_name/batch_config.json -g 1 &
    1b) Alternatively, "Ctrl+Z" to pause the process, then "bg" to put it in background 
    2) Avoid job being kill after SSH disconnection by "disown"

"""
import argparse, papermill, json, logging, os, meta
from multiprocessing import Queue, Process
from time import sleep
from typing import List


def run_json(json: str):
    """Run a single model."""
    # Load the json
    cfg = meta.Config.from_json(json)

    papermill.execute_notebook(
        os.path.join(cfg.tf_root, "master.ipynb"),
        os.path.join(cfg.model_folder, "output.ipynb"),
        parameters=cfg.papermill_cfg,
    )


def run_one(cfg: dict, free_gpu_queue, which_gpu=None):
    """Run one parameterized notebook.

    Keyword arguments:
    cfg -- a dictionary of config parameters to pass to the notebook
    free_gpu_queue -- a multiprocessing.Queue() to record the status of the run
    which_gpu -- the GPU to run the model on
    """
    # Inject GPU setting into cfg.params
    # GPU instance is handle by each notebook
    cfg["params"]["which_gpu"] = which_gpu

    logging.info(f"Start running {cfg['code_name']} on {which_gpu}")

    # Run the notebook with papermill
    papermill.execute_notebook(
        cfg["in_notebook"],
        cfg["out_notebook"],
        parameters=cfg["params"],
    )

    logging.info(f"Finished running {cfg['code_name']} on {which_gpu}")

    sleep(10)  # Allows GPU memory to release properly
    free_gpu_queue.put(which_gpu)  # Release GPU


def main(
    batch_json: str,
    resume_from: int = None,
    run_only: int = None,
    gpus: List[int] = None,
):
    """Run a batch of models."""
    # Set available GPUs for models to run on
    if gpus is None:
        gpus = [1, 1, 2, 2, 3, 3]

    # Load the batch json
    with open(batch_json) as f:
        batch_cfgs = json.load(f)

    if resume_from is not None:
        batch_cfgs = batch_cfgs[resume_from:]

    if run_only is not None:
        batch_cfgs = [batch_cfgs[run_only]]

    # Configure logging to track the progress of the run
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"models/{batch_cfgs[0]['params']['batch_name']}/run.log",
    )

    # Create a queue to record which GPU is not in use
    free_gpu_queue = Queue()

    # Run the model
    for i, cfg in enumerate(batch_cfgs):

        if i < len(gpus):
            # Allocate one model to each available GPU
            job = Process(target=run_one, args=(cfg, free_gpu_queue, gpus[i]))
            job.start()
        else:
            # Queue for next run: Wait for a free GPU and run the remaining models
            while True:
                try:
                    free_gpu = free_gpu_queue.get()
                    job = Process(target=run_one, args=(cfg, free_gpu_queue, free_gpu))
                    job.start()
                    break  # Run successfuly, break the while loop and continue with next model (for loop)
                except:
                    sleep(5)  # Check every 5 seconds
                    continue


if __name__ == "__main__":
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("json_file", type=str, help="path to batch_config.json")
    parser.add_argument("-r", "--resume_from", type=int, help="resume run from")
    parser.add_argument("-o", "--run_only", type=int, help="run only this index")
    parser.add_argument("-g", "--gpus", type=int, nargs="+", help="list of GPUs")
    args = parser.parse_args()
    main(args.json_file, args.resume_from, args.run_only, args.gpus)
