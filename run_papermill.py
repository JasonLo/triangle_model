import os, argparse, papermill, json, logging
from multiprocessing import Queue, Process
from tqdm import tqdm
from time import sleep


def run_one(cfg: dict, queue, which_gpu=None) -> int:
    """Run one parameterized notebook.

    Keyword arguments:
    cfg -- a dictionary of config parameters to pass to the notebook
    queue -- a multiprocessing.Queue() to record the status of the run

    Returns:
    which_gpu -- the GPU slot that was used (for the next queue to run on)
    """
    # Inject GPU setting into cfg.params
    # GPU allocation is done by individual notebook
    cfg["params"]["which_gpu"] = which_gpu

    logging.info(f"Start running model {cfg['code_name']} on {which_gpu}")

    papermill.execute_notebook(
        cfg["in_notebook"],
        cfg["out_notebook"],
        parameters=cfg["params"],
    )

    logging.info(f"Finished running model {cfg['code_name']} on {which_gpu}")

    sleep(10)  # Allows GPU memory to release
    queue.put(which_gpu)  # Record which GPU was used to queue


def main(batch_json: str, resume_from: int=None):
    """Run a batch of models."""
    # Set available GPUs for models to run on
    available_gpus = [0, 0, 1, 1, 2, 2]
    # Load the batch json
    with open(batch_json) as f:
        batch_cfgs = json.load(f)

    if resume_from is not None:
        batch_cfgs = batch_cfgs[resume_from:]

    # Create logging file and format
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f"models/{batch_cfgs[0]['params']['batch_name']}/batch.log"
    )

    # Create a queue to record which GPU was used
    q = Queue()
    processes = []

    # Run the model
    for i, cfg in enumerate(batch_cfgs):
        if i < 6:
            # Cold start with default GPU allocations (which_gpu=None)
            # processes.append(Process(target=run_one, args=(cfg, q, i%3)))
            processes.append(Process(target=run_one, args=(cfg, q, available_gpus[i])))
            processes[i].start()
            sleep(1)
        else:
            # Warm start with next available GPU
            [p.join() for p in processes] # Detect finished process
            released_gpu = q.get()
            processes.append(Process(target=run_one, args=(cfg, q, released_gpu)))
            processes[i].start()
            sleep(1)


if __name__ == "__main__":
    """Command line entry point, run batch with batch cfg
    If run on server:
        1) Put process to background by using "&"
        e.g.: python3 quick_run_papermill.py -f models/batch_run/batch_name/batch_config.json -g 1 &
        2) Avoid job being kill after disconnection by calling "disown"
    """
    parser = argparse.ArgumentParser(description="Train TF model with config json")
    parser.add_argument("-f", "--json_file", required=True, type=str)
    parser.add_argument("-r", "--resume_from", required=False, type=int)
    args = parser.parse_args()

    if args.resume_from is None:
        main(args.json_file)
    else:
        main(args.json_file, args.resume_from)
