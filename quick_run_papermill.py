import os, argparse, papermill, json
from multiprocessing import Pool
from tqdm import tqdm


def run_batch(cfg:dict):
    """
    Using papermill to run parameterized notebook
    To prevent overwriting, set default overwrite to False if needed
    """

    # Inject GPU setting into cfg.params
    # which_gpu = cfg['sn'] % 3
    which_gpu = 1
    
    cfg["params"]["which_gpu"] = which_gpu
    print(f"Running model {cfg['code_name']} on GPU: {which_gpu}")

    os.makedirs(cfg["model_folder"], exist_ok=True)
    papermill.execute_notebook(
        cfg["in_notebook"],
        cfg["out_notebook"],
        parameters=cfg["params"],
    )
    clear_output()


def main(batch_json:str, resume_from:int = 8, n_pools:int = 2):
    """
    Using papermill to run parameterized notebook
    To prevent overwriting, set default overwrite to False if needed
    """

    with open(batch_json) as f:
        batch_cfgs = json.load(f)

    batch_cfgs = batch_cfgs[resume_from:]

    with Pool(n_pools) as p:
        p.map(run_batch, batch_cfgs)


if __name__ == "__main__":
    """Command line entry point, run batch with batch cfg
    If run on server:
        1) Put process to background by using "&"
        e.g.: python3 quick_run_papermill.py -f models/batch_run/batch_name/batch_config.json -g 1 &
        2) Avoid job being kill after disconnection by calling "disown"
    """
    parser = argparse.ArgumentParser(description="Train TF model with config json")
    parser.add_argument("-f", "--json_file", required=True, type=str)
    args = parser.parse_args()

    main(args.json_file)
