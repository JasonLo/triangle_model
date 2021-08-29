import os, argparse, papermill, json
from tqdm import tqdm


def run_batch(cfg, which_gpu: int):
    """
    Using papermill to run parameterized notebook
    To prevent overwriting, set default overwrite to False if needed
    """

    print(f"Running model {cfg['code_name']} on GPU: {which_gpu}")

    # Inject GPU setting into cfg.params
    cfg["params"]["which_gpu"] = which_gpu

    os.makedirs(cfg["model_folder"], exist_ok=True)
    papermill.execute_notebook(
        cfg["in_notebook"],
        cfg["out_notebook"],
        parameters=cfg["params"],
    )
    clear_output()


def main(batch_json, which_gpu: int = 0):
    """
    Using papermill to run parameterized notebook
    To prevent overwriting, set default overwrite to False if needed
    """

    with open(batch_json) as f:
        batch_cfgs = json.load(f)

    for cfg in tqdm(batch_cfgs):
        try:
            if cfg['sn'] > 1:
                run_batch(cfg, which_gpu)
        except Exception:
            pass


if __name__ == "__main__":
    """Command line entry point, run batch with batch cfg
    If run on server:
        1) Put process to background by using "&"
        e.g.: python3 quick_run_papermill.py -f models/batch_run/batch_name/batch_config.json -g 1 &
        2) Avoid job being kill after disconnection by calling "disown"
    """
    parser = argparse.ArgumentParser(description="Train TF model with config json")
    parser.add_argument("-f", "--json_file", required=True, type=str)
    parser.add_argument("-g", "--which_gpu", required=False, type=int)
    args = parser.parse_args()

    if args.which_gpu:
        main(args.json_file, args.which_gpu)
    else:
        main(args.json_file)
