import os, argparse, papermill, json
from tqdm import tqdm



def main(json_file, which_gpu: int = 0):
    """
    Using papermill to run parameterized notebook
    To prevent overwriting, set default overwrite to False if needed
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    try:
        # Use only selected GPU(s)
        tf.config.set_visible_devices(gpus[which_gpu], 'GPU')
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    import meta, benchmark_hs04

    cfg = meta.Config.from_json(json_file)
    benchmark_hs04.main(cfg.model_folder)


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
