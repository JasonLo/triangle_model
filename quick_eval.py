import os, argparse, papermill, json
from tqdm import tqdm

def split_gpu(which_gpu:int, n_splits:int=2):
    """
    Split GPU usage across multiple GPUs
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    memory_size = int(11000/n_splits) # Titan X on Uconn server
    logical_gpus = [tf.config.LogicalDeviceConfiguration(memory_limit=memory_size) for _ in range(n_splits)]

    try:
        # Use only selected GPU(s)
        tf.config.set_visible_devices(gpus[which_gpu], 'GPU')
        tf.config.set_logical_device_configuration(gpus[which_gpu], logical_gpus)

    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass



def main(model_folder, which_gpu: int = 0):
    """
    Using papermill to run parameterized notebook
    To prevent overwriting, set default overwrite to False if needed
    """
    split_gpu(which_gpu)

    import benchmark_hs04
    benchmark_hs04.run_test6_cosine(model_folder)




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
