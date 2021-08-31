import os, argparse, papermill, json
from tqdm import tqdm
from multiprocessing import Pool

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


def run_one_eval(cfg):
    import tensorflow as tf
    which_gpu = cfg['sn'] % 3
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[which_gpu], 'GPU')
    
    import benchmark_hs04   
    cfg["params"]["which_gpu"] = which_gpu
    print(f"Running model {cfg['code_name']} on GPU: {which_gpu}")
    benchmark_hs04.run_test6_cosine(cfg['code_name'], cfg['params']['batch_name'])

def main(json_file, which_gpu: int = 0):  

    with open(json_file) as f:
        batch_cfgs = json.load(f)

    with Pool(3) as p:
        p.map(run_one_eval, batch_cfgs)





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
