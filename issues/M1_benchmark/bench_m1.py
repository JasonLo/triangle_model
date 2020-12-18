def run_model():
# %% Environment

    import tensorflow as tf
    import sys
    import pickle
    sys.path.append("/home/jupyter/tf/src")
    import meta
    import modeling
    import data_wrangling

# %% Create model

    config_dict = {
        "code_name": "bench",
        "tf_root": "/home/jupyter/tf",
        "sample_name": "hs04",
        "rng_seed": 53797,
        "ort_units": 119,
        "pho_units": 250,
        "pho_hidden_units": 100,
        "pho_cleanup_units": 50,
        "pho_noise_level": 0.0,
        "sem_units": 2446,
        "sem_hidden_unit": 500,
        "sem_cleanup_units": 50,
        "sem_nosie_level": 0.0,
        "activation": "sigmoid",
        "tau": 1 / 3,
        "max_unit_time": 4.0,
        "output_ticks": 4,
        "n_mil_sample": 1.,
        "batch_size": 100,
        "learning_rate": 0.005,
        "save_freq": 10,
    }

    cfg = meta.ModelConfig(**config_dict)
    cfg.save()
    data = data_wrangling.MyData()
    tf.random.set_seed(cfg.rng_seed)
    model = modeling.HS04(cfg)


# %% Compile model

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
            beta_1=0.0,
            beta_2=0.999,
            amsgrad=False
        ),
        metrics=["BinaryAccuracy", "mse"],
    )


# %% Training

    my_sampling = data_wrangling.Sampling(cfg, data)

    checkpoint = modeling.ModelCheckpoint_custom(
        cfg.path["weights_checkpoint_fstring"],
        save_weights_only=True,
        period=cfg.save_freq,
    )

    history = model.fit(
        my_sampling.sample_generator(),
        steps_per_epoch=cfg.steps_per_epoch,
        epochs=cfg.total_number_of_epoch,
        verbose=0,
        callbacks=[checkpoint],
    )

    with open(cfg.path["history_pickle"], "wb") as f:
        pickle.dump(history.history, f)

    model.save("model.h5")


# %% Run from terminal call
if __name__ == "__main__":
    from time import time
    start = time()
    run_model()
    print(time()-start)
