#%% Transplant mikenet weights and biases into a TF model

import os
os.chdir('/home/jupyter/triangle_model/')
import troubleshooting
import meta, modeling
import tensorflow as tf

#%% Create MikeNet weight object
mn_weight = troubleshooting.MikeNetWeight("mikenet/Reading_Weight_v10") 


#%% Create a TF model

code_name = "surgery_v10"
batch_name = None
tf_root = "/home/jupyter/triangle_model"

# Model configs
ort_units = 364
pho_units = 200
sem_units = 2446
hidden_os_units = 500
hidden_op_units = 500
hidden_ps_units = 300
hidden_sp_units = 300
pho_cleanup_units = 50
sem_cleanup_units = 50
pho_noise_level = 0.
sem_noise_level = 0.
activation = "sigmoid"

tau = 1 / 3
max_unit_time = 4.0
output_ticks = 13
inject_error_ticks = 11

# Training configs
learning_rate = 0.005
zero_error_radius = 0.1
save_freq = 20

# Environment configs
tasks = ("pho_sem", "sem_pho", "pho_pho", "sem_sem", "ort_pho", "ort_sem", "triangle")
wf_compression = "log"
wf_clip_low = 0
wf_clip_high = 999_999_999
oral_start_pct = 1.0
oral_end_pct = 1.0

oral_sample = 1_800_000
# oral_tasks_ps = (0.4, 0.4, 0.1, 0.1, 0.)
oral_tasks_ps = (0.4, 0.4, 0.05, 0.15, 0., 0., 0.)
transition_sample = 800_000
reading_sample = 15_000_000
# reading_tasks_ps = (0.2, 0.2, 0.05, 0.05, 0.5)
reading_tasks_ps = (0.2, 0.2, 0.05, 0.05, .1, .1, .3)

batch_size = 100
rng_seed = 2021

cfg = meta.ModelConfig.from_global(globals_dict=globals())
model = modeling.MyModel(cfg)
model.build()


# %% Transplanting

for weight in model.weights:
    try:
        name = weight.name[:-2]
        weight.assign(mn_weight.weights_tf[name])
        print(f"Grafted mikenet weight {name} to tf.weights")

        # Post-load weight sanity check
        tf.debugging.assert_equal(mn_weight.weights_tf[name], weight)

    except KeyError:
        print(f"Missing weight {name} in mikenet")
        pass




# %% Save to epoch 1 file

weight_path = cfg.saved_weights_fstring.format(epoch=1)
model.save_weights(weight_path, overwrite=True, save_format="tf")

# %%
