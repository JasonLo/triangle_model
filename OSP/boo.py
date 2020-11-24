import imp, data_wrangling, meta
imp.reload(data_wrangling)


cfg = meta.model_cfg(json_file='models/sampling_test_experimental_tmp_sem/model_config.json')
data = data_wrangling.my_data()

sampler = data_wrangling.sampling(cfg, data)

samp_gen = sampler.sample_generator(True)

x, y = next(samp_gen)

[tmp.shape for tmp in x]








