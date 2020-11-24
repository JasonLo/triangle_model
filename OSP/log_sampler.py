import meta
import data_wrangling
import pandas as pd
from IPython.display import clear_output

cfg = meta.model_cfg(json_file='models/booboo/model_config.json')
data = data_wrangling.MyData()


def dry_run_sampler(sample_name, cfg):
    cfg.sample_name = sample_name
    sampler = data_wrangling.Sampling(cfg, data)
    next(sampler.sample_generator())
    while sampler.debug_log_epoch[-1] <= 100:
        print(sampler.debug_log_epoch)
        next(sampler.sample_generator())
        clear_output(wait=True)

    df_corpus_size = pd.DataFrame({"epoch": sampler.debug_log_epoch,
                                   "corpus_size": sampler.debug_log_corpus_size})
    df_corpus_size.to_csv(
        f"working/dynamic_frequency/{sample_name}_corpus_size.csv")
    sampler.debug_log_dynamic_wf.to_csv(
        f"working/dynamic_frequency/{sample_name}_dynamic_corpus.csv")
    print("All done.")


[dry_run_sampler(sample_name, cfg)
 for sample_name in ['experimental', 'chang', 'hs04']]
