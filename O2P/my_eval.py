class training_history():
    def __init__(self, pickle_file):
        import pandas as pd
        import pickle

        self.pickle_file = pickle_file
        pickle_in = open(self.pickle_file, "rb")
        hist_obj = pickle.load(pickle_in)
        self.history = pd.DataFrame(hist_obj)
        self.history['epoch'] = self.history.index

    def plot_loss(self):
        return self.plot(col_contains='_loss', plot_title='Loss')

    def plot_acc(self):
        return self.plot(col_contains='Accuracy', plot_title='Binary accuracy')

    def plot_mse(self):
        return self.plot(col_contains='_mse', plot_title='MSE')

    def plot_all(self, save_file=None):
        # plot all 3 training history plots
        # Optionally save plot to html file, see altair plot save documentation
        self.all_plots = self.plot_loss() & self.plot_acc() | self.plot_mse()
        if save_file is not None:
            self.all_plots.save(save_file)
        return self.all_plots

    def plot(self, col_contains, plot_title):
        import altair as alt
        alt.data_transformers.disable_max_rows()

        sel_cols = [col for col in self.history.columns if col_contains in col]
        sel_pd = self.history[['epoch'] + sel_cols].melt('epoch')

        plot = alt.Chart(sel_pd).mark_line().encode(
            x='epoch',
            y='value',
            color=alt.Color('variable', legend=None),
            tooltip=['epoch',
                     'variable']).interactive().properties(title=plot_title)

        return plot


class strain():
    def __init__(self, model_shell, cfg, data):
        import pandas as pd
        # from my_eval import get_pronunciation_fast, get_all_pronunciations_fast
        self.model = model_shell
        self.data = data
        self.cfg = cfg
        self.y_true = get_all_pronunciations_fast(self.data.y_strain,
                                                  self.data.phon_key)

        self.i_hist = pd.DataFrame()  # item history
        self.e_hist = pd.DataFrame()  # epoch history

    def start_evaluate(self):
        import numpy as np
        import pandas as pd
        from IPython.display import clear_output

        for model_idx, model_h5_name in enumerate(self.cfg.path_weights_list):
            # Verbose progress
            clear_output(wait=True)
            progress = model_idx + 1
            totalworks = len(self.cfg.path_weights_list)
            print("Evaluating Strain:", np.round(100 * progress / totalworks,
                                                 0), "%")

            self.model.load_weights(model_h5_name)
            y_pred_matrix = self.model.predict(self.data.x_strain)

            for timestep in range(self.cfg.n_timesteps):
                # Extract output from test set
                y_pred = get_all_pronunciations_fast(y_pred_matrix[timestep],
                                                     self.data.phon_key)
                item_eval, cond_eval = self.eval1_strain(
                    model_h5_name, y_pred, y_pred_matrix, timestep)

                # Stack epoch results to global dataframe
                self.i_hist = pd.concat([self.i_hist, item_eval],
                                        ignore_index=True,
                                        axis=0)
                self.e_hist = pd.concat([self.e_hist, cond_eval],
                                        ignore_index=True,
                                        axis=0)

        clear_output()
        self.parse_strain()
        print('All done')

    def eval1_strain(self, this_h5_name, y_pred, y_pred_matrix, timestep):
        import pandas as pd
        # Item level statistics
        item_eval = self.data.df_strain  # Copy keys
        item_eval['model'] = this_h5_name
        item_eval['epoch'] = item_eval.model.str.slice(-7, -3).astype(int)
        item_eval['timestep'] = timestep
        item_eval['output'] = y_pred
        item_eval['acc'] = get_accuracy(y_pred, self.y_true)
        item_eval['sse'] = get_sse(y_pred_matrix[timestep], self.data.y_strain)

        # Flattened condition level statistics
        pivot_item = item_eval.pivot_table(
            ['acc', 'sse'], columns=['pho_consistency', 'frequency'])
        labels = list(pivot_item.keys().to_series().str.join('_'))
        cond_eval = pd.DataFrame([list(pivot_item)], columns=labels)
        cond_eval['model'] = this_h5_name
        cond_eval['epoch'] = cond_eval.model.str.slice(-7, -3).astype(int)
        cond_eval['timestep'] = timestep

        return item_eval, cond_eval

    def read_eval_from_file(self):
        import pandas as pd
        self.i_hist = pd.read_csv(self.cfg.strain_item_csv)
        self.e_hist = pd.read_csv(self.cfg.strain_epoch_csv)
        self.parse_strain()
        print('Done')

    def parse_strain(self):
        self.e_hist_long = self.e_hist[[
            'epoch', 'timestep', 'acc_CON_HF', 'acc_CON_LF', 'acc_INC_HF',
            'acc_INC_LF', 'sse_CON_HF', 'sse_CON_LF', 'sse_INC_HF',
            'sse_INC_LF'
        ]].melt(['epoch', 'timestep'])

        self.e_hist_long[['metric', 'pho', 'wf'
                          ]] = self.e_hist_long.variable.str.split(pat='_',
                                                                   expand=True)
        self.e_hist_long['condition'] = self.e_hist_long[
            'pho'] + '_' + self.e_hist_long['wf']
        self.e_hist_long['sample'] = self.e_hist_long[
            'epoch'] * self.cfg.steps_per_epoch * self.cfg.batch_size
        self.e_hist_long['sample_mil'] = self.e_hist_long['sample'] / 1e6
        self.e_hist_long[
            'unit_time'] = self.e_hist_long['timestep'] * self.cfg.tau

        self.i_hist['unit_time'] = self.i_hist['timestep'] * self.cfg.tau

        self.e_hist_long['model_version'] = self.cfg.code_name
        self.e_hist_long['test_set'] = 'strain'

        self.i_hist['model_version'] = self.cfg.code_name
        self.i_hist['test_set'] = 'strain'

        self.i_hist.to_csv(self.cfg.strain_item_csv, index=False)
        self.e_hist_long.to_csv(self.cfg.strain_epoch_csv, index=False)

    def plot_development(self, plot_time_step=None):
        import altair as alt
        from altair.expr import datum

        if plot_time_step is None: plot_time_step = self.cfg.n_timesteps - 1

        base = alt.Chart(
            self.e_hist_long[lambda df: df['timestep'] == plot_time_step]
        ).mark_line(point=True).encode(
            x='sample_mil',
            y='value',
            color='condition',
            tooltip=['epoch', 'timestep', 'sample', 'value'])

        dev_plot = alt.hconcat()
        for m in ['acc', 'sse']:
            dev_plot |= base.transform_filter(datum.metric == m).properties(
                title=m)

        return dev_plot

    def plot_time_course(self, plot_epoch=None):
        import altair as alt
        from altair.expr import datum

        if plot_epoch is None: plot_epoch = self.cfg.nEpo

        base = alt.Chart(
            self.e_hist_long[lambda df: df['epoch'] == plot_epoch]).mark_line(
                point=True).encode(
                    x='unit_time',
                    y='value',
                    color='condition',
                    tooltip=['epoch', 'timestep', 'sample', 'value'])

        time_plot = alt.hconcat()
        for m in ['acc', 'sse']:
            time_plot |= base.transform_filter(datum.metric == m).properties(
                title=m)

        return time_plot


class grain():
    def __init__(self, model_shell, cfg, data):
        import pandas as pd
        self.model = model_shell
        self.data = data
        self.cfg = cfg
        self.y_large_grain_true = get_all_pronunciations_fast(
            self.data.y_large_grain, self.data.phon_key)
        self.y_small_grain_true = get_all_pronunciations_fast(
            self.data.y_small_grain, self.data.phon_key)

        self.i_hist = pd.DataFrame()  # item history
        self.e_hist = pd.DataFrame()  # epoch history

    def start_evaluate(self):
        import numpy as np
        import pandas as pd
        from IPython.display import clear_output

        for model_idx, model_h5_name in enumerate(self.cfg.path_weights_list):

            # Verbose progress
            clear_output(wait=True)
            progress = model_idx + 1
            totalworks = len(self.cfg.path_weights_list)
            print("Evaluating Grain:", np.round(100 * progress / totalworks,
                                                0), "%")

            self.model.load_weights(model_h5_name)
            y_pred_matrix = self.model.predict(self.data.x_grain)

            for timestep in range(self.cfg.n_timesteps):

                # Extract output from test set
                y_pred = get_all_pronunciations_fast(y_pred_matrix[timestep],
                                                     self.data.phon_key)
                item_eval, cond_eval = self.eval1_grain(
                    model_h5_name, y_pred, y_pred_matrix, timestep)

                # Stack epoch results to global dataframe
                self.i_hist = pd.concat([self.i_hist, item_eval],
                                        ignore_index=True,
                                        axis=0)
                self.e_hist = pd.concat([self.e_hist, cond_eval],
                                        ignore_index=True,
                                        axis=0)

        clear_output()
        self.parse_grain()
        print('Done')

    def eval1_grain(self, this_h5_name, y_pred, y_pred_matrix, timestep):
        import pandas as pd
        # Item level statistics
        item_eval = self.data.df_grain  # Copy keys
        item_eval['model'] = this_h5_name
        item_eval['epoch'] = item_eval.model.str.slice(-7, -3).astype(int)
        item_eval['timestep'] = timestep
        item_eval['output'] = y_pred
        item_eval['acc_large_grain'] = get_accuracy(y_pred,
                                                    self.y_large_grain_true)
        item_eval['acc_small_grain'] = get_accuracy(y_pred,
                                                    self.y_small_grain_true)
        item_eval['acc_acceptable'] = item_eval['acc_large_grain'] + item_eval[
            'acc_small_grain']
        item_eval['sse_large_grain'] = get_sse(y_pred_matrix[timestep],
                                               self.data.y_large_grain)
        item_eval['sse_small_grain'] = get_sse(y_pred_matrix[timestep],
                                               self.data.y_small_grain)
        item_eval['sse_acceptable'] = item_eval[[
            'sse_large_grain', 'sse_small_grain'
        ]].min(axis=1)

        # Flattened condition level statistics
        pivot_item = item_eval.pivot_table([
            'acc_large_grain', 'acc_small_grain', 'acc_acceptable',
            'sse_small_grain', 'sse_large_grain'
        ],
                                           columns='condition')
        labels = [None] * 10

        for i, col in enumerate(pivot_item.keys().values):
            labels[i * 5:(i + 1) * 5] = list(pivot_item.index.values + '_' +
                                             col)

        cond_eval = pd.DataFrame([pivot_item.values.flatten('F')],
                                 columns=labels)
        cond_eval['model'] = this_h5_name
        cond_eval['epoch'] = cond_eval.model.str.slice(-7, -3).astype(int)
        cond_eval['timestep'] = timestep

        return item_eval, cond_eval

    def read_eval_from_file(self):
        import pandas as pd
        self.i_hist = pd.read_csv(self.cfg.grain_item_csv)
        self.e_hist = pd.read_csv(self.cfg.grain_epoch_csv)
        self.parse_grain()
        print('Done')

    def parse_grain(self):
        self.e_hist['sample'] = self.e_hist[
            'epoch'] * self.cfg.steps_per_epoch * self.cfg.batch_size
        self.e_hist['sample_mil'] = self.e_hist['sample'] / 1e6
        self.e_hist['unit_time'] = self.e_hist['timestep'] * self.cfg.tau

        self.e_hist_long = self.e_hist[[
            'epoch', 'timestep', 'sample', 'sample_mil', 'unit_time',
            'acc_large_grain_ambiguous', 'acc_small_grain_ambiguous',
            'acc_large_grain_unambiguous', 'acc_small_grain_unambiguous',
            'sse_large_grain_ambiguous', 'sse_small_grain_ambiguous',
            'sse_large_grain_unambiguous', 'sse_small_grain_unambiguous'
        ]].melt(['epoch', 'timestep', 'sample', 'sample_mil', 'unit_time'])

        self.e_hist_long[['metric', 'res1', 'res2', 'condition'
                          ]] = self.e_hist_long.variable.str.split(pat='_',
                                                                   expand=True)
        self.e_hist_long['respond'] = self.e_hist_long[
            'res1'] + '_' + self.e_hist_long['res2']
        self.e_hist_long['cond_resp'] = self.e_hist_long[
            'condition'] + '_' + self.e_hist_long['respond']
        self.e_hist_long = self.e_hist_long.drop(
            columns=['res1', 'res2', 'variable'])

        self.e_hist_long['model_version'] = self.cfg.code_name
        self.e_hist_long['test_set'] = 'grain'

        self.i_hist['model_version'] = self.cfg.code_name
        self.i_hist['test_set'] = 'grain'

        self.i_hist.to_csv(self.cfg.grain_item_csv, index=False)
        self.e_hist_long.to_csv(self.cfg.grain_epoch_csv, index=False)

    def plot_development(self, plot_time_step=None):
        import altair as alt
        from altair.expr import datum

        if plot_time_step is None: plot_time_step = self.cfg.n_timesteps - 1

        data = self.e_hist_long[lambda df: df['timestep'] == plot_time_step]
        sel = alt.selection(type='single',
                            on='click',
                            fields=['condition'],
                            empty='all')

        base = alt.Chart(data).add_selection(sel).mark_line(point=True).encode(
            x='sample_mil',
            y='value',
            color='cond_resp',
            opacity=alt.condition(sel, alt.value(1), alt.value(0)),
            tooltip=['epoch', 'timestep', 'sample', 'value'])

        dev_plot = alt.hconcat()
        for m in ['acc', 'sse']:
            dev_plot |= base.transform_filter(datum.metric == m).properties(
                title=m)

        return dev_plot

    def plot_time_course(self, plot_epoch=None):
        import altair as alt
        from altair.expr import datum

        if plot_epoch is None: plot_epoch = self.cfg.nEpo

        data = self.e_hist_long[lambda df: df['epoch'] == plot_epoch]
        sel = alt.selection(type='single',
                            on='click',
                            fields=['condition'],
                            empty='all')

        base = alt.Chart(data).add_selection(sel).mark_line(point=True).encode(
            x='unit_time',
            y='value',
            color='cond_resp',
            opacity=alt.condition(sel, alt.value(1), alt.value(0)),
            tooltip=['epoch', 'timestep', 'sample', 'value'])

        dev_plot = alt.hconcat()
        for m in ['acc', 'sse']:
            dev_plot |= base.transform_filter(datum.metric == m).properties(
                title=m)

        return dev_plot


################################################################
################### function for evaluations ###################
################################################################


def get_pronunciation_fast(act, phon_key):
    import numpy as np

    phonemes = list(phon_key.keys())
    act10 = np.tile([v for k, v in phon_key.items()], 10)

    d = np.abs(act10 - act)
    d_mat = np.reshape(d, (38, 10, 25))
    sumd_mat = np.squeeze(np.sum(d_mat, 2))
    map_idx = np.argmin(sumd_mat, 0)
    out = str()
    for x in map_idx:
        out += phonemes[x]
    return out


def get_all_pronunciations_fast(act, phon_key):
    import numpy as np
    return np.apply_along_axis(get_pronunciation_fast, 1, act, phon_key)


def get_accuracy(output, target):
    import numpy as np
    # Pronunciation level metric
    current_word = 0
    accuracy_list = []
    target = target.tolist()
    for pronunciation in output:
        accuracy_list.append(int(pronunciation == target[current_word]))
        current_word += 1
    return np.array(accuracy_list)


def get_mean_accuracy(output, target):
    import numpy as np
    # Pronunciation level metric
    return np.mean(get_accuracy(output, target))


def get_sse(output, target):
    import numpy as np
    # output level metric
    # Sum of square error
    sse_list = []
    target = target.tolist()
    for i in range(len(output)):
        sse_list.append(np.sum(np.square(output[i] - target[i])))
    return np.array(sse_list)


def get_mean_sse(output, target):
    import numpy as np
    # Mean sum of square error
    # ouput level metric
    return np.mean(get_sse(output, target)) / len(output)


def gen_pkey(p_file="patterns/mapping_v2.txt"):
    import pandas as pd
    # read phonological patterns from the mapping file
    # See Harm & Seidenberg PDF file
    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict('list')
    return m_dict
