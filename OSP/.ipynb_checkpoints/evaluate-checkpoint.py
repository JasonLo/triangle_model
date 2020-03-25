import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import altair as alt
from IPython.display import clear_output
from data_wrangling import test_set_input


def gen_pkey(p_file="../common/patterns/mappingv2.txt"):
    # read phonological patterns from the mapping file
    # See Harm & Seidenberg PDF file
    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict('list')
    return m_dict


class training_history():
    def __init__(self, pickle_file):
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


################################################################
################### function for evaluations ###################
################################################################


def get_pronunciation_fast(act, phon_key):
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
    return np.apply_along_axis(get_pronunciation_fast, 1, act, phon_key)


def get_accuracy(output, target):
    current_word = 0
    accuracy_list = []
    target = target.tolist()
    for pronunciation in output:
        accuracy_list.append(int(pronunciation == target[current_word]))
        current_word += 1
    return np.array(accuracy_list)


def get_mean_accuracy(output, target):
    return np.mean(get_accuracy(output, target))


def get_sse(output, target):
    sse_list = []
    target = target.tolist()
    for i in range(len(output)):
        sse_list.append(np.sum(np.square(output[i] - target[i])))
    return np.array(sse_list)


def get_mean_sse(output, target):
    return np.mean(get_sse(output, target)) / len(output)


def plot_development(df, ys, cond='condition', plot_time_step=None):
    # Choose last time step as default plot
    if plot_time_step is None: plot_time_step = df['timestep'].max()

    title_suffix = ' (at timestep {})'.format(plot_time_step + 1)

    base = alt.Chart(
        df[lambda df: df['timestep'] == plot_time_step]).mark_line(
            point=True).encode(x='sample_mil', color=cond)

    dev_plot = alt.hconcat()

    for m in ys:
        dev_plot |= base.encode(y=m,
                                tooltip=['epoch', 'timestep', 'sample_mil',
                                         m]).properties(title=m + title_suffix)
    return dev_plot


def plot_time_course(df, ys, cond='condition', plot_epoch=None):
    # Choose last epoch as default plot
    if plot_epoch is None: plot_epoch = df['epoch'].max()

    title_suffix = ' (at epoch {})'.format(plot_epoch)

    base = alt.Chart(df[lambda df: df['epoch'] == plot_epoch]).mark_line(
        point=True).encode(x='unit_time', color=cond)

    time_plot = alt.hconcat()
    for m in ys:
        time_plot |= base.encode(
            y=m, tooltip=['epoch', 'timestep', 'sample_mil',
                          m]).properties(title=m + title_suffix)

    return time_plot


def plot_variables(model, save_file=None):
    nv = len(model.trainable_variables)
    plt.figure(figsize=(20, 20), facecolor='w')
    for i in range(nv):

        # Expand dimension for biases
        if model.trainable_variables[i].numpy().ndim == 1:
            plot_data = model.trainable_variables[i].numpy()[np.newaxis, :]
        else:
            plot_data = model.trainable_variables[i].numpy()

        plt.subplot(3, 3, i + 1)
        plt.title(model.trainable_variables[i].name)
        plt.imshow(plot_data,
                   cmap='jet',
                   interpolation='nearest',
                   aspect="auto")
        plt.colorbar()

    if save_file is not None: plt.savefig(save_file)


class testset():
    def __init__(self, cfg, data, model, x_test, x_test_wf, x_test_img, y_test,
                 key_df):
        self.model = model
        self.cfg = cfg
        self.phon_key = data.phon_key

        self.key_df = key_df  # Test set keys

        self.x_test = x_test
        self.x_test_wf = x_test_wf
        self.x_test_img = x_test_img

        self.y_true_matrix = y_test  # Matrix form y_true
        self.y_true = get_all_pronunciations_fast(self.y_true_matrix,
                                                  self.phon_key)

        self.i_hist = pd.DataFrame()  # item history

    def eval_one(self, epoch, h5_name, timestep, y_pred_matrix):
        from modeling import input_s

        # Item level statistics
        item_eval = self.key_df
        item_eval['model'] = h5_name
        item_eval['epoch'] = epoch
        item_eval['timestep'] = timestep

        y_pred = get_all_pronunciations_fast(y_pred_matrix[timestep],
                                             self.phon_key)
        item_eval['output'] = y_pred
        item_eval['acc'] = get_accuracy(y_pred, self.y_true)
        item_eval['sse'] = get_sse(y_pred_matrix[timestep], self.y_true_matrix)

        return item_eval

    def start_evaluate(self, test_use_semantic, output=None):

        for model_idx, model_h5_name in enumerate(self.cfg.path_weights_list):

            # Verbose progress
            clear_output(wait=True)
            progress = model_idx + 1
            totalworks = len(self.cfg.path_weights_list)
            print("Evaluating test set: {}%".format(
                np.round(100 * progress / totalworks, 0)))

            epoch = self.cfg.saved_epoch_list[model_idx]
            self.model.load_weights(model_h5_name)

            test_input = test_set_input(self.x_test, self.x_test_wf,
                                        self.x_test_img, self.y_true_matrix,
                                        epoch, self.cfg, test_use_semantic)

            y_pred_matrix = self.model.predict(test_input)

            for timestep in range(self.cfg.n_timesteps):

                # Extract output from test set
                y_pred = get_all_pronunciations_fast(y_pred_matrix[timestep],
                                                     self.phon_key)
                item_eval = self.eval_one(epoch, model_h5_name, timestep,
                                          y_pred_matrix)

                if self.cfg.use_semantic:
                    item_eval['input_s'] = test_input[
                        1][:, timestep,
                           0]  # Dimension guide: item, timestep, p_unit_id
                else:
                    item_eval['input_s'] = 0

                # Stack epoch results to global dataframe
                self.i_hist = pd.concat([self.i_hist, item_eval],
                                        ignore_index=True,
                                        axis=0)

        clear_output()
        self.parse_eval()

        print('All done \n')

        if output is not None:
            self.i_hist.to_csv(output, index=False)
            print('Saved file to {}'.format(output))

    def parse_eval(self):
        self.i_hist['uuid'] = self.cfg.uuid
        self.i_hist['code_name'] = self.cfg.code_name
        self.i_hist['unit_time'] = self.i_hist['timestep'] * self.cfg.tau
        # self.i_hist['condition'] = self.i_hist['pho_consistency'] + '_' + self.i_hist['frequency']
        self.i_hist['sample'] = self.i_hist[
            'epoch'] * self.cfg.steps_per_epoch * self.cfg.batch_size
        self.i_hist['sample_mil'] = self.i_hist['sample'] / 1e6

    def read_eval_from_file(self, file):
        import pandas as pd
        self.i_hist = pd.read_csv(file)
        print('Done')

    def parse_cond_df(self, cond):

        self.cdf = self.i_hist[[
            'code_name', 'epoch', 'sample_mil', 'timestep', 'unit_time', cond,
            'input_s', 'acc', 'sse'
        ]]
        self.cdf = self.cdf.groupby(['code_name', 'epoch', 'timestep', cond],
                                    as_index=False).mean()

    def plot_ys(self, ys, cond='condition', save_file=None):
        import pandas as pd
        self.parse_cond_df(cond)
        plots = plot_development(self.cdf, ys, cond) & plot_time_course(
            self.cdf, ys, cond)
        if save_file is not None:
            plots.save(save_file)

        return plots


class strain_eval(testset):
    def __init__(self, cfg, data, model):
        super().__init__(cfg, data, model, data.x_strain, data.x_strain_wf,
                         data.x_strain_img, data.y_strain, data.df_strain)

    def parse_eval(self):
        super().parse_eval()
        self.i_hist['cond_wf'] = self.i_hist['frequency']
        self.i_hist['cond_pho'] = self.i_hist['pho_consistency']
        self.i_hist['cond_img'] = self.i_hist['imageability']
        self.i_hist['condition_pf'] = self.i_hist[
            'pho_consistency'] + '_' + self.i_hist['frequency']
        self.i_hist['condition_pfi'] = self.i_hist[
            'pho_consistency'] + '_' + self.i_hist[
                'frequency'] + '_' + self.i_hist['imageability']


class grain_eval():
    def __init__(self, cfg, data, model):
        self.model = model
        self.cfg = cfg
        self.phon_key = data.phon_key
        self.key_df = data.df_grain
        self.x_test = data.x_grain
        self.x_test_wf = data.x_grain_wf
        self.x_test_img = data.x_grain_img

        self.grain_small = testset(cfg, data, model, self.x_test,
                                   self.x_test_wf, self.x_test_img,
                                   data.y_small_grain, self.key_df)
        self.grain_large = testset(cfg, data, model, self.x_test,
                                   self.x_test_wf, self.x_test_img,
                                   data.y_large_grain, self.key_df)

    def start_evaluate(self, test_use_semantic, output=None):

        self.grain_small.start_evaluate(test_use_semantic=False)
        self.grain_large.start_evaluate(test_use_semantic=False)

        self.i_hist = self.grain_large.i_hist.rename(columns={
            'acc': 'acc_large_grain',
            'sse': 'sse_large_grain'
        })
        self.i_hist = pd.concat(
            [self.i_hist, self.grain_small.i_hist[['acc', 'sse']]], axis=1)
        self.i_hist = self.i_hist.rename(columns={
            'acc': 'acc_small_grain',
            'sse': 'sse_small_grain'
        })

        self.i_hist['acc_acceptable'] = self.i_hist[
            'acc_large_grain'] + self.i_hist['acc_small_grain']
        self.i_hist['sse_acceptable'] = self.i_hist[[
            'sse_large_grain', 'sse_small_grain'
        ]].min(axis=1)

        testset.parse_eval(self)

        if output is not None:
            self.i_hist.to_csv(output, index=False)
            print('Saved file to {}'.format(output))

    def read_eval_from_file(self, file):
        testset.read_eval_from_file(self, file)

    def parse_cond_df(self, cond='condition'):

        self.cdf = self.i_hist[[
            'code_name', 'epoch', 'sample_mil', 'timestep', 'unit_time', cond,
            'input_s', 'acc_acceptable', 'sse_acceptable'
        ]]
        self.cdf = self.cdf.rename(columns={
            'acc_acceptable': 'acc',
            'sse_acceptable': 'sse'
        })
        self.cdf = self.cdf.groupby(['code_name', 'epoch', 'timestep', cond],
                                    as_index=False).mean()

    def plot_ys(self, ys, cond='condition', save_file=None):
        self.parse_cond_df(cond)
        return testset.plot_ys(self, ys, cond, save_file=save_file)
