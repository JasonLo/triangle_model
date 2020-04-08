import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import altair as alt
alt.data_transformers.enable("default")
alt.data_transformers.disable_max_rows()
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
            tooltip=['epoch', 'variable']
        ).interactive().properties(title=plot_title)

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
    if plot_time_step is None:
        plot_time_step = df['timestep'].max()

    title_suffix = ' (at timestep {})'.format(plot_time_step + 1)

    base = alt.Chart(df[lambda df: df['timestep'] == plot_time_step]
                    ).mark_line(point=True).encode(x='sample_mil', color=cond)

    dev_plot = alt.hconcat()

    for m in ys:
        dev_plot |= base.encode(
            y=m, tooltip=['epoch', 'timestep', 'sample_mil', m]
        ).properties(title=m + title_suffix)
    return dev_plot


def plot_time_course(df, ys, cond='condition', plot_epoch=None):
    # Choose last epoch as default plot
    if plot_epoch is None:
        plot_epoch = df['epoch'].max()

    title_suffix = ' (at epoch {})'.format(plot_epoch)

    base = alt.Chart(df[lambda df: df['epoch'] == plot_epoch]
                    ).mark_line(point=True).encode(x='unit_time', color=cond)

    time_plot = alt.hconcat()
    for m in ys:
        time_plot |= base.encode(
            y=m, tooltip=['epoch', 'timestep', 'sample_mil', m]
        ).properties(title=m + title_suffix)

    return time_plot


def plot_variables(model, save_file=None):
    """
    Plot all the trainable variables in a model in heatmaps
    """
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
        plt.imshow(
            plot_data, cmap='jet', interpolation='nearest', aspect="auto"
        )
        plt.colorbar()

    if save_file is not None:
        plt.savefig(save_file)


class testset():
    """
    Testset class for evaluating testset
    1. Load model h5 by cfg files provided list (cfg.path_weights_list)
    2. Evaluate test set in each h5 (including every timesteps)
    3. Stitch to one csv file
    """
    def __init__(
        self, cfg, data, model, x_test, x_test_wf, x_test_img, y_test, key_df
    ):
        self.model = model
        self.cfg = cfg
        self.phon_key = data.phon_key

        self.key_df = key_df  # Test set keys

        self.x_test = x_test
        self.x_test_wf = x_test_wf
        self.x_test_img = x_test_img

        self.y_true_matrix = y_test  # Matrix form y_true
        self.y_true = get_all_pronunciations_fast(
            self.y_true_matrix, self.phon_key
        )

        self.i_hist = pd.DataFrame()  # item history

    def eval_one(self, epoch, h5_name, timestep, y_pred_matrix):
        from modeling import input_s

        # Item level statistics
        item_eval = self.key_df
        item_eval['model'] = h5_name
        item_eval['epoch'] = epoch
        item_eval['timestep'] = timestep

        y_pred = get_all_pronunciations_fast(
            y_pred_matrix[timestep], self.phon_key
        )
        
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
            print(
                "Evaluating test set: {}%".format(
                    np.round(100 * progress / totalworks, 0)
                )
            )

            epoch = self.cfg.saved_epoch_list[model_idx]
            self.model.load_weights(model_h5_name)

            test_input = test_set_input(
                self.x_test, self.x_test_wf, self.x_test_img,
                self.y_true_matrix, epoch, self.cfg, test_use_semantic
            )

            y_pred_matrix = self.model.predict(test_input)

            for timestep in range(self.cfg.n_timesteps):

                # Extract output from test set
                y_pred = get_all_pronunciations_fast(
                    y_pred_matrix[timestep], self.phon_key
                )
                item_eval = self.eval_one(
                    epoch, model_h5_name, timestep, y_pred_matrix
                )

                if self.cfg.use_semantic:
                    item_eval['input_s'] = test_input[
                        1][:, timestep,
                           0]  # Dimension guide: item, timestep, p_unit_id
                else:
                    item_eval['input_s'] = 0

                # Stack epoch results to global dataframe
                self.i_hist = pd.concat(
                    [self.i_hist, item_eval], ignore_index=True, axis=0
                )

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

class strain_eval(testset):
    """
    For evaluating Strain results, inherit from testset class
    """
    def __init__(self, cfg, data, model):
        super().__init__(
            cfg, data, model, data.x_strain, data.x_strain_wf,
            data.x_strain_img, data.y_strain, data.df_strain
        )

    def parse_eval(self):
        super().parse_eval()
        self.i_hist['cond_wf'] = self.i_hist['frequency']
        self.i_hist['cond_pho'] = self.i_hist['pho_consistency']
        self.i_hist['cond_img'] = self.i_hist['imageability']
        self.i_hist['condition_pf'] = self.i_hist[
            'pho_consistency'] + '_' + self.i_hist['frequency']
        self.i_hist['condition_pfi'
                   ] = self.i_hist['pho_consistency'] + '_' + self.i_hist[
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

        self.grain_small = testset(
            cfg, data, model, self.x_test, self.x_test_wf, self.x_test_img,
            data.y_small_grain, self.key_df
        )
        self.grain_large = testset(
            cfg, data, model, self.x_test, self.x_test_wf, self.x_test_img,
            data.y_large_grain, self.key_df
        )

    def start_evaluate(self, test_use_semantic, output=None):

        self.grain_small.start_evaluate(test_use_semantic=False)
        self.grain_large.start_evaluate(test_use_semantic=False)

        self.i_hist = self.grain_large.i_hist.rename(
            columns={
                'acc': 'acc_large_grain',
                'sse': 'sse_large_grain'
            }
        )
        self.i_hist = pd.concat(
            [self.i_hist, self.grain_small.i_hist[['acc', 'sse']]], axis=1
        )
        self.i_hist = self.i_hist.rename(
            columns={
                'acc': 'acc_small_grain',
                'sse': 'sse_small_grain'
            }
        )

        self.i_hist['acc_acceptable'] = self.i_hist[
            'acc_large_grain'] * self.i_hist['acc_small_grain'] 
        self.i_hist['sse_acceptable'] = self.i_hist[[
            'sse_large_grain', 'sse_small_grain'
        ]].min(axis=1)

        testset.parse_eval(self)

        if output is not None:
            self.i_hist.to_csv(output, index=False)
            print('Saved file to {}'.format(output))


def make_df_wnw(df, selected_cond):
    """
    This function make a word vs. nonword data file for plotting
    1) filter to last time step
    2) filter by selected_cond
    3) pivot by experiment (exp) and clean

    Inputs: 
        df: compiled batch results data file (cond is the filter column)
        selected_cond: select the condition in word and nonword condition 

    Output:
        plt_df: datafile for plotting with these columns:
            code_name, epoch, nonword_acc, word_acc
    """

    df_sel = df.loc[(df.timestep == df.timestep.max()) &
                    (df.cond.isin(selected_cond)),
                    ['code_name', 'epoch', 'acc', 'exp']]

    pvt = df_sel.pivot_table(index=['code_name', 'epoch'],
                             columns='exp').reset_index()

    plt_df = pd.DataFrame()
    plt_df['code_name'] = pvt.code_name
    plt_df['epoch'] = pvt.epoch
    plt_df['nonword_acc'] = pvt.acc.grain
    plt_df['word_acc'] = pvt.acc.strain

    return plt_df

            
class vis():
    """
    Visualization for a single run
    It parse the datafiles with:
    - parse_strain_cond_df
    - parse_grain_cond_df
    - parse_cond_df (= concat all parsed test sets file)
    - parse_wnw_df (Restructure for condition datafile for Word vs. Nonword plot)
    Then visualize
    
    """
    # Visualize single model
    # Which will parse item level data to condition level data
    # Then plot with Altair
    def __init__(self, model_folder, s_item_csv, g_item_csv):
        from evaluate import training_history, strain_eval, grain_eval, plot_development
        from data_wrangling import my_data
        import altair as alt
        
        self.model_folder = model_folder
        self.load_config()
                    
        self.read_eval_from_file(s_item_csv, g_item_csv)
        self.max_epoch = self.strain_i_hist['epoch'].max()

    def load_config(self):
        from meta import model_cfg
        self.cfg = model_cfg(self.model_folder + '/model_config.json', bypass_chk=True)
        
    def training_hist(self):
        self.t_hist = training_history(self.cfg.path_history_pickle)
        return self.t_hist.plot_all()
        
    def read_eval_from_file(self, s_item_csv, g_item_csv):
        self.strain_i_hist = pd.read_csv(self.model_folder + '/' + s_item_csv)
        self.grain_i_hist = pd.read_csv(self.model_folder + '/' + g_item_csv)
    
    # Condition level parsing
    def parse_strain_cond_df(self, cond):
        self.scdf = self.strain_i_hist[['code_name', 'epoch', 'sample_mil', 'timestep',
                                        'unit_time', cond, 'input_s', 'acc', 'sse']]
        self.scdf = self.scdf.groupby(['code_name', 'epoch', 'timestep', cond],
                                      as_index=False).mean() 
        self.scdf['cond'] = self.scdf[cond]
        self.scdf['exp'] = 'strain'
        
    def parse_grain_cond_df(self, cond):
        self.gcdf = self.grain_i_hist[['code_name', 'epoch', 'sample_mil', 'timestep',
                                       'unit_time', cond, 'input_s',
                                       'acc_acceptable', 'sse_acceptable',
                                       'acc_small_grain', 'sse_small_grain',
                                       'acc_large_grain', 'sse_large_grain']]
        self.gcdf = self.gcdf.rename(columns={'acc_acceptable':'acc', 'sse_acceptable':'sse'})
        self.gcdf = self.gcdf.groupby(['code_name', 'epoch', 'timestep', cond],
                                      as_index=False).mean()
        self.gcdf['cond'] = self.gcdf[cond]
        self.gcdf['exp'] = 'grain'
        
    def parse_cond_df(self, cond_strain='condition_pf', cond_grain='condition', output=None):
        self.parse_strain_cond_df(cond_strain)
        self.parse_grain_cond_df(cond_grain)
        self.cdf = pd.concat([self.scdf, self.gcdf], sort=False)

        if output is not None:
            self.cdf.to_csv(output, index=False)
            print('Saved file to {}'.format(output))
                 
    # Visualization
    def plot_dev(self, y, exp=None, condition='cond', timestep=None):
        """
        Plot developlment (x = epoch)
        Inputs:
        - y: what to plot on y-axis
        - exp: filter on exp column (e.g., 'strain', 'grain')
        - condition: column that group the line color (i.e., separate line by which column)
        - timestep: filter on timestep column, if none provided, take the last timestep
        """
        
        if timestep == None: timestep=self.cfg.n_timesteps
        timestep -= 1 # Reindex

        # Select data
        if exp is not None: 
            plot_df = self.cdf.loc[(self.cdf.exp==exp) & (self.cdf.timestep==timestep),]
        else:
            plot_df = self.cdf.loc[self.cdf.timestep==timestep,]

        # Plotting
        title = '{} at timestep {} / unit time {}'.format(y, timestep + 1, self.cfg.max_unit_time)
        sel = alt.selection(type='single', on='click', fields=[condition], empty='all')
        plot = alt.Chart(
                    plot_df
                ).mark_line(
                    point=True
                ).encode(
                    y=alt.Y(y, scale=alt.Scale(domain=(0, 1))),
                    x='epoch:Q',
                    color=condition,
                    opacity=alt.condition(sel, alt.value(1), alt.value(0)),
                    tooltip=['epoch', 'timestep', 'sample_mil', 'acc', 'sse']
                ).add_selection(sel
                ).interactive(
                ).properties(title=title)

        return plot
    
    def plot_dev_interactive(self, y, exp=None, condition='cond'):
        """
        Interactive version (slider = timestep) of development plot
        Inputs:
        - y: what to plot on y-axis
        - exp: filter on exp column (e.g., 'strain', 'grain')
        - condition: column that group the line color (i.e., separate line by which column)
        """
        
        # Condition highlighter from legend
        select_cond = alt.selection(
            type='multi', on='click', fields=[condition], empty='all', bind="legend"
        )
        
        # Slider timestep filter
        slider_time = alt.binding_range(min=0, max=self.cfg.n_timesteps - 1, step=1)
        select_time = alt.selection_single(
            name="filter",
            fields=['timestep'],
            bind=slider_time,
            init={'timestep': self.cfg.n_timesteps - 1}
        )
        
        # Interactive development plot
        plot_dev = alt.Chart(self.cdf).mark_line(point=True).encode(
            y=alt.Y(y, scale=alt.Scale(domain=(0, 1))),
            x='epoch:Q',
            color=condition,
            opacity=alt.condition(select_cond, alt.value(1), alt.value(0.1)),
            tooltip=['epoch', 'timestep', 'sample_mil', 'acc', 'sse']
        ).add_selection(select_time, select_cond).transform_filter(select_time).properties(
            title='Development plot'
        )

        return plot_dev

    def plot_time(self, y, exp=None, condition='cond', epoch=None):  
        if epoch == None: epoch = self.max_epoch

        # Select data
        if exp is not None: 
            plot_df = self.cdf.loc[(self.cdf.exp==exp) & (self.cdf.epoch == epoch),]
        else:
            plot_df = self.cdf.loc[self.cdf.epoch == epoch,]

        # Plotting
        title = '{} at epoch {} '.format(y, epoch)
        sel = alt.selection(type='single', on='click', fields=[condition], empty='all')
        
        plot = alt.Chart(
                    plot_df
                ).mark_line(
                    point=True
                ).encode(
                    y=alt.Y(y, scale=alt.Scale(domain=(0, 1))),
                    x='unit_time:Q',
                    color=condition,
                    opacity=alt.condition(sel, alt.value(1), alt.value(0)),
                    tooltip=['epoch', 'timestep', 'sample_mil', 'acc', 'sse']
                ).add_selection(sel
                ).interactive(
                ).properties(title=title)

        return plot
    
    def plot_time_interactive(self, y, exp=None, condition='cond'):
        
        # Condition highlighter from legend
        select_cond = alt.selection(
            type='multi', on='click', fields=[condition], empty='all', bind="legend"
        )
            
        # Slider epoch filter
        slider_epoch = alt.binding_range(
            min=self.cfg.save_freq, max=self.cfg.nEpo, step=self.cfg.save_freq
        )
        
        select_epoch = alt.selection_single(
            name="filter",
            fields=['epoch'],
            bind=slider_epoch,
            init={'epoch': self.cfg.nEpo}
        )
        
        # Plot
        plot_time = alt.Chart(self.cdf).mark_line(point=True).encode(
            y=alt.Y(y, scale=alt.Scale(domain=(0, 1))),
            x='unit_time:Q',
            color=condition,
            opacity=alt.condition(select_cond, alt.value(1), alt.value(0.1)),
            tooltip=['epoch', 'timestep', 'sample_mil', 'acc', 'sse']
        ).add_selection(select_epoch, select_cond).transform_filter(select_epoch).properties(
            title='Interactive time plot',
        )
        
        return plot_time
    
    def plot_wnw(self, selected_cond):

        wnw_df = make_df_wnw(self.cdf, selected_cond)

        wnw_plot = (
            alt.Chart(wnw_df).mark_line(point=True).encode(
                y=alt.Y("nonword_acc:Q", scale=alt.Scale(domain=(0, 1))),
                x=alt.X("word_acc:Q", scale=alt.Scale(domain=(0, 1))),
                color=alt.Color("epoch", scale=alt.Scale(scheme="redyellowgreen")),
                tooltip=["code_name", "word_acc", "nonword_acc"],
            ).properties(
                title="Word vs. Nonword accuracy at final time step"
            )
        )
        
        # Plot diagonal
        diagline = alt.Chart(pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1]
        })).mark_line().encode(x=alt.X('x', axis=alt.Axis(labels=False)), 
                               y=alt.Y('y', axis=alt.Axis(labels=False)))

        wnw_with_diag = diagline + wnw_plot
        
        return wnw_with_diag
    
    def plots(self, mode, ys, cond_strain='condition_pf', cond_grain='condition'):
        # Mode = dev(d) / time(t)
        self.parse_cond_df(cond_strain, cond_grain)
        
        plots = alt.hconcat()
        for y in ys:
            if mode == 'd':
                plots |= self.plot_dev(y)
            elif mode == 't':
                plots |= self.plot_time(y, self.max_epoch)
            else:
                print('Use d for development plot, use t for time plot')
            
        return plots
        
    def export_result(self):
        self.parse_cond_df()
        return self.cdf.reset_index(drop=True)
