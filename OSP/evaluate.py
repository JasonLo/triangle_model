import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import ast, h5py

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
        self.all_plots = self.plot_loss() | self.plot_mse() | self.plot_acc()
        
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
    return 1 * np.array(output == target)


def get_mean_accuracy(output, target):
    return np.mean(get_accuracy(output, target))


def get_sse(output, target):
    """ Get sum squared error at last axis (item level)
    """
    return np.sum(np.square(output - target), axis=-1)


def get_mean_sse(output, target):
    return np.mean(get_sse(output, target))

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

        self.y_true_matrix = y_test
        
        if type(self.y_true_matrix) is not dict:
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
        
        # Special case when only have one timestep output                
        if self.cfg.output_ticks > 1:
            y_pred_matrix_at_this_time = y_pred_matrix[timestep]
        else:
            y_pred_matrix_at_this_time = y_pred_matrix

        # Extract output from test set
        y_pred = get_all_pronunciations_fast(
            y_pred_matrix_at_this_time, self.phon_key
        )

        item_eval['output'] = y_pred

        item_eval['acc'] = get_accuracy(y_pred, self.y_true)
        
        
        item_eval['sse'] = get_sse(y_pred_matrix_at_this_time, self.y_true_matrix)

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
            

            for timestep in range(self.cfg.output_ticks):
                
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
        self.i_hist['unit_time'] = round(
            (
                self.i_hist['timestep'] +
                (self.cfg.n_timesteps - self.cfg.output_ticks + 1)
            ) * self.cfg.tau, 2
        )
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

    def start_evaluate(self, output=None):
        """
        Always zero "semantic input"
        """
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

        self.i_hist['acc_acceptable'] = (self.i_hist.acc_large_grain | self.i_hist.acc_small_grain)
        self.i_hist['sse_acceptable'] = self.i_hist[[
            'sse_large_grain', 'sse_small_grain'
        ]].min(axis=1)

        testset.parse_eval(self)

        if output is not None:
            self.i_hist.to_csv(output, index=False)
            print('Saved file to {}'.format(output))

class taraban_eval(testset):
    """
    Evaluate Taraban testset
    """
    def __init__(self, cfg, data, model):
        super().__init__(
            cfg, data, model, data.x_taraban, data.x_taraban_wf,
            data.x_taraban_img, data.y_taraban, data.df_taraban
        )
        
class glushko_eval(testset):
    """
    Evaluate Glushko testset
    Need to handle variable y_true
    """
    def __init__(self, cfg, data, model):
        super().__init__(
            cfg, data, model, data.x_glushko, data.x_glushko_wf,
            data.x_glushko_img, data.y_glushko, data.df_glushko
        )

        self.y_dict = data.y_glushko
        self.pho_dict = data.pho_glushko

    def eval_one(self, epoch, h5_name, timestep, y_pred_matrix):
        from modeling import input_s

        # Item level statistics
        item_eval = self.key_df
        item_eval['model'] = h5_name
        item_eval['epoch'] = epoch
        item_eval['timestep'] = timestep
        
        # Special case when only have one timestep output                
        if self.cfg.output_ticks > 1:
            y_pred_matrix_at_this_time = y_pred_matrix[timestep]
        else:
            y_pred_matrix_at_this_time = y_pred_matrix
        
        y_pred = get_all_pronunciations_fast(
            y_pred_matrix_at_this_time, self.phon_key
        )

        item_eval['output'] = y_pred

        # Calculate accuracy in each word and each ans
        acc_list = []
        for i, y in enumerate(y_pred):
            y_true_list = self.pho_dict[self.key_df.word[i]]
            acc = 1 * np.max([y == ans for ans in y_true_list])
            acc_list.append(acc)

        # Calculate sse in each word and each ans
        sse_list = []
        for i, y in enumerate(y_pred_matrix_at_this_time):
            y_true_matrix_list = self.y_dict[self.key_df.word[i]]
            sse = np.min(
                [np.sum(np.square(y - ans)) for ans in y_true_matrix_list]
            )
            sse_list.append(sse)

        item_eval['acc'] = acc_list
        item_eval['sse'] = sse_list

        return item_eval
    
def make_df_wnw(df, word_cond, nonword_cond):
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

    df_sel = df.loc[(df.unit_time == df.unit_time.max()) &
                    (df.cond.isin(word_cond + nonword_cond)),
                    ['code_name', 'epoch', 'acc', 'cond']]

    df_sel['wnw'] = list(
        map(lambda x: "word" if x in word_cond else "nonword", df_sel.cond)
    )

    pvt = df_sel.pivot_table(index=['code_name', 'epoch'],
                             columns='wnw').reset_index()

    plt_df = pd.DataFrame()
    plt_df['code_name'] = pvt.code_name
    plt_df['epoch'] = pvt.epoch
    plt_df['word_acc'] = pvt.acc.word
    plt_df['nonword_acc'] = pvt.acc.nonword

    return plt_df

            
class vis():
    """
    Visualization for a single run
    It parse the datafiles with:
    - parse_strain_cond_df
    - parse_grain_cond_df
    - parse_cond_df (concat all parsed test sets file)
    - parse_wnw_df (Restructure for condition datafile for Word vs. Nonword plot)
    """

    # Visualize single model
    # Which will parse item level data to condition level data
    # Then plot with Altair
    def __init__(self, model_folder):
        from data_wrangling import my_data
        from meta import model_cfg
        import altair as alt

        self.model_folder = model_folder
        self.cfg = model_cfg(
            self.model_folder + '/model_config.json', bypass_chk=True
        )
        self.strain_i_hist = pd.read_csv(
            self.model_folder + '/result_strain_item.csv'
        )
        self.grain_i_hist = pd.read_csv(
            self.model_folder + '/result_grain_item.csv'
        )
        self.taraban_i_hist = pd.read_csv(
            self.model_folder + '/result_taraban_item.csv'
        )
        self.glushko_i_hist = pd.read_csv(
            self.model_folder + '/result_glushko_item.csv'
        )

        self.parse_cond_df()
        self.weight = weight(self.cfg.path_weights_list[-1])

    def training_hist(self):
        self.t_hist = training_history(self.cfg.path_history_pickle)
        return self.t_hist.plot_all()
    
    def load_weight(self, epoch=None):
        
        if epoch is not None:
            self.weight = weight(self.cfg.path_weights_checkpoint.format(epoch=epoch))            

    # Condition level parsing
    def parse_strain_cond_df(self, cond):
        self.scdf = self.strain_i_hist[[
            'code_name', 'epoch', 'sample_mil', 'timestep', 'unit_time', cond,
            'input_s', 'acc', 'sse'
        ]]
        self.scdf = self.scdf.groupby(
            ['code_name', 'epoch', 'timestep', cond], as_index=False
        ).mean()
        self.scdf['cond'] = self.scdf[cond]
        self.scdf['exp'] = 'strain'

    def parse_grain_cond_df(self, cond):
        self.gcdf = self.grain_i_hist[[
            'code_name', 'epoch', 'sample_mil', 'timestep', 'unit_time', cond,
            'input_s', 'acc_acceptable', 'sse_acceptable', 'acc_small_grain',
            'sse_small_grain', 'acc_large_grain', 'sse_large_grain'
        ]]
        self.gcdf = self.gcdf.rename(
            columns={
                'acc_acceptable': 'acc',
                'sse_acceptable': 'sse'
            }
        )
        self.gcdf = self.gcdf.groupby(
            ['code_name', 'epoch', 'timestep', cond], as_index=False
        ).mean()
        self.gcdf['cond'] = self.gcdf[cond]
        self.gcdf['exp'] = 'grain'

    def parse_taraban_cond_df(self, cond):
        self.tcdf = self.taraban_i_hist[[
            'code_name', 'epoch', 'sample_mil', 'timestep', 'unit_time', cond,
            'input_s', 'acc', 'sse'
        ]]
        self.tcdf = self.tcdf.groupby(
            ['code_name', 'epoch', 'timestep', cond], as_index=False
        ).mean()
        self.tcdf['cond'] = self.tcdf[cond]
        self.tcdf['exp'] = 'taraban'

    def parse_glushko_cond_df(self, cond):
        self.gkcdf = self.glushko_i_hist[[
            'code_name', 'epoch', 'sample_mil', 'timestep', 'unit_time', cond,
            'input_s', 'acc', 'sse'
        ]]
        self.gkcdf = self.gkcdf.groupby(
            ['code_name', 'epoch', 'timestep', cond], as_index=False
        ).mean()
        self.gkcdf['cond'] = self.gkcdf[cond]
        self.gkcdf['exp'] = 'glushko'

    def parse_cond_df(self, output=None):
        self.parse_strain_cond_df('condition_pf')
        self.parse_grain_cond_df('condition')
        self.parse_taraban_cond_df('cond')
        self.parse_glushko_cond_df('cond')

        self.cdf = pd.concat(
            [self.scdf, self.gcdf, self.tcdf, self.gkcdf], sort=False
        ).reset_index(drop=True)
        self.cdf['unit_time'] = round(self.cdf.unit_time, 2)  # Round to 2dp

        if output is not None:
            self.cdf.to_csv(output, index=False)
            print('Saved file to {}'.format(output))

    def plot_dev_interactive(self, y, exp=None, condition='cond'):
        """
        Interactive version (slider = unit_time) of development plot
        Inputs:
        - y: what to plot on y-axis
        - exp: filter on exp column (e.g., 'strain', 'grain')
        - condition: column that group the line color (i.e., separate line by which column)
        """
        if exp is not None:
            df = self.cdf.loc[self.cdf.exp.isin(exp)]
        else:
            df = self.cdf

        # Condition highlighter from legend
        select_cond = alt.selection(
            type='multi',
            on='click',
            fields=[condition],
            empty='all',
            bind="legend"
        )

        # Slider unit time filter
        time_options = np.linspace(self.cfg.tau, self.cfg.max_unit_time, self.cfg.n_timesteps).round(2)
        radio_time = alt.binding_radio(options=time_options, name = "Unit time: ")

        select_time = alt.selection_single(
            name="filter",
            fields=['unit_time'],
            bind=radio_time,
            init={'unit_time': self.cfg.max_unit_time}
        )

        # Interactive development plot
        plot_dev = alt.Chart(df).mark_line(point=True).encode(
            y=alt.Y(y, scale=alt.Scale(domain=(0, 1))),
            x='epoch:Q',
            color=condition,
            opacity=alt.condition(select_cond, alt.value(1), alt.value(0.1)),
            tooltip=['epoch', 'unit_time', 'sample_mil', 'acc', 'sse']
        ).add_selection(select_time,
                        select_cond).transform_filter(select_time).properties(
                            title='Development plot'
                        )

        return plot_dev

    def plot_time_interactive(self, y, exp=None, condition='cond'):

        if exp is not None:
            df = self.cdf.loc[self.cdf.exp.isin(exp)]
        else:
            df = self.cdf

        # Condition highlighter from legend
        select_cond = alt.selection(
            type='multi',
            on='click',
            fields=[condition],
            empty='all',
            bind="legend"
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
        plot_time = alt.Chart(df).mark_line(point=True).encode(
            y=alt.Y(y, scale=alt.Scale(domain=(0, 1))),
            x='unit_time:Q',
            color=condition,
            opacity=alt.condition(select_cond, alt.value(1), alt.value(0.1)),
            tooltip=['epoch', 'unit_time', 'sample_mil', 'acc', 'sse']
        ).add_selection(select_epoch,
                        select_cond).transform_filter(select_epoch).properties(
                            title='Interactive time plot',
                        )

        return plot_time

    def plot_wnw(self, word_cond, nonword_cond):

        wnw_df = make_df_wnw(self.cdf, word_cond, nonword_cond)

        wnw_line = alt.Chart(wnw_df).mark_line().encode(
            y=alt.Y("nonword_acc:Q", scale=alt.Scale(domain=(0, 1))),
            x=alt.X("word_acc:Q", scale=alt.Scale(domain=(0, 1))),
            tooltip=["code_name", "word_acc", "nonword_acc"],
        )

        wnw_point = wnw_line.mark_point().encode(
            color=alt.
            Color("epoch:Q", scale=alt.Scale(scheme="redyellowgreen"))
        )

        # Plot diagonal
        diagonal = alt.Chart(pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1]
        })).mark_line(color='black').encode(
            x=alt.X('x', axis=alt.Axis(labels=False)),
            y=alt.Y('y', axis=alt.Axis(labels=False))
        )

        wnw_plot = diagonal + wnw_line + wnw_point

        return wnw_plot
    
    
def parse_mikenet_weight(file):
    """Weight parser for MikeNet
    file: file path
    outputs: all weights and biases matrix in pd.Series() format
    All TAOS and DELAYS are ignored
    """
    raw = dict()
    with open(file, "r") as f:
        for i, line in enumerate(f):
            try:
                # Detect number
                line = float(line)
            except:
                pass

            if type(line) is str:
                # Write to raw dictionary if not at the beginning of file
                if i > 0:
                    raw[vname] = vector

                # Clean matrix name
                vname = line.strip()
                vector = []
            else:
                # Gather matrix values
                vector.append(line)
        else:
            # End of file, one last write to raw dict
            raw[vname] = vector

    # Pack useful matrix into pd.Dataframe()
    woh = pd.Series(raw["Ortho -> Hidden"], name="w_oh")
    whp = pd.Series(raw["Hidden -> Phono"], name="w_hp")
    wpp = pd.Series(raw["Phono -> Phono"], name="w_pp")
    wpc = pd.Series(raw["Phono -> PhoHid"], name="w_pc")
    wcp = pd.Series(raw["PhoHid -> Phono"], name="w_cp")

    biasp = pd.Series(raw["Bias -> Phono"], name="bias_p")
    biash = pd.Series(raw["Bias -> Hidden"], name="bias_h")
    biasc = pd.Series(raw["Bias -> PhoHid"], name="bias_c")

    return [biasc, biash, biasp, wcp, whp, woh, wpc, wpp]


class weight:
    """Weight class with multiple formats
    Directly ingest h5 and parse to list of numpy array (nd-array) and pandas series (flatten)
    pd : pd.Series()
    np : np.array()
    """

    def __init__(self, file, format="tf"):
        if format == "tf":
            """Default loading format TensorFlow weight.h5
            """
            f = h5py.File(file, "r")
            ws = f["rnn"]

            self.pd = []
            self.np = []
            self.names = []

            for key in ws.keys():
                self.names.append(key.replace(":0", ""))

                tmp_np = np.array(ws[key][()])
                # Fix single dimension matrix
                if tmp_np.ndim == 1:
                    tmp_np = tmp_np[np.newaxis, :]

                self.np.append(tmp_np)
                self.pd.append(pd.Series(ws[key][()].flatten(), name=key))

        if format == "mn":
            """Construct from MikeNet weight file
            """
            self.pd = parse_mikenet_weight(file)
            self.names = [w.name for w in self.pd]
            
        self.df = pd.concat(map(self.series_to_df, self.pd))
        self.df['abs_weight'] = self.df.weight.abs()
            
    def series_to_df(self, series):
        f = pd.DataFrame(series)
        f.columns = ["weight"]
        f["matrix"] = series.name.replace(":0", "")
        return f
    
    def violinplot(self, savefig=None):
        
        plt.figure(figsize=(15, 5), facecolor="w")
        plt.subplot(121)
        sns.violinplot(x="weight", y="matrix", data=self.df, scale="width")
        plt.subplot(122)
        sns.violinplot(x="abs_weight", y="matrix", data=self.df, scale="width", cut=0)
        
        if savefig is not None:
            plt.savefig(savefig)
            
        plt.show()
    
    def heatmap(self, savefig=None):

        plt.figure(figsize=(20, 20), facecolor="w")

        for i, key in enumerate(self.names):
            plt.subplot(3, 3, i + 1)
            plt.title(key)
            plt.imshow(self.np[i], cmap="jet", interpolation="nearest", aspect="auto")
            plt.colorbar()

        if savefig is not None:
            plt.savefig(savefig)

        plt.show()

    def boxplot(self, savefig=None):

        plt.figure(figsize=(20, 20))

        for i, key in enumerate(self.names):
            w = self.pd[i]
            plt.subplot(3, 3, i + 1)
            stats = " (Absolute weight: M = {0:.2f}, qt95 = {1:.2f}, max = {2:.2f})".format(
                w.abs().mean(), w.abs().quantile(0.95), w.abs().max()
            )
            plt.title(w.name + stats)
            w.plot.box()

        if savefig is not None:
            plt.savefig(savefig)

        plt.show()

    def basic_stat(self):
        return pd.concat([w.describe() for w in self.pd], axis=1)
    