import pandas as pd
import altair as alt

def parse_mikenet_sims(file_name, task):
    """ Parse item level raw output from MikeNet to Condition level format
    This version only export accuracy results
    file_name: text file of MikeNet output
    task: "strain" or "grain"
    output: formatted pandas dataframe
    """

    df = pd.read_table(file_name)
    df["ID"] = df.run_id
    df["Trial.Scaled"] = df.trial / 1e6
    df["Pnoise"] = df.pnoise
    df["Epsilon"] = df.epsilon
    df["Measure"] = "Accuracy"

    assert (task == "strain") or (task == "grain")

    if task == "strain":
        # Strain specific parsing
        df["Type"] = df.frequency_type + "_" + df.inc_con
        df["Freq"] = df.frequency_type
        df["Cons"] = df.inc_con
        df["Score"] = df.critical_hit

    else:
        # Grain specific parsing
        df["Type"] = df.control_critical.apply(
            lambda x: "NW_AMB" if x == "critical" else "NW_UN"
        )
        df["Freq"] = "NW"
        df["Cons"] = "NW"
        df["Score"] = df.acceptable_hit

    grouping_variables = [
        "ID",
        "Trial.Scaled",
        "Hidden",
        "PhoHid",
        "Pnoise",
        "Epsilon",
        "Type",
        "Measure",
        "Freq",
        "Cons",
    ]

    # Aggregate condition level
    parsed_df = df.groupby(grouping_variables, as_index=False).mean()

    # Export (Preserving variable order)
    export_variables = grouping_variables[0:8] + ["Score"] + grouping_variables[8:10]

    return parsed_df[export_variables]

def parse_from_file(filename):
    """File parser for Zevin's sims
    1. Read from csv
    2. Rename columns
    3. Select accuracy meausre only
    4. Add origin (acc = 0 when epoch = 0)
    5. Select cleanup unit == 20
    6. Add "type" column to indicate word / nonword
    7. Add "rank_pc" column to indicate model percent ranking
    
    """
    df = pd.read_csv(filename, index_col=0)
    df.rename(
        columns={
            "ID": "code_name",
            "Trial.Scaled": "epoch",
            "Hidden": "hidden_units",
            "PhoHid": "cleanup_units",
            "Pnoise": "p_noise",
            "Epsilon": "learning_rate",
            "Type": "cond",
            "Measure": "measure",
            "Score": "score",
            "Freq": "cond_freq",
            "Cons": "cond_cons",
        },
        inplace=True,
    )
    df = df.loc[df.measure == "Accuracy"]
    df = add_origin(df)
    df = df.loc[df.cleanup_units == 20]
    
    df["type"] = df.cond.apply(
        lambda x: "word" if x in ["HF_CON", "HF_INC", "LF_CON", "LF_INC"] else "nonword"
    )
    
    df = get_rank(df)   
    return df


def add_origin(df):
    """Add origin data point in each model"""

    if df.epoch.min() > 0:
        # Borrow epoch == 1.0 as a frame for epoch = 0
        tmp = df.loc[df.epoch == 1.0,].copy()
        tmp.score = 0
        tmp.epoch = 0
        df_with_origin = pd.concat([df, tmp], ignore_index=True)
        return df_with_origin.sort_values(
            by=["code_name", "cond", "epoch"]
        ).reset_index(drop=True)

    else:
        print("Already have origin, returning original df")
        return df


    
def count_grid(df, hpar, input_type="parsed"):
    """Counting how many runs in each h-param cell 
    """
    
    if input_type == "parsed":
        id_var = "code_name"
        lr_var = "learning_rate"
        noise_var = "p_noise"
        cleanup_var = "cleanup_units"
        hidden_var = "hidden_units"
    else:
        id_var = "ID"
        lr_var = "Epsilon"
        noise_var = "Pnoise"
        cleanup_var = "PhoHid"
        hidden_var = "Hidden"
        
    
    
    settings = df[[id_var] + hpar].pivot_table(index=id_var)
    settings[id_var] = settings.index
    settings[lr_var] = settings[lr_var].round(4)

    count_settings = settings.pivot_table(
        index=hpar, aggfunc="count", values=id_var,
    )
    count_settings.reset_index(inplace=True)
    count_settings.rename(columns={id_var: "n"}, inplace=True)

    return (
        alt.Chart(count_settings)
        .mark_rect()
        .encode(
            x=noise_var+":O",
            y=alt.Y(hidden_var+":O", sort="descending"),
            row=alt.Row(lr_var+":O", sort="descending"),
            column=alt.Column(cleanup_var+":O", sort="descending"),
            color="n:O",
            tooltip=hpar + ["n"],
        )
        .properties(title="Model counts")
    )
    
def get_rank(df):
    """Grand mean based rank pc
    """
    gacc = df.groupby("code_name", as_index=False).mean()
    gacc = gacc[["code_name", "score"]]
    gacc["rank_pc"] = gacc.score.rank(pct=True)
    return df.merge(gacc[["code_name", "rank_pc"]], how="left")


class Select_Model:
    """ Helper class for defining TD
    I: Selection:
    1. Control space filter
    2. Rank filter
    3. Accuracy filter (developmental)
    
    II: Plotting:
    1. Where are the selected model in the control space
    2. How's their average performance (in each cond / mean of all conds)
    3. Some basic descriptives in title
    """

    def __init__(self, df):
        self.df = df

    def count_model(self):
        return len(self.df.code_name.unique())

    # Selection related functions

    def select_by_performance(self, threshold_low, threshold_hi, t_low, t_hi):

        n_pre = self.count_model()
        tmp = self.pivot_to_wide(self.df, t_low, t_hi)
        # Selected models
        tmp = tmp.loc[(tmp.t_low < threshold_low) & (tmp.t_hi > threshold_hi)]

        # Create full dataframe of selected models
        self.df = (
            self.df.loc[self.df.code_name.isin(tmp.code_name)]
            .sort_values(by=["code_name", "cond", "epoch"])
            .reset_index()
        )

        n_post = self.count_model()
        print(f"Selected {n_post} models from the original {n_pre} models")

    def select_by_control(self, hidden_units=None, p_noise=None, learning_rate=None):

        n_pre = self.count_model()
        if hidden_units is not None:
            self.df = self.df.loc[self.df.hidden_units.isin(hidden_units)]
        if p_noise is not None:
            self.df = self.df.loc[self.df.p_noise.isin(p_noise)]
        if learning_rate is not None:
            self.df = self.df.loc[self.df.learning_rate.isin(learning_rate)]

        n_post = self.count_model()
        print(f"Selected {n_post} models from the original {n_pre} models")

    def select_by_rankpc(self, minpc, maxpc):
        n_pre = self.count_model()
        self.df = self.df.loc[(self.df.rank_pc >= minpc) & (self.df.rank_pc <= maxpc)]
        n_post = self.count_model()
        print(f"Selected {n_post} models from the original {n_pre} models")

    def select_by_cond(self, conds):
        n_pre = self.count_model()
        self.df = self.df.loc[self.df.cond.isin(conds)]
        n_post = self.count_model()
        print(f"Selected {n_post} models from the original {n_pre} models")

    # Descriptives related functions

    def get_rankpc_desc(self):
        desc = self.df.groupby("code_name").mean().reset_index().rank_pc.describe()
        return f"M:{desc['mean']:.3f} SD: {desc['std']:.3f} Min: {desc['min']:.3f} Max: {desc['max']:.3f}"

    def get_acc_desc(self):
        desc = self.df.groupby("code_name").mean().reset_index().score.describe()
        return f"M:{desc['mean']:.3f} SD: {desc['std']:.3f} Min: {desc['min']:.3f} Max: {desc['max']:.3f}"

    # Plotting related functions

    def pivot_to_wide(self, df, t_low, t_hi):
        """ Create a pivot table of model's t_low and t_hi as column
        df: input datafile
        t_low: epoch used in applying threshold_low
        t_hi : epoch used in applying threshold_hi
        """
        tmp = df.loc[(df.epoch.isin([t_low, t_hi]))]

        index_names = [
            "code_name",
            "hidden_units",
            "p_noise",
            "learning_rate",
        ]

        pvt = tmp.pivot_table(
            index=index_names, columns="epoch", values="score",
        ).reset_index()

        # Rename new columns
        pvt.columns = index_names + ["t_low", "t_hi"]
        return pvt

    def plot_control_space(self):
        """Plot selected models at control space"""

        pdf = self.df.groupby("code_name").mean().round(3).reset_index()

        control_space = (
            alt.Chart(pdf)
            .mark_rect()
            .encode(
                x="p_noise:O",
                y=alt.Y("hidden_units:O", sort="descending"),
                column=alt.Column("learning_rate:O", sort="descending"),
                color="count(code_name)",
            )
        )
        return control_space

    def plot_mean_development(self, show_sd):
        """Plot the mean development of all selected models"""

        development_space_sd = (
            alt.Chart(self.df)
            .mark_errorband(extent="stdev")
            .encode(
                y=alt.Y("score:Q", scale=alt.Scale(domain=(0, 1))),
                x="epoch:Q",
                color="cond:N",
            )
            .properties(
                title="Developmental space: Accuracy in each condition over epoch"
            )
        )

        development_space_mean = development_space_sd.mark_line().encode(
            y="mean(score):Q"
        )

        this_plot = (
            (development_space_mean + development_space_sd)
            if show_sd
            else development_space_mean
        )
        return this_plot

    def plot_all_cond_mean(self, show_sd):
        """Plot the average accuracy in all conditions over epoch of all selected models"""
        group_var = ["code_name", "hidden_units", "p_noise", "learning_rate", "epoch"]
        pdf = self.df.groupby(group_var).mean().reset_index()

        dev_all_sd = (
            alt.Chart(pdf)
            .mark_errorband(extent="stdev")
            .encode(y=alt.Y("score:Q", scale=alt.Scale(domain=(0, 1))), x="epoch:Q",)
            .properties(
                title="Developmental space: Mean Accuracy in all conditions over epoch"
            )
        )

        dev_all_m = dev_all_sd.mark_line().encode(y="mean(score):Q")

        this_plot = (dev_all_m + dev_all_sd) if show_sd else dev_all_m
        return this_plot

    def make_wnw(self):

        variates = ["hidden_units", "p_noise", "learning_rate"]

        df_wnw = self.df.loc[
            (self.df.cond.isin(["HF_INC", "NW_UN"])),
            variates + ["code_name", "epoch", "cond", "score"],
        ]

        df_wnw = df_wnw.pivot_table(
            index=variates + ["epoch", "code_name"], columns="cond"
        ).reset_index()

        df_wnw.columns = df_wnw.columns = [
            "".join(c).strip() for c in df_wnw.columns.values
        ]
        df_wnw.rename(
            columns={"scoreHF_INC": "word_acc", "scoreNW_UN": "nonword_acc",},
            inplace=True,
        )

        df_wnw["word_advantage"] = df_wnw.word_acc - df_wnw.nonword_acc
        return df_wnw

    def plot_wnw(self, mean=False):
        """ Performance space plot """
        df = self.make_wnw()

        if mean:
            df = df.groupby("epoch").mean().reset_index()

        wnw_line = (
            alt.Chart(df)
            .mark_line()
            .encode(
                y=alt.Y("nonword_acc:Q", scale=alt.Scale(domain=(0, 1))),
                x=alt.X("word_acc:Q", scale=alt.Scale(domain=(0, 1))),
                tooltip=["code_name", "epoch", "word_acc", "nonword_acc"],
                color="code_name:N",
            )
        )

        diagonal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(color="#D3D3D3")
            .encode(
                x=alt.X("x", axis=alt.Axis(title="word")),
                y=alt.X("y", axis=alt.Axis(title="nonword")),
            )
        )

        return (diagonal + wnw_line).properties(
            title="Performance space: Nonword accuracy vs. Word accuracy"
        )

    def stat_header(self):

        n = len(self.df.code_name.unique())

        t = [
            "Grand mean rank: " + self.get_rankpc_desc(),
            "Grand mean acc  : " + self.get_acc_desc(),
        ]

        return [f" (n={n})"] + t

    def plot(self, title=None, show_sd=True):
        """Plot all relevant stuffs"""

        if title is not None:
            t = [title] + self.stat_header()

        all_plot = (
            self.plot_control_space()
            & (
                self.plot_mean_development(show_sd=show_sd)
                | self.plot_all_cond_mean(show_sd=show_sd)
            )
        ).properties(title=t)

        return all_plot
    
