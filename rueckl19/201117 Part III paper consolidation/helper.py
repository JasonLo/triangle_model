import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()

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


class SimResults:
    """ All sim results handler
    I: Selection:
    1. Control space h-param filter
    2. Control space region filter
    3. DVs Condition filter
    
    II: Plotting:
    1. Where are the selected model in the control space
    2. How's their average performance (in each cond / mean of all conds)
    3. Some basic descriptives in title
    """

    def __init__(self, df):
        self.df = df
        self._label_control_space()

        # Reuseable plotting element
        self.diagonal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(color="#D3D3D3")
            .encode(
                x=alt.X("x", axis=alt.Axis(title="Word")),
                y=alt.Y("y", axis=alt.Axis(title="Nonword")),
            )
        )

    def _label_control_space(self):

        # Cell label
        self.df["cell_code"] = (
            "h"
            + self.df.hidden_units.astype(str)
            + "_p"
            + self.df.p_noise.astype(str)
            + "_l"
            + self.df.learning_rate.astype(str)
            + "_c"
            + self.df.cleanup_units.astype(str)
        )

        self.df["risk_count"] = (
            (self.df.hidden_units < 100) * 1
            + (self.df.p_noise > 3) * 1
            + (self.df.learning_rate < 0.004) * 1
        )

        # Region label
        cond_list_p = [
            self.df.p_noise < 2,
            (self.df.p_noise >= 2) & (self.df.p_noise < 4),
            self.df.p_noise >= 4,
        ]

        cond_list_h = [
            self.df.hidden_units >= 250,
            (self.df.hidden_units < 250) & (self.df.hidden_units >= 100),
            self.df.hidden_units < 100,
        ]
        cond_list_e = [
            self.df.learning_rate >= 0.01,
            (self.df.learning_rate >= 0.004) & (self.df.learning_rate < 0.01),
            self.df.learning_rate < 0.004,
        ]
        choice_list = ["Good", "Base", "Bad"]

        self.df["control_region_p"] = np.select(cond_list_p, choice_list)
        self.df["control_region_h"] = np.select(cond_list_h, choice_list)
        self.df["control_region_e"] = np.select(cond_list_e, choice_list)

        self.df["control_region"] = (
            "p"
            + self.df.control_region_p
            + "_h"
            + self.df.control_region_h
            + "_e"
            + self.df.control_region_e
        )

    def count_model(self):
        return len(self.df.code_name.unique())

    def select_by_control(
        self,
        hidden_units=None,
        p_noise=None,
        learning_rate=None,
        cleanup_units=None,
        verbose=True,
    ):
        """Control space filter by h-params"""

        n_pre = self.count_model()
        if hidden_units is not None:
            self.df = self.df.loc[self.df.hidden_units.isin(hidden_units)]
        if p_noise is not None:
            self.df = self.df.loc[self.df.p_noise.isin(p_noise)]
        if learning_rate is not None:
            self.df = self.df.loc[self.df.learning_rate.isin(learning_rate)]
        if cleanup_units is not None:
            self.df = self.df.loc[self.df.cleanup_units.isin(cleanup_units)]

        n_post = self.count_model()

        if verbose:
            print(f"Selected {n_post} models from the original {n_pre} models")

    def select_by_region(self, region_name, verbose=True):
        """Control space filter by good/base/bad label"""

        n_pre = self.count_model()
        self.df = self.df.loc[self.df.control_region.isin(region_name)]
        n_post = self.count_model()
        if verbose:
            print(f"Selected {n_post} models from the original {n_pre} models")

    def select_by_cond(self, conds, verbose=True):
        """Filter DVs by condition"""

        n_pre = self.count_model()
        self.df = self.df.loc[self.df.cond.isin(conds)]
        n_post = self.count_model()

        if verbose:
            print(f"Selected {n_post} models from the original {n_pre} models")

    ### Plotting ###

    def plot_control_space(self, color="count(code_name)", with_cleanup=False):
        """Plot selected models at control space"""
        pdf = (
            self.df.groupby(["cell_code", "control_region", "code_name"])
            .mean()
            .round(3)
            .reset_index()
        )

        self.select_control_space = alt.selection(
            type="multi", on="click", empty="none", fields=["cell_code"],
        )

        control_space = (
            alt.Chart(pdf)
            .mark_rect(stroke="white", strokeWidth=2)
            .encode(
                x="p_noise:O",
                y=alt.Y("hidden_units:O", sort="descending"),
                column=alt.Column("learning_rate:O", sort="descending"),
                color=color,
                detail="cell_code",
                opacity=alt.condition(
                    self.select_control_space, alt.value(1), alt.value(0.2)
                ),
            )
            .add_selection(self.select_control_space)
        )

        if with_cleanup:
            control_space = (
                alt.Chart(pdf)
                .mark_rect(stroke="white", strokeWidth=2)
                .encode(
                    x="p_noise:O",
                    y=alt.Y("hidden_units:O", sort="descending"),
                    column=alt.Column("learning_rate:O", sort="descending"),
                    row=alt.Row("cleanup_units:O", sort="descending"),
                    color=color,
                    detail="cell_code",
                    opacity=alt.condition(
                        self.select_control_space, alt.value(1), alt.value(0.2)
                    ),
                )
                .add_selection(self.select_control_space)
            )
        else:
            control_space = (
                alt.Chart(pdf)
                .mark_rect(stroke="white", strokeWidth=2)
                .encode(
                    x="p_noise:O",
                    y=alt.Y("hidden_units:O", sort="descending"),
                    column=alt.Column("learning_rate:O", sort="descending"),
                    color=color,
                    detail="cell_code",
                    opacity=alt.condition(
                        self.select_control_space, alt.value(1), alt.value(0.2)
                    ),
                )
                .add_selection(self.select_control_space)
            )

        return control_space
    
    
    def _interactive_dev(self, show_sd, baseline=None):
        """Plot the mean development of all selected models"""
        development_space_mean = (
            alt.Chart(self.df)
            .mark_line()
            .encode(
                y=alt.Y("mean(score):Q", title="Accuracy", scale=alt.Scale(domain=(0, 1))),
                x=alt.X("epoch:Q", title="Sample (M)"),
                color=alt.Color(
                    "cond:N", legend=alt.Legend(orient='none', legendX=300, legendY=180), title="Condition"
                ),
            )
            .properties(title="Developmental space")
            .transform_filter(self.select_control_space)
        )

        if show_sd:
            development_space_sd = development_space_mean.mark_errorband(
                extent="stdev"
            ).encode(y=alt.Y("score:Q", title="Accuracy", scale=alt.Scale(domain=(0, 1))))
            development_space_mean += development_space_sd

        if baseline is not None:
            development_space_mean += baseline

        return development_space_mean
    
    def make_wnw(self):
        """ Averaged word vs. nonword over epoch
        """

        variates = ["hidden_units", "p_noise", "learning_rate"]

        df_wnw = self.df.loc[
            (self.df.cond.isin(["HF_INC", "NW_UN"])),
            variates + ["code_name", "cell_code", "epoch", "cond", "score"],
        ]

        df_wnw = df_wnw.pivot_table(
            index=variates + ["epoch", "code_name", "cell_code"], columns="cond"
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

    def _interactive_wnw(self, baseline=None):
        """ Private function for interactive plot: Performance space plot """
        df = self.make_wnw()

        base_wnw = (
            alt.Chart(df)
            .mark_line(color="black")
            .encode(
                y=alt.Y("mean_nw:Q", scale=alt.Scale(domain=(0, 1)), title="Nonword"),
                x=alt.X("mean_w:Q", scale=alt.Scale(domain=(0, 1)), title="Word"),
                tooltip=["epoch", "mean_w:Q", "mean_nw:Q"],
            )
            .transform_filter(self.select_control_space)
            .transform_aggregate(
                mean_w="mean(word_acc)", mean_nw="mean(nonword_acc)", groupby=["epoch"]
            )
            .transform_calculate(
                color="if(datum.epoch===0.05, 'red', if(datum.epoch === 0.3, 'green', ''))"
            )
        )

        points = (
            base_wnw.mark_circle(size=200)
            .encode(color=alt.Color("color:N", scale=None))
            .add_selection(alt.selection_single())
        )

        base_wnw += points

        if baseline is not None:
            base_wnw += baseline

        return (self.diagonal + base_wnw).properties(
            title="Performance space (Nonword vs. Word)"
        )

    def plot_mean_dev(self, show_sd, by_cond=True, interactive=False, baseline=None):
        """Plot the mean development of all selected models
        interactive = True for plot_interactive() ONLY!
        baseline: Overlay a baseline plot
        show_sd: Show SD in plot or not
        by_cond: True: plot condition in separate line; False: Aggregate condition before plotting
        """
        
        if by_cond:
            pdf = self.df
            
        else:
            group_var = ["code_name", "hidden_units", "p_noise", "learning_rate", "epoch"]
            pdf = self.df.groupby(group_var).mean().reset_index()
        
        
        development_space_mean = (
            alt.Chart(pdf)
            .mark_line()
            .encode(
                x="epoch:Q", 
                y=alt.Y("mean(score):Q", scale=alt.Scale(domain=(0, 1)))
            )
        )

        development_space_sd = (
            alt.Chart(pdf)
            .mark_errorband(extent="stdev")
            .encode(y=alt.Y("score:Q", scale=alt.Scale(domain=(0, 1))), x="epoch:Q",)
        )

        if by_cond:
            development_space_sd = development_space_sd.encode(
                color=alt.Color("cond:N", legend=alt.Legend(legendX=300, legendY=180), title="Condition")
                
            )

        if interactive:
            development_space_sd = development_space_sd.transform_filter(
                self.select_control_space
            )

        # Add Mean
        development_space_mean = development_space_sd.mark_line().encode(
            y="mean(score):Q"
        )

        if show_sd:
            development_space_mean += development_space_sd

        if baseline is not None:
            development_space_mean += baseline

        return development_space_mean

    def plot_mean_wnw(self, baseline=None):
        """Plot all perforamance space only"""

        df = self.make_wnw()
        df = df.groupby("epoch").mean().reset_index()

        base_wnw = (
            alt.Chart(df)
            .mark_line(color="black")
            .encode(
                y=alt.Y("nonword_acc:Q", scale=alt.Scale(domain=(0, 1))),
                x=alt.X("word_acc:Q", scale=alt.Scale(domain=(0, 1))),
                tooltip=["epoch", "word_acc:Q", "nonword_acc:Q"],
            )
            .transform_calculate(
                color="if(datum.epoch===0.05, 'red', if(datum.epoch === 0.3, 'green', ''))"
            )
        )

        points = base_wnw.mark_circle(size=200).encode(
            color=alt.Color("color:N", scale=None)
        )

        base_wnw += points

        if baseline is not None:
            base_wnw += baseline

        base_wnw += self.diagonal

        return base_wnw

    def plot_interactive(self, title=None, show_sd=True, base_dev=None, base_wnw=None):
        """Plot averaged developmental and performance space + interactive control space selection"""

        all_plot = (
            self.plot_control_space()
            & (
                self._interactive_dev(show_sd=show_sd, baseline=base_dev)
                | self._interactive_wnw(baseline=base_wnw)
            )
        ).properties(title=f"{title} (n = {len(self.df.code_name.unique())})")

        return all_plot

    def plot_heatmap_wadv(self, mode="dev"):
        """Plot word advantage heatmap
        mode: (dev)elopment or (per)formance
        """
        assert mode == "per" or mode == "dev"

        if mode == "dev":
            x = "epoch:O"

        if mode == "per":
            x = alt.X("word_acc:Q", bin=alt.Bin(maxbins=20))

        df = self.make_wnw()

        plot = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=x,
                y=alt.Y("hidden_units:O", sort="descending"),
                row=alt.Row("learning_rate:O", sort="descending"),
                column="p_noise:O",
                tooltip=["epoch", "word_acc:Q", "nonword_acc:Q"],
                color=alt.Color(
                    "word_advantage",
                    scale=alt.Scale(scheme="redyellowgreen", domain=(-0.3, 0.3)),
                ),
            )
        )

        return plot

    def plot_performance_multiline(self, var, color="blues"):

        df = self.make_wnw()
        df = df.groupby(["epoch"] + [var]).mean().reset_index()

        sel = alt.selection(type="multi", on="click", fields=[var])

        plot = (
            alt.Chart(df)
            .mark_line()
            .encode(
                y=alt.Y("nonword_acc:Q", scale=alt.Scale(domain=(0, 1))),
                x=alt.X("word_acc:Q", scale=alt.Scale(domain=(0, 1))),
                color=alt.Color(var, type="ordinal", scale=alt.Scale(scheme=color)),
                opacity=alt.condition(sel, alt.value(1), alt.value(0.1)),
                tooltip=[var, "epoch", "word_acc", "nonword_acc"],
            )
            .add_selection(sel)
        )

        return self.diagonal + plot

    
class RDGrouping(SimResults):
    def __init__(self, main_df, baseline_df):
        super().__init__(main_df)

        # Criterion measures
        self.include_conds = ["HF_CON", "HF_INC", "LF_CON", "LF_INC"]

        self.df = main_df
        self.td_df = baseline_df
        self.td_stat = self.get_stat(self.td_df)

        # Make all dynamic data files
        self._make_dfs()

    def _make_dfs(self):
        self.cadf = self._make_cadf()
        self.zdf = self._make_zdf(self.cadf)
        self.pcdf = self._make_pcdf(self.cadf)
        self.mzdf = self._melt_zdf(self.zdf)
        self.mpcdf = self._melt_pcdf(self.pcdf)
        
    def get_stat(self, df):
        """Baseline statistics
        Return mean and sd by epoch in word 
        """
        return (
            df.loc[df.cond.isin(self.include_conds),]
            .groupby(["code_name", "epoch"])
            .mean()
            .reset_index()
            .groupby(["epoch"])
            .agg(["mean", "std"])
            .score.reset_index()
        ).to_dict()

    def _reduce_epoch_resolution(self, df):
        sel_epoch = [0.01, 0.03, 0.05, 0.07, 0.09, 0.2, 0.4, 0.6, 0.8, 1.0]
        return df.loc[
            df.epoch.isin(sel_epoch),
        ]

    def _calcuate_z_deviance(self, row):
        """Calcuate z score relative to TD at each epoch
        """
        m = self.td_stat["mean"][row["epoch_idx"]]
        sd = self.td_stat["std"][row["epoch_idx"]]

        # Avoid zero division
        if sd == 0:
            sd = 1e-6

        return (row["score"] - m) / sd

    def _calcuate_percetage_of_baseline(self, row):
        """Calcuate % relative to TD at each epoch
        """
        m = self.td_stat["mean"][row["epoch_idx"]]

        return row["score"] / m

    def _make_cadf(self):
        """Make condition avergage df (aggregate cond) with reduced epoch resolution"""
        cadf = (
            self.df.loc[self.df.cond.isin(self.include_conds),]
            .groupby(["code_name", "epoch"])
            .mean()
            .reset_index()
        )
        cadf["learning_rate"] = round(cadf.learning_rate, 4)
        cadf["epoch_idx"] = cadf.apply(
            lambda x: x.epoch * 100 if x.epoch <= 0.1 else x.epoch * 10 + 9, axis=1
        )
        cadf["epoch_idx"] = cadf.epoch_idx.astype(int)
        return self._reduce_epoch_resolution(cadf)

    def _make_pcdf(self, cadf):
        """ Make percetange based df"""
        cadf["pc"] = cadf.apply(self._calcuate_percetage_of_baseline, axis=1)

        # Different cutoff of RDs by percentage (0 = RD, 1 = TD)
        cadf["pc_group_50"] = 1 * (cadf.pc > 0.50)
        cadf["pc_group_55"] = 1 * (cadf.pc > 0.55)
        cadf["pc_group_60"] = 1 * (cadf.pc > 0.60)
        cadf["pc_group_65"] = 1 * (cadf.pc > 0.65)
        cadf["pc_group_70"] = 1 * (cadf.pc > 0.70)
        cadf["pc_group_75"] = 1 * (cadf.pc > 0.75)
        cadf["pc_group_80"] = 1 * (cadf.pc > 0.80)
        cadf["pc_group_85"] = 1 * (cadf.pc > 0.85)
        cadf["pc_group_90"] = 1 * (cadf.pc > 0.90)

        return cadf

    def _make_zdf(self, cadf):
        """Make z-score based df"""

        cadf["z_deviance"] = cadf.apply(self._calcuate_z_deviance, axis=1)

        # Different cutoff of RDs (0 = RD, 1 = TD)
        cadf["group_10"] = 1 * (cadf.z_deviance > -1.0)
        cadf["group_11"] = 1 * (cadf.z_deviance > -1.1)
        cadf["group_12"] = 1 * (cadf.z_deviance > -1.2)
        cadf["group_13"] = 1 * (cadf.z_deviance > -1.3)
        cadf["group_14"] = 1 * (cadf.z_deviance > -1.4)
        cadf["group_15"] = 1 * (cadf.z_deviance > -1.5)
        cadf["group_16"] = 1 * (cadf.z_deviance > -1.6)
        cadf["group_17"] = 1 * (cadf.z_deviance > -1.7)
        cadf["group_18"] = 1 * (cadf.z_deviance > -1.8)
        cadf["group_19"] = 1 * (cadf.z_deviance > -1.9)
        cadf["group_20"] = 1 * (cadf.z_deviance > -2.0)
        cadf["group_21"] = 1 * (cadf.z_deviance > -2.1)
        cadf["group_22"] = 1 * (cadf.z_deviance > -2.2)
        cadf["group_23"] = 1 * (cadf.z_deviance > -2.3)
        cadf["group_24"] = 1 * (cadf.z_deviance > -2.4)
        cadf["group_25"] = 1 * (cadf.z_deviance > -2.5)
        cadf["group_26"] = 1 * (cadf.z_deviance > -2.6)
        cadf["group_27"] = 1 * (cadf.z_deviance > -2.7)
        cadf["group_28"] = 1 * (cadf.z_deviance > -2.8)
        cadf["group_29"] = 1 * (cadf.z_deviance > -2.9)
        cadf["group_30"] = 1 * (cadf.z_deviance > -3.0)

        return cadf

    def _melt_pcdf(self, df):

        mdf = df.melt(
            id_vars=[
                "code_name",
                "epoch",
                "hidden_units",
                "cleanup_units",
                "p_noise",
                "learning_rate",
            ],
            value_vars=[
                "pc_group_50",
                "pc_group_55",
                "pc_group_60",
                "pc_group_65",
                "pc_group_70",
                "pc_group_75",
                "pc_group_80",
                "pc_group_85",
                "pc_group_90",
            ],
        )

        mdf["cutoff"] = mdf.variable.str[-2:].astype(float)

        return mdf

    def _melt_zdf(self, df):

        mdf = df.melt(
            id_vars=[
                "code_name",
                "epoch",
                "hidden_units",
                "cleanup_units",
                "p_noise",
                "learning_rate",
            ],
            value_vars=[
                "group_10",
                "group_11",
                "group_12",
                "group_13",
                "group_14",
                "group_15",
                "group_16",
                "group_17",
                "group_18",
                "group_19",
                "group_20",
                "group_21",
                "group_22",
                "group_23",
                "group_24",
                "group_25",
                "group_26",
                "group_27",
                "group_28",
                "group_29",
                "group_30",
            ],
        )

        mdf["cutoff"] = mdf.variable.str[-2:].astype(float) / 10

        return mdf

    def plot_heatmap(self, var):
        """Z-score deviance over epoch"""
        if var == "z_deviance":
            domain = (-5, 5)
            df = self.zdf
        elif var == "score":
            domain = (0, 1)
            df = self.zdf
        elif var == "pc":
            domain = (0, 1)
            df = self.pcdf

        mean_var = f"mean({var})"

        hm = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x="p_noise:O",
                y=alt.Y("hidden_units:O", sort="descending"),
                row=alt.Column("learning_rate:O", sort="descending"),
                column="epoch:O",
                color=alt.Color(
                    mean_var, scale=alt.Scale(domain=domain, scheme="redyellowgreen"),
                ),
                tooltip=["mean(score)", mean_var],
            )
        )

        return hm

    def _get_acc_cut(self, epoch, xsd, cond=None):
        """Get accuracy cut off value with reference to xsd below mean of TD
        td_df: data file of typically developing readers (created by Select_Model().df)
        epoch: at what epoch to classify RD [list]
        xsd: how many sd below mean of TD
        cond: include what condition, default = all conditions (no filtering)
        """

        sel = (
            self.td_df.loc[self.td_df.epoch.isin(epoch)]
            if (cond is None)
            else self.td_df.loc[
                self.td_df.epoch.isin(epoch) & self.td_df.cond.isin(cond)
            ]
        )

        stat = sel.groupby("code_name").mean().score.agg(["mean", "std"])
        return stat["mean"] - xsd * stat["std"]

    def select_by_relative_sd(self, epoch, xsd, cond=None):
        """Select the models that has at least
        X SD <xsd> below mean of TD at <epoch>"""

        tmp = (
            self.df.loc[self.df.epoch.isin(epoch)]
            if (cond is None)
            else self.df.loc[self.df.epoch.isin(epoch) & self.df.cond.isin(cond)]
        )

        mean_tmp = tmp.groupby("code_name").mean().reset_index()
        sel = mean_tmp.loc[mean_tmp.score < self._get_acc_cut(epoch, xsd, cond)]
        self.df = self.df.loc[self.df.code_name.isin(sel["code_name"])]

        # remake dfs
        self._make_dfs()

    def plot_interactive_group_heatmap(self, version="pc"):
        """ Plot interactive grouping heatmap
        version: "pc" percentage definition
                 "z" z-score definition
        """
        assert (version == "pc") or (version == "z")
        if version == "pc":
            use_df = self.mpcdf
            slider = alt.binding_range(
                min=50.0, max=90.0, step=5.0, name="percentage cutoff:"
            )
            selector = alt.selection_single(
                name="SelectorName",
                fields=["cutoff"],
                bind=slider,
                init={"cutoff": 80.0},
            )
        else:
            use_df = self.mzdf
            slider = alt.binding_range(min=1.0, max=3.0, step=0.1, name="z cutoff:")
            selector = alt.selection_single(
                name="SelectorName",
                fields=["cutoff"],
                bind=slider,
                init={"cutoff": 2.0},
            )

        df = (
            use_df.groupby(
                ["hidden_units", "p_noise", "learning_rate", "epoch", "cutoff"]
            )
            .mean()
            .reset_index()
        )

        interactive_group_heatmap = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x="p_noise:O",
                y=alt.Y("hidden_units:O", sort="descending"),
                row=alt.Column("learning_rate:O", sort="descending"),
                column="epoch:O",
                color=alt.Color(
                    "value", scale=alt.Scale(domain=(0, 1), scheme="redyellowgreen"),
                ),
            )
            .add_selection(selector)
            .transform_filter(selector)
        )

        return interactive_group_heatmap
