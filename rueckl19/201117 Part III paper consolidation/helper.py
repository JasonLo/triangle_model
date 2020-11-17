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
