import pandas as pd

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

    
def get_rank(df):
    """Grand mean based rank pc
    """
    gacc = df.groupby("code_name", as_index=False).mean()
    gacc = gacc[["code_name", "score"]]
    gacc["rank_pc"] = gacc.score.rank(pct=True)
    return df.merge(gacc[["code_name", "rank_pc"]], how="left")

