import pandas as pd


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
