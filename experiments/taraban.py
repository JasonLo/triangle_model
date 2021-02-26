import streamlit as st
import sqlite3
import pandas as pd
import altair as alt
import statsmodels.formula.api as smf

# Streamlit setting
alt.data_transformers.disable_max_rows()
st.set_page_config(layout="wide")

# Layout
st.title("Examine 2 ways interaction in Taraban")

with st.beta_expander("More details", expanded=False):
    """
    A total of 20 models were ran in this batch.
    Main control parameter of interest: error_injection_ticks
    - 2 ticks (11-12) (n=10)
    - 11 ticks (2-12) (n=10)

    Other control parameter settings:
    - "ort_units": 119,
    - "pho_units": 250,
    - "sem_units": 2446,
    - "hidden_os_units": 500,
    - "hidden_op_units": 100,
    - "hidden_ps_units": 500,
    - "hidden_sp_units": 500,
    - "pho_cleanup_units": 20,
    - "sem_cleanup_units": 50,
    - "pho_noise_level": 0.0,
    - "sem_noise_level": 0.0,
    - "activation": "sigmoid",
    - "tau": 1 / 3,
    - "max_unit_time": 4.0,
    - "learning_rate": 0.01,
    - "zero_error_radius": 0.1,
    - "n_mil_sample": 2.0,
    - "batch_size": 100,

    Mean accuracy in each condition was calculated within each model
    Confidence intervel was calculated across model
    """

developmental_plot_container = st.beta_container()
bar_plot_container = st.beta_container()
sel_run_lme = st.checkbox("Run LME?")
stat_container = st.beta_container()

# Load data
@st.cache
def load_raw_data(batch_name):
    """Read averaged data from batch results database"""

    con = sqlite3.connect(
        f"/home/jupyter/tf/models/batch_run/{batch_name}/batch_results.sqlite"
    )

    query = """
    SELECT 
        code_name, 
        epoch, 
        timetick, 
        y, 
        testset, 
        AVG(acc) as acc, 
        AVG(sse) as sse, 
        AVG(conditional_sse) as conditional_sse
    FROM taraban
    WHERE testset IN ('taraban_hf-exc', 'taraban_hf-reg-inc', 'taraban_lf-exc', 'taraban_lf-reg-inc')
    GROUP BY
        code_name, 
        epoch,
        timetick,
        y,
        testset
    """
    df = pd.read_sql_query(query, con)
    batch_config = pd.read_sql_query("SELECT * FROM batch_config", con)
    df = df.merge(
        batch_config[["code_name", "inject_error_ticks"]], "left", on="code_name"
    )
    df[["testset_name", "condition"]] = df.testset.str.split("_", expand=True)
    df['freq'] = df.condition.str.slice(0, 2)
    df['cons'] = df.condition.str.slice(3)

    max_epoch = df.epoch.max()

    return df, max_epoch

df, max_epoch = load_raw_data("error_injection_timing_test")

# Interactive sidebar

st.sidebar.header("Select inputs")
sel_error_ticks = st.sidebar.radio("How many error injection ticks", (2, 11))

sel_epoch = st.sidebar.select_slider(
    "Select epoch",
    list(range(10)) + list(range(10, max_epoch + 1, 10)),
    value=max_epoch,
)

sel_timetick = st.sidebar.slider(
    "Select timeticks range (average between)", 2, 12, value=(11, 12)
)

sel_measurement = st.sidebar.radio(
    "Select dependent variable", ("acc", "sse", "conditional_sse")
)

# Filter data
df_filtered = df.loc[
    (df.epoch == sel_epoch)
    & (df.timetick.isin(sel_timetick))
    & (df.inject_error_ticks == sel_error_ticks)
]

df_filteded_ignore_epoch = df.loc[
    (df.timetick.isin(sel_timetick))
    & (df.inject_error_ticks == sel_error_ticks)
]

# Plot

if not (len(df_filteded_ignore_epoch) > 0):
    st.warning("No data in selected range")
    st.stop()

def developmental_plot(output):
    """ Plot developmental
    Since developmental plot will ignore, only use sel_timetick, sel_error_ticks
    """
    plot_df = df_filteded_ignore_epoch.loc[df_filteded_ignore_epoch.y==output]

    legend_selection = alt.selection_multi(fields=["testset"], bind="legend")

    plot = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x="epoch:Q",
            y=f"mean(acc):Q",
            color="testset",
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0)),
        )
        .add_selection(legend_selection)
        .properties(title=f"{output.upper()} output")
    )

    return plot


with developmental_plot_container:
    st.header("Mean accuracy by condition over development")
    "*Ignore epoch & DV selection"

    left_col, right_col = st.beta_columns(2)
    left_col.write(developmental_plot("pho"))
    right_col.write(developmental_plot("sem"))


if not (len(df_filtered) > 0):
    st.warning("No data in selected range")
    st.stop()


def my_barchart(output):
    """Plot 2-ways interaction barchart"""
    
    plot_df = df_filtered.loc[df_filtered.y==output]
    error_bar = (
        alt.Chart(plot_df, width=400)
        .mark_errorbar(extent="ci")
        .encode(
            x="testset:N", y=alt.Y(f"{sel_measurement}:Q", scale=alt.Scale(domain=(0, y_max)))
        )
        .properties(title=f"{output.upper()} output")
    )
    bar = error_bar.mark_bar().encode(y=f"mean({sel_measurement}):Q", color="testset")
    plot = (bar + error_bar).properties(title=f"{output.upper()}")
    return plot

with bar_plot_container:
    st.header("Mean (CI) ACC and SSE in frequency by consistency")

    if sel_measurement=='acc':
        y_max = 1
    else:
        y_max = st.number_input("Maximum y value on SSE plots", value=1.0, step=0.01)

    left_col, right_col = st.beta_columns(2)
    left_col.write(my_barchart("pho"))
    right_col.write(my_barchart("sem"))

# Run LME

def run_lme(output):
    lme_df = df_filtered.loc[df_filtered.y==output]
    model = smf.mixedlm(
        f"{sel_measurement} ~ freq * cons", 
        lme_df, 
        groups=lme_df["code_name"]
    )
    fit = model.fit()
    return fit.summary()

if sel_run_lme:

    with stat_container:
        st.header("Linear mixed effect model")
        left_col, right_col = st.beta_columns(2)

        pho_summary = run_lme('pho')
        sem_summary = run_lme('sem')

        with left_col:
            "DV in PHO ~ Frequency x Consistency"
            pho_summary.tables[1]

        with right_col:
            "DV in SEM ~ Frequency x Consistency"
            sem_summary.tables[1]

        with st.beta_expander("More details", expanded=False):
            """
            Mean value in each condition is used in the linear mixed effect model (LME),
            model ID is set as random effect (Groups)
            """
            detail_left, detail_right = st.beta_columns(2)
            detail_left.write(pho_summary.tables[0])
            detail_right.write(sem_summary.tables[0])


