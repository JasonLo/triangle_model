import streamlit as st
import altair as alt
from google.cloud import bigquery
client = bigquery.Client(location="US", project="mimetic-core-276919")

# Streamlit setting
alt.data_transformers.disable_max_rows()
st.set_page_config(layout="wide")

# Layout
st.title("Taraban")

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
def load_raw_data(source):
    """Read averaged data from batch results database"""

    query = f"""
    SELECT 
        code_name, 
        epoch, 
        timetick, 
        y, 
        testset, 
        AVG(acc) as acc, 
        AVG(sse) as sse, 
        AVG(conditional_sse) as conditional_sse
    FROM {source}
    GROUP BY
        code_name, 
        epoch,
        timetick,
        y,
        testset
    """
    query_job = client.query(query)
    df = query_job.to_dataframe()
 
    max_epoch = df.epoch.max()

    return df, max_epoch

df, max_epoch = load_raw_data("triangle_oral.taraban")

# Interactive sidebar

st.sidebar.header("Select inputs")


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
]

df_filteded_ignore_epoch = df.loc[
    (df.timetick.isin(sel_timetick))
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
    """Plot 3-ways interaction barchart"""
    
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
        f"{sel_measurement} ~ freq * cons * img", 
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
            "Accuracy ~ Frequency x Consistency x Imageability"
            pho_summary.tables[1]

        with right_col:
            "Sum squared error ~ Frequency x Consistency x Imagebility"
            sem_summary.tables[1]

        with st.beta_expander("More details", expanded=False):
            """
            Mean value in each condition is used in the linear mixed effect model (LME),
            model ID is set as random effect (Groups)
            """
            detail_left, detail_right = st.beta_columns(2)
            detail_left.write(pho_summary.tables[0])
            detail_right.write(sem_summary.tables[0])


