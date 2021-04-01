#%% 
import data_wrangling
import pandas as pd
import altair as alt
# %% Load test sets from file
d = data_wrangling.MyData()
d.load_testsets()

# %% A function to get number of features
def get_n_features(word):
    if word in d.testsets['train']['item']:
        idx = d.testsets['train']['item'].index(word)
        semantic_vector = d.testsets['train']['sem'][idx,:]
        return(sum(semantic_vector)) 


# %% Parse Cortese
cortese = pd.read_csv(
    "../preprocessing/cortese2004norms.csv", skiprows=9, na_filter=False
)
cortese = cortese[["item", "rating"]]
cortese = cortese.rename({"item": "word", "rating":"img"}, axis=1)
cortese["n_features"] = cortese.word.apply(get_n_features)


# %% Plot
p = alt.Chart(cortese).mark_point().encode(x="img", y="n_features", tooltip=["word", "img", "n_features"])
p + p.transform_regression('img', 'n_features').mark_line(color="red")

# %% Pearson's r
cortese.corr()

