#%%
import pandas as pd

#%% Read raw corpus
df1 = pd.read_csv("6kdict", header=None, names=['word', 'op', 'p'], sep='\t')
chang_words = df1.word

df2 = pd.read_csv("../../dataset/df_train.csv")
zevin_words = df2.word


#%% Is all Zevin words in Chang words? Yes
all(zevin_words.isin(chang_words))

#%% What is not in Zevin words. Mainly weird low frequency words
ooz_word = chang_words[~chang_words.isin(zevin_words)]
ooz_word.to_csv("ooz_word.csv")


#%% OOZ word probability?
df1.loc[~chang_words.isin(zevin_words)].sort_values("p")

# %% Do Chang clip their sampling p? Yes
df1.p.describe()

#%% Is high probablity word in Chang paper having high WF in Zevin dict?
max_p_words = df1.loc[df1.p==1, "word"]
df2.loc[df2.word.isin(max_p_words), ("word", "wf")].sort_values("wf")
