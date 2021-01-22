#%% Packing pkey to json
import pandas as pd
import json

p_file="/home/jupyter/tf/dataset/mappingv2.txt"
mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
m_dict = mapping.set_index(0).T.to_dict('list')


json_file = "/home/jupyter/tf/dataset/pho_key.dict"
with open(json_file, "w") as f:
    json.dump(m_dict, f)


with open("/home/jupyter/tf/dataset/pho_key.dict", "r") as f:
    test_load_pkey = json.load(f)