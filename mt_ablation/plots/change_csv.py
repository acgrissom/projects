import pandas as pd
import pandasql as sql
import re
import numpy as np
case_col = {"case" : []}
shuf_col = {"shuffled" : []}
df = pd.read_csv("model_scores.csv")
df = df.assign(case = '')
df = df.assign(shuffled = '')
pert_col = df["perturbation"]
#control_match = pert_col.str.contains(r'control')
#df.loc[control_match, "case"] = "control"
#df.loc[df.perturbtion.str.contains('control'), 'case'] = control
#df.loc[df.perturbtion.str.contains('control'), 'case'] = perturbation
df['case'] = np.where(~df.perturbation.str.contains('control'), 'perturbation', 'perturbation')
df['case'] = np.where(df.perturbation.str.contains('control'), 'control', 'perturbation')
df['perturbation'] = df['perturbation'].str.replace("control","")
df['perturbation'] = df['perturbation'].str.strip()


df.shuffled = np.where(df.model_type.str.contains('non-shuffled'), 'non-shuffled', 'shuffled')
df['model_type'] = df['model_type'].str.replace('non-shuffled-','')
df['model_type'] = df['model_type'].str.replace('-non-shuffled','')
df['model_type'] = df['model_type'].str.replace('shuffled-','')
df['model_type'] = df['model_type'].str.replace('-shuffled','')

df['model_type'] = df['model_type'].str.replace('--','-')
df['model_type'] = df['model_type'].str.strip()


#df['model_type'] = df['model_type'].apply(lambda x : x.replace("control","").strip())

#control_df = df[control_match]
#control_df = "control"

#print(control_df.head())
print(df.head())

df.to_csv("./new.csv")

