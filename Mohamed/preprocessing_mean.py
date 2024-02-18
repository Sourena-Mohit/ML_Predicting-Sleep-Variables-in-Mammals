
#%% import libraries
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler

#%% import file
file = "data/Input/sleep_merged.csv"
data = pd.read_csv(file)
df = copy.deepcopy(data)
#%% vore encoding
df = pd.get_dummies(df, columns=["Vore"], prefix=["Vore"], drop_first=True)
df[["Vore_insecti","Vore_herbi","Vore_omni"]] = df[["Vore_insecti","Vore_herbi","Vore_omni"]].astype(int)
df["Vore"] = data["Vore"]

#%% predation exposure danger 
df["Predation"] = df["Predation"].fillna(df.groupby("Order")["Predation"].transform("mean").round())
df["Exposure"] = df["Exposure"].fillna(df.groupby("Order")["Exposure"].transform("mean").round())
df["Danger"] = df["Danger"].fillna(df.groupby("Order")["Danger"].transform("mean").round())

df["Predation"] = df["Predation"].fillna(df.groupby("Vore")["Predation"].transform("mean").round())
df["Exposure"] = df["Exposure"].fillna(df.groupby("Vore")["Exposure"].transform("mean").round())
df["Danger"] = df["Danger"].fillna(df.groupby("Vore")["Danger"].transform("mean").round())

#%% brainwt


## BodyMass to BrainMass Ratio is , on average in mammals, 1% (with some exception)
#bodybrain = 0.01 * df["BodyWt"]
#df.loc[df["BrainWt"] == 0, "BrainWt"] = bodybrain[df["BrainWt"] == 0]
df["BrainWt"] = df["BrainWt"].replace(0, np.nan)

mean_brainwt = df["BrainWt"].mode()
df["BrainWt"] = df["BrainWt"].fillna(mean_brainwt)
print("nb of NA", df["BrainWt"].isna().sum())
# df["BrainWt"]  = df["BrainWt"].replace(np.nan, mean_brainwt)

#%% Gestation, Lifespan

mean_gestation = df.loc[df["Gestation"] != np.nan , "Gestation"].mean().round(2)
print(mean_gestation)
df["Gestation"]  = df["Gestation"].replace(np.nan, mean_gestation)

mean_lifespan  = df["LifeSpan"].mean().round(3)
print(mean_lifespan)
df["LifeSpan"] = df["LifeSpan"].fillna(mean_lifespan)

#%% dreaming
mean_dreaming = df.loc[df["Dreaming"] != np.nan, "Dreaming"].mean().round(2)
print(mean_dreaming)
df["Dreaming"] = df["Dreaming"].replace(np.nan, mean_dreaming)

#%% Nondreaming
mean_nondreaming = df.loc[df["NonDreaming"] != np.nan, "NonDreaming"].mean().round(2)
print(mean_nondreaming)
df["NonDreaming"] = df["NonDreaming"].replace(np.nan, mean_nondreaming)

#%% first saving before standard
df["BrainBodyRatio"] = df["BrainWt"] / df["BodyWt"]

df = df.sort_values(by="Species", ascending=True)
df.to_csv("data/Output/cleaned_sleep_merged_mean.csv", index=False)
#%% Standardization
dff = copy.deepcopy(df)
# I will standardize rather than normalize
col_std = ['BodyWt', 'BrainWt', 'TotalSleep', 'Awake', 'NonDreaming', 'Dreaming', 'LifeSpan', 'Gestation']

scaler = StandardScaler()

for col in col_std:
    dff[col] = scaler.fit_transform(dff[col].values.reshape(-1, 1))

df = dff.copy()
#%% save file
    
df = df.sort_values(by="Species", ascending=True)
df.to_csv("data/Output/cleaned_sleep_merged_mean.csv", index=False)
# %%
