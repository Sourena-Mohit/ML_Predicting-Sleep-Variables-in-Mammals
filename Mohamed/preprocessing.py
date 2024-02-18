import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
 
file = "data/Input/sleep_merged.csv"

data = pd.read_csv(file)

df = copy.deepcopy(data)

## data processing

#We one-hot encode the feature "Vore"
df = pd.get_dummies(df, columns=["Vore"], prefix=["Vore"], drop_first=True)
df[["Vore_insecti","Vore_herbi","Vore_omni"]] = df[["Vore_insecti","Vore_herbi","Vore_omni"]].astype(int)
df["Vore"] = data["Vore"]

#%% predation, exposure, danger
#Mean imputation of the features by grouping them in Order and in Vore for meaningful imputation
df["Predation"] = df["Predation"].fillna(df.groupby("Order")["Predation"].transform("mean").round())
df["Exposure"] = df["Exposure"].fillna(df.groupby("Order")["Exposure"].transform("mean").round())
df["Danger"] = df["Danger"].fillna(df.groupby("Order")["Danger"].transform("mean").round())

df["Predation"] = df["Predation"].fillna(df.groupby("Vore")["Predation"].transform("mean").round())
df["Exposure"] = df["Exposure"].fillna(df.groupby("Vore")["Exposure"].transform("mean").round())
df["Danger"] = df["Danger"].fillna(df.groupby("Vore")["Danger"].transform("mean").round())

#%% Imputation result
#create a dictionary that contains the report for all the performance of the next regression imputations
imputation_results = {}

#%% brain weight
df["BrainBodyRatioNa"] = df["BrainWt"] / df["BodyWt"]

df["BrainWt"].mode()
#Replace 0 values of brain weight by nan
df["BrainWt"] = df["BrainWt"].replace(0, np.nan)
#Remove NaN rows
df_model = df[["BodyWt","BrainWt"]].dropna()
#Reshape feature to the right shape for processing
feature = df_model["BodyWt"].values.reshape(-1,1)
target = df_model["BrainWt"].values.reshape(-1,1)
#split data
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size = 0.1, random_state = 11)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Getting the mse and r2 to test for performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#saving into result report
imputation_results["BrainWt"] = {"MSE": mse, "R2": r2}


df_model_null = df[df["BrainWt"].isnull()]
df_model_complete = df[-df["BrainWt"].isnull()]

model = LinearRegression()

# Train the model using the complete data
X = df_model_complete["BodyWt"].values.reshape(-1, 1)
y = df_model_complete["BrainWt"].values
model.fit(X, y)

# Use the trained model to predict "BrainWt" for the missing values
X_predict = df_model_null["BodyWt"].values.reshape(-1, 1)
y_pred = model.predict(X_predict)
y_pred = np.maximum(y_pred, 0)

#rounding
y_pred = np.round(y_pred,2)
# Replace the missing "BrainWt" values with the predicted values
df.loc[df["BrainWt"].isnull(), "BrainWt"] = y_pred

df["BrainBodyRatioNoNa"] = df["BrainWt"] / df["BodyWt"]

#%% LifeSpan & Gestation

#%%% Lifespan1
# We going to fill some LifeSpan rows where gestation is available
df_model = df.dropna()
df_model_features = df_model[["BrainWt", "TotalSleep","Gestation"]]
df_model_target = df_model[["LifeSpan"]]

X_train, X_test, y_train, y_test = train_test_split(df_model_features, df_model_target, test_size = 0.1, random_state = 11)
y_train = y_train.values.reshape(-1,1)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Getting the mse and r2 to test for performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

imputation_results["Lifespan1"] = {"MSE": mse, "R2": r2}

#imputation

df_model_features_test = df[df["LifeSpan"].isnull() & -df["Gestation"].isnull()]
X_test = df_model_features_test[["BrainWt","TotalSleep","Gestation"]]

y_pred = model.predict(X_test)
df.loc[df["LifeSpan"].isnull() & -df["Gestation"].isnull(), "LifeSpan"] = y_pred
df["LifeSpan"] = np.round(df["LifeSpan"],1)


#%%% Gestation
# We going to FULLY fill out Gestation missing values through regression imputation
df_model = df.dropna()
df_model_features = df_model[["BodyWt","BrainWt", "TotalSleep"]]
df_model_target = df_model[["Gestation"]]

X_train, X_test, y_train, y_test = train_test_split(df_model_features, df_model_target, test_size = 0.1, random_state = 11)
y_train = y_train.values.reshape(-1,1)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Getting the mse and r2 to test for performance

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

imputation_results["Gestation"] = {"MSE": mse, "R2": r2}

df_model_features_test = df[df["Gestation"].isnull()]
X_test = df_model_features_test[["BodyWt","BrainWt", "TotalSleep"]]

y_pred = model.predict(X_test)
gestation_minimum = df["Gestation"].min()
y_pred = np.round(np.maximum(y_pred, gestation_minimum),1)
df.loc[df["Gestation"].isnull(), "Gestation"] = y_pred
df["Gestation"] = np.round(df["Gestation"],1)



#%%% Lifespan2
# We going to fill the remaining LifeSpan rows now that all gestation is available
df_model = df.dropna()
df_model_features = df_model[["BrainWt", "TotalSleep","Gestation"]]
df_model_target = df_model[["LifeSpan"]]

X_train, X_test, y_train, y_test = train_test_split(df_model_features, df_model_target, test_size = 0.1, random_state = 11)
y_train = y_train.values.reshape(-1,1)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Getting the mse and r2 to test for performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

imputation_results["Lifespan2"] = {"MSE": mse, "R2": r2}

df_model_features_test = df[df["LifeSpan"].isnull() & -df["Gestation"].isnull()]
X_test = df_model_features_test[["BrainWt","TotalSleep","Gestation"]]

y_pred = model.predict(X_test)
df.loc[df["LifeSpan"].isnull() & -df["Gestation"].isnull(), "LifeSpan"] = y_pred
df["LifeSpan"] = np.round(df["LifeSpan"],1)



#%% Dreaming

df_model = df[-df["Dreaming"].isnull()]
df_model_null_dreaming = df[df["Dreaming"].isnull()]


# Select the features and target
features = df_model[["TotalSleep","Exposure"]]
target = df_model[["Dreaming"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#Getting the mse and r2 to test for performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

imputation_results["Dreaming"] = {"MSE": mse, "R2": r2}

df_dreaming_null = df[df['Dreaming'].isnull()]

X_test = df_dreaming_null[["TotalSleep","Exposure"]]


# Make predictions on the test set
y_imputation = model.predict(X_test)


df.loc[df["Dreaming"].isnull(),"Dreaming"] = y_imputation

df["Dreaming"] = df["Dreaming"].round(2)


#%% non dreaming

df_ndreaming_null = df[df["NonDreaming"].isnull()]
df_ndreaming_complete = df[-df["NonDreaming"].isnull()]

feature = df_ndreaming_complete[["TotalSleep","BrainWt","BodyWt","Dreaming","Gestation"]]
target = df_ndreaming_complete[["NonDreaming"]]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=3)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#Getting the mse and r2 to test for performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 
imputation_results["NonDreaming"] = {"MSE": mse, "R2": r2}

x = df_ndreaming_null[["TotalSleep","BrainWt","BodyWt","Dreaming","Gestation"]]

y_imputation = model.predict(x)

df.loc[df["NonDreaming"].isnull(),"NonDreaming"] = y_imputation

df["NonDreaming"] = df["NonDreaming"].round(2)


df = df.sort_values(by="Species", ascending=True)
df.to_csv("data/clean_unstandardized.csv", index=False)

#%% Standardization

# I will standardize rather than normalize
col_std = ['BodyWt', 'BrainWt', 'TotalSleep', 'Awake', 'NonDreaming', 'Dreaming', 'LifeSpan', 'Gestation']

# scaler = StandardScaler()
scaler = MinMaxScaler()

for col in col_std:
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

#%% Performance report

df_results = pd.DataFrame(imputation_results)

#%% SAVING FILES
    
df = df.sort_values(by="Species", ascending=True)
df.to_csv("data/Output/cleaned_new_sleep_merged.csv", index=False)


df_results.to_csv("data/Output/processing_results.csv", index=["MSE","R2"])