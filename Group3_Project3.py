#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src=https://www.usine-digitale.fr/mediatheque/2/3/5/001002532_896x598_c.jpg" width="300" alt="cognitiveclass.ai logo"  />
# </center>
# 
# # Introduction
# ## Project: Predicting sleep variables in mammals
# ## Project Objectives:Bbuild a model to predict the sleeping attributes
# TotalSleep and Dreaming from the general, ecological and biological attributes
# ## Members:
# <ul>
#   <li>Seyed Sourena MOHIT TABATABAIE</li>
#   <li>Brandt Charles OLSON</li>
#   <li>Mohamed JOUHARI</li>
#   <li>Hugo BEFFEYTE</li>
#   <li>Kawtar ABIDINE</li>
# </ul>
# 
# ## Project hosting : [ML_Project3_Group3 GitHub Repository](https://github.com/Sourena-Mohit-DSTI/ML_Project3_Group3)
# .

# # Exploratory Analysis of The Data

# <span style="color:green;font-weight:700;font-size:24px"> STEP 1: Dataset Preprocessing
# </span>

# <span style="color:blue;font-weight:700;font-size:20px">1 - Importing the libraries
# </span>

# In[33]:


#! pip install filename


# In[3]:


import pandas as pd
import numpy as np
import copy

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')

init_notebook_mode(connected=True)  # Initialize notebook for offline plot

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from scipy.stats import anderson, kstest, shapiro


# <span style="color:blue;font-weight:700;font-size:20px">2 - Importing the dataset and redefine it 
# </span>

# In[4]:


# Read TSV file with a header
file_path = 'data/sleep_merged.tsv'
df = pd.read_csv(file_path, sep="\t",header=0)

# Display the class of the DataFrame
print(type(df))  # <class 'pandas.core.frame.DataFrame'>

# Display the number of rows and columns in the DataFrame
print(df.shape)  # Not right shape !

# Display the DataFrame
print(df)


# <span style="color:red;font-size: 15px; font-style: italic; font-weight: bold;">--> We can see that we do not have the right shape of the dataset! We should have 88 rows (including the header) and 16 columns (or features).</span>
# 
# 

# <h3>Read/Save Other Data Formats</h3>
# 
# | Data Formate |        Read       |            Save |
# | ------------ | :---------------: | --------------: |
# | csv          |  `pd.read_csv()`  |   `df.to_csv()` |
# | json         |  `pd.read_json()` |  `df.to_json()` |
# | excel        | `pd.read_excel()` | `df.to_excel()` |
# | hdf          |  `pd.read_hdf()`  |   `df.to_hdf()` |
# | sql          |  `pd.read_sql()`  |   `df.to_sql()` |
# | ...          |        ...        |             ... |
# 

# In[3]:


# Remove rows where all values are null
df_cleaned = df.dropna(how='all')

# Save the cleaned dataset to a new TSV file
output_file = 'cleaned_data_no_null_rows.tsv'
df_cleaned.to_csv(output_file, sep='\t', index=False)

print(f"Cleaned dataset (without null rows) saved to {output_file}")


# In[4]:


# Read TSV file with a header
file_path = 'cleaned_data_no_null_rows.tsv'
df = pd.read_csv(file_path, sep="\t",header=0)

# Display the class of the DataFrame
print(type(df))  # <class 'pandas.core.frame.DataFrame'>

# Display the number of rows and columns in the DataFrame
print(df.shape)  # we can see that now we have the right shape of data set 

# Display the 5 first rows to confirm
df.head(10)


# **<p style="background-color:red; color:white;">After this, we just sorted the columns in our dataset in the Excel and saved it as a CSV. Since this is a Jupyter Notebook, we couldn't include this part directly, so we added the CSV file as sleep_merged_main.csv in our repository to continue the notebook. This approach allows us to maintain the integrity of our analysis and facilitates seamless collaboration among team members.</p>**
# 

# In[5]:


df = pd.read_csv("data/sleep_merged_main.csv")


# <div class="alert alert-danger alertdanger" style="margin-top: 10px">
# <h4>Find the name of the columns of the dataframe.</h4>
#     make a copy of dataframe to work with.
# </div>

# In[6]:


df_copy = copy.deepcopy(df)


# In[7]:


cols = df.columns


# In[8]:


cols


# <span style="color:blue;font-weight:700;font-size:20px">3 - Basic Insight of Dataset
# </span>
# <p>
# After reading data into Pandas dataframe, it is time for us to explore the dataset.<br>
# 
# There are several ways to obtain essential insights of the data to help us better understand our dataset.
# 
# </p>

# <h4>Data Types</h4>
# <p>
# Data has a variety of types.<br>
# 
# The main types stored in Pandas dataframes are <b>object</b>, <b>float</b>, <b>int</b>, <b>bool</b> and <b>datetime64</b>. In order to better learn about each attribute, it is always good for us to know the data type of each column. In Pandas:
# 
# </p>

# In[9]:


df.dtypes


# <h4>Describe</h4>
# statistical summary of each column e.g. count, column mean value, column standard deviation, etc., we use the describe method
# This method will provide various summary statistics, excluding <code>NaN</code> (Not a Number) values.
# 

# In[10]:


df.describe()


# <p>
# This shows the statistical summary of all numeric-typed (int, float) columns.<br>
# 
# For example, the attribute "BodyWt" has 87 counts, the mean value of this column is 161.38, the standard deviation is 768.8, the minimum value is 0.005 and the maximum value is 6654. <br>
# 
# However, what if we would also like to check all the columns including those that are of type object? <br><br>
# 
# add an argument <code>include = "all"</code> inside the bracket.
# 
# </p>

# In[11]:


# describe all the columns in "df" 
df.describe(include = "all")


# <p>
# Now it provides the statistical summary of all the columns, including object-typed attributes.<br>
# 
# We can now see how many unique values there, which one is the top value and the frequency of top value in the object-typed columns.<br>
# 
# Some values in the table above show as "NaN". This is because those numbers are not available regarding a particular column type.<br>
# 
# </p>

# <h4>Info</h4>
# Another method you can use to check your dataset is: dataframe.info()
# It provides a concise summary of your DataFrame.
# 
# This method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.
# 

# In[12]:


# look at the info of "df"
df.info()


# <span style="color:green;font-weight:700;font-size:24px"> STEP 2: Data Wrangling </span>

# <span style="color:blue;font-weight:700;font-size:20px">1 - Changing the Encoding of Categorical Features and Analysis
# </span>

# <h4 id="indicator">Indicator Variable (or Dummy Variable)</h4>
# <b>What is an indicator variable?</b>
# <p>
#     An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning. 
# </p>
# 
# <b>Why we use indicator variables?</b>
# 
# <p>
#     We use indicator variables so we can use categorical variables for regression analysis in the later modules.
# </p>

# <div style="background-color:#ADD8E6; padding: 18px;">
# <b>Vore Column Encoding</b>
# </div>

# In[13]:


# Display unique values in the 'vore' column
unique_vore_values = df['Vore'].unique()
print("Unique values in 'vore' column:", unique_vore_values)


# In[14]:


# Perform one-hot encoding on the 'vore' column
df = pd.get_dummies(df, columns=['Vore'], prefix='Vore',drop_first=True)

# Print the updated DataFrame with one-hot encoding
print("Updated DataFrame with one-hot encoding:")
print(df)


# to use the 'vore' column as a feature in regression models, and it contains boolean values representing some binary attribute, it's common to convert them to 0 and 1 before using them in the regression model. This conversion allows you to treat them as numerical variables.

# In[15]:


# Convert boolean values to 0 and 1 for the specified columns
bool_columns = ['Vore_omni', 'Vore_herbi', 'Vore_insecti']
df[bool_columns] = df[bool_columns].astype(int)

# Print the updated DataFrame with one-hot encoding and converted boolean values
print("Updated DataFrame with one-hot encoding and converted boolean values:")
print(df)


# In[16]:


# add the Vore Column itself again to dataset
df["Vore"] = df_copy["Vore"]


# In[17]:


# Update the Columns Variable
cols = df.columns


# In[18]:


cols


# In[19]:


unique_vore = df["Vore"].unique()

# Create subplots
fig, axes = plt.subplots(2, len(unique_vore), figsize=(15, 10))

# Loop through each unique value in "Vore" and create scatter plots for Brain Weight vs. Total Sleep
for i, diet in enumerate(unique_vore):
    dff = df[df["Vore"] == diet]
    ax = axes[0, i]
    sns.scatterplot(data=dff, x='BrainWt', y='Dreaming', ax=ax, marker="o", size=2, color='black')
    ax.set_title(f"Brain Weight vs. Total Sleep for {diet}")
    ax.set_xlabel('Brain Weight (kg)')
    ax.set_ylabel('Total Sleep (hours)')
    ax.set_xscale('log')

    ax.set_facecolor('#C0F012')  # Dark background
    ax.grid(True, linestyle='--', alpha=0.8, color="black")

# Loop through each unique value in "Vore" and create scatter plots for Body Weight vs. Total Sleep
for i, diet in enumerate(unique_vore):
    dff = df[df["Vore"] == diet]
    ax = axes[1, i]
    sns.scatterplot(data=dff, x='BodyWt', y='TotalSleep', ax=ax, marker="o", size=2, color='black')
    ax.set_title(f"Body Weight vs. Total Sleep for {diet}")
    ax.set_xlabel('Body Weight (kg)')
    ax.set_ylabel('Total Sleep (hours)')
    ax.set_xscale('log')

    ax.set_facecolor('#C7B5F7')  # Dark background
    ax.grid(True, linestyle='--', alpha=0.8, color="black")

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# <div style="background-color:#ADD8E6; padding: 18px;">
# <b>Species,Conservation and Order</b>
# </div>

# In[20]:


# Display unique values in the 'Order' column
unique_order_values = df['Order'].unique()
print("Unique values in 'order' column:", unique_order_values)


# In[21]:


# Count of species per Order
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='Order', order=df["Order"].value_counts().index, hue='Order', palette="viridis", legend=False)
plt.title('Count of Species per Order')
plt.xlabel('Count')
plt.ylabel('Order', fontsize=18)
plt.show()


# In[22]:


# Conservation status
plt.figure(figsize=(5, 5))
df['Conservation'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Piechart of Conservation Status ratio for Species')
plt.ylabel('')
plt.show()


# In[23]:


# Boxplot of Lifespan by Conservation status
order = ["domesticated","lc","cd","nt","vu","en","cr"]

plt.figure(figsize=(12, 6)) 
sns.boxplot(data=df, x='Conservation', y='LifeSpan', linewidth=1, palette="Set2", showfliers=True, hue='Conservation', order=order, legend=False)
sns.stripplot(data=df, y="LifeSpan", x="Conservation", color="black", size=0.4)

plt.title('Boxplot of Lifespan values by Conservation Status')
plt.xlabel('Conservation Status')
plt.ylabel('Lifespan (years)')
plt.show()


# In[24]:


# Boxplot of Gestation by Conservation status
order = ["domesticated", "lc", "cd", "nt", "vu", "en", "cr"]

plt.figure(figsize=(12, 6)) 
sns.boxplot(data=df, x='Conservation', y='Gestation', linewidth=1, palette="Set2", showfliers=True, hue='Conservation', order=order)
sns.stripplot(data=df, y="Gestation", x="Conservation", color="black", size=0.4)

plt.title('Boxplot of Gestation values by Conservation Status')
plt.xlabel('Conservation Status')
plt.ylabel('Gestation (days)')
plt.legend(loc='upper right', title='Conservation Status', frameon=False)
plt.show()


# #### There seemed to be a dependency between cd and gestation , but cd has only 2 records

# In[25]:


print(df[df["Conservation"] == "cd"])


# In[26]:


df["Conservation_count"] = df.groupby("Conservation")["Conservation"].transform("count")


# In[27]:


sns.countplot(data=df, y="Conservation", hue="Conservation", order=df['Conservation'].value_counts().index, palette="viridis", legend=False)


# <span style="color:blue;font-weight:700;font-size:20px">2 - Identify missing data and Data Cleaning
# </span>
# 
# <h4>Evaluating for Missing Data</h4>
# 
# The missing values are converted by default. We use the following functions to identify these missing values. There are two methods to detect missing data:
# 
# <ol>
#     <li><b>.isnull()</b></li>
#     <li><b>.notnull()</b></li>
# </ol>
# The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data.

# In[28]:


missing_data = df.isnull()
missing_data.head(5)


# <h4>Count missing values in each column</h4>
# <p>
# Using a for loop in Python, we can quickly figure out the number of missing values in each column. As mentioned above, "True" represents a missing value and "False" means the value is present in the dataset.  In the body of the for loop the method ".value_counts()" counts the number of "True" values. 
# </p>

# In[29]:


df.isna().sum()


# In[30]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")  


# <div style="background-color:#ADD8E6; padding: 10px;">
# Based on the summary above, each column has 87 rows of data and 9 out of 16 column contain missing values.
# 
# <ol>
#     <li>"Conservation": <b>29 missing data</b></li>
#     <li>"BrainWt": <b>23 missing data</b></li>
#     <li>"NonDreaming": <b>40 missing data</b></li>
#     <li>"Dreaming" : <b>24 missing data</b></li>
#     <li>"LifeSpan": <b>33 missing data</b></li>
#     <li>"Gestation": <b>33 missing data</b></li>
#     <li>"Predation": <b>29 missing data</b></li>
#     <li>"Exposure": <b>29 missing data</b></li>
#     <li>"Danger": <b>29 missing data</b></li>
# </ol>
# </div>

# In[31]:


# Calculate the percentage of missing values for each column
percent_missing = df.isnull().sum() * 100 / len(df)

# Create a DataFrame to store the results
missing_value_df = pd.DataFrame({
    'column_name': df.columns,
    'percent_missing': percent_missing
})

# Sort the DataFrame by missing percentage (if needed)
missing_value_df.sort_values('percent_missing', inplace=True)

# Display the results
print(missing_value_df)


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plotting missing values
plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap')
plt.show()


# <h3 id="deal_missing_values">Deal with missing data</h3>
# <b>How to deal with missing data?</b>
# 
# <ol>
#     <li>Drop data<br>
#         a. Drop the whole row<br>
#         b. Drop the whole column
#     </li>
#     <li>Replace data<br>
#         a. Replace it by mean<br>
#         b. Replace it by frequency<br>
#         c. Replace it based on other functions
#     </li>
# </ol>

# ### Correlation in Data Analysis
# 
# Correlation measures the **strength and direction** of the linear relationship between two variables. It quantifies how closely the data points in one variable move in relation to the data points in another variable. The correlation coefficient ranges from **-1 to 1**:
# 
# - **Positive correlation (0 to 1)**: When one variable increases, the other tends to increase as well. A value close to 1 indicates a strong positive relationship.
# - **Negative correlation (-1 to 0)**: When one variable increases, the other tends to decrease. A value close to -1 indicates a strong negative relationship.
# - **No correlation (around 0)**: The variables are not linearly related.
# 
# Correlation does **not** imply causation. Even if two variables are highly correlated, it doesn't mean one causes the ot#her.
# 
# ### Higher Correlation
# 
# When we say "higher correlation," we mean a **stronger linear relationship** between the variables:
# 
# - If the correlation coefficient is **0.8**, it indicates a **strong positive correlation**.
# - If the correlation coefficient is **-0.6**, it indicates a **moderate negative correlation**.
# - If the correlation coefficient is **0.1**, it suggests a **weak positive correlation** (closer to no #correlation).
# 
# ### Interpreting Correlation
# 
# - **Positive correlation**: As one variable increases, the other tends to increase. For instance, height and weight often have a positive correlation.
# - **Negative correlation**: As one variable increases, the other tends to decrease. For example, temperature and ice cream sales may have a negative correlation.
# - **No correlation**: When changes in one variable do not consistently predict changes in the other.

# In[33]:


# Filtering only numerical columns for correlation matrix
num_df = df.select_dtypes(include=["float", "int"])
num_df = num_df.drop(["Vore_herbi","Vore_insecti","Vore_omni"], axis=1)

correlation_matrix = num_df.iloc[:].corr()
# Display the correlation matrix
print(correlation_matrix)


# In[34]:


# Create subplots with two columns to display two heatmaps side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Calculate and draw the Pearson correlation heatmap
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm',
            vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axes[0])
axes[0].set_title("Pearson Correlation")

# Calculate and draw the Spearman correlation heatmap
sns.heatmap(num_df.corr(method='spearman'), annot=True, cmap='vlag',
            vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axes[1])
axes[1].set_title("Spearman Correlation")

# Display the plot
plt.show()


# In[35]:


# Create the heatmap
fig2 = px.imshow(correlation_matrix, labels=dict(x="Column Index", y="Column Index"))

# Customize the heatmap (optional)
fig2.update_layout(
    title="Correlation Heatmap",
    xaxis_title="Column Index",
    yaxis_title="Column Index",
    coloraxis_colorbar=dict(title="Correlation"),
    width=900,  # Adjust width as needed
    height=700  # Adjust height as needed
)

# Show the plot
fig2.show()



# <div style="background-color:#ADD8E6; padding: 18px;">
# <b>Distribution Analysis of Numeric Variables</b>
# </div>

# In[36]:


# List of numeric columns to plot
numeric_cols = [col for col in df.columns if df[col].dtype in ["float64", "int64"] and col not in ['Predation', 'Exposure', 'Danger']]

# Define a color palette for the plots
color_palette = sns.color_palette("Set2", len(numeric_cols))

# Create a figure for the subplots with a smaller size and non-white background
fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(8, 4 * len(numeric_cols)))  # Adjust figsize
fig.set_facecolor('#f0f0f0')  # Set background color to light gray

for i, col in enumerate(numeric_cols):
    # Plot the histogram on the left (first column)
    sns.histplot(data=df, x=col, bins="auto", kde=True, ax=axes[i, 0], color=color_palette[i])
    axes[i, 0].set_title(f"Histogram of {col}")
    axes[i, 0].set_xlabel("Values")
    axes[i, 0].set_ylabel("Frequency")
    
    # Plot the boxplot on the right (second column)
    sns.boxplot(data=df, x=col, ax=axes[i, 1], color=color_palette[i])
    axes[i, 1].set_title(f"Boxplot of {col}")
    axes[i, 1].set_xlabel("Variable")
    axes[i, 1].set_ylabel("Values")

# Add a title to the entire figure
plt.suptitle("Distribution Analysis of Numeric Variables", fontsize=14, y=1.02)  # Adjust fontsize

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[37]:


# Histograms for continuous variables
sns.histplot(df['TotalSleep'], kde=True)
plt.show()



# The effect of applying a logarithmic transformation to features in machine learning models depends on the distribution and characteristics of the data. Logarithmic transformations are often used to handle skewed data distributions, reduce the impact of outliers, and improve the interpretability of relationships.

# In[39]:


print("Minimum value in 'BrainWt' column:", df['BrainWt'].min())
print("Minimum value in 'BodyWt' column:", df['BodyWt'].min())


# In[40]:


# Define a small positive value
epsilon = 1e-10

# Apply logarithmic transformation to non-zero and positive values
df['LogBrainWt'] = np.where(df['BrainWt'] > 0, np.log(df['BrainWt'] + epsilon), np.nan)
df['LogBodyWt'] = np.where(df['BodyWt'] > 0, np.log(df['BodyWt']), np.nan)

# Plot the pairplot
sns.pairplot(df, vars=["TotalSleep", "LogBrainWt", "LogBodyWt"])


# Lets creat a new Column using Feature engineering 

# In[41]:


df["BrainBodyRatio"] = df["BrainWt"] / df["BodyWt"]
sns.scatterplot(data=df, x="BrainBodyRatio", y="TotalSleep")


# <div style="background-color:#b22222; padding: 10px;">
# 
#      Some useful functions to check for correlations and normality, that we will often need throughout this notebook
#     
#      Next we will define a function to test not graphically but analyticaly the normality of our features
# </div>

# In[42]:


# define a correlation function that returns the corr a target column has with other columns if corr > t (threshold)
# use both pearson and spearman because some features are not normally distributed etc.

def corr_func(target: str, t):
    spear_corr = {}
    pears_corr = {}
    for col in cols:
        if df[col].dtype in ["float64", "int64"] and col != target:
            # Pairwise deletion to include only rows where there is no missing values at all in either of the two features
            valid_rows = df[[col, target]].dropna()

            if len(valid_rows) > 0:  # Check if there are valid rows for correlation (since I know the dataset, it's useless here but i still include it)
                correlation, p_value = spearmanr(valid_rows[col], valid_rows[target])
                correlation2 = valid_rows[col].corr(valid_rows[target])
                if abs(correlation) >= t:
                    spear_corr[col] = round(correlation, 2)
                elif abs(correlation2) >= t:
                    pears_corr[col] = round(correlation2, 2)
    print(f"Correlation for {target} : \n\n Spearman correlation is: {spear_corr}", "\n\n", f"Pearson correlation is: {pears_corr}")

    return


# In[43]:


from scipy.stats import kstest, anderson, norm

def normal_test(feature, alpha):
    result = {}

    # Kstest
    ks_stat, ks_pvalue = kstest(feature, 'norm')
    ks_normal = ks_pvalue > alpha
    result['Kolmogorov-Smirnov'] = {'Statistic': ks_stat, 'p_value': ks_pvalue, 'Normal': ks_normal}

    ''''#Anderson
    ad_statistic, ad_critical_values, ad_significance_level = anderson(feature, dist="norm")
    ad_normal = ad_statistic < ad_critical_values[ad_significance_level == alpha][0]'''
    ## ANDERSON TEST seems to fail unexpectedly, I try to make it work, and to not block the code if it fails


    # Check if the sample size is sufficient for Anderson-Darling
    if len(feature) >= 3:
        try:
            ad_statistic, ad_critical_values, ad_significance_level = anderson(feature, dist="norm")
            ad_normal = ad_statistic < ad_critical_values[ad_significance_level == alpha][0]
            result['Anderson-Darling'] = {'Statistic': ad_statistic, 'Critical Values': ad_critical_values, 'Significance Level': ad_significance_level, 'Normal': ad_normal}
        except (ValueError, IndexError):
            result['Anderson-Darling'] = {'Statistic': None, 'Critical Values': None, 'Significance Level': None, 'Normal': False}
            result['Error'] = 'Anderson-Darling test failed'
    else:
        result['Error'] = 'Insufficient data points for testing'


    return result


# <div style="background-color:#b22222; padding: 10px;">
#     
#     To assess whether a feature is normal or not, we can run some statistical test, extract their p-value and compare it to the p value cutoff. (we can   set it 0.05, higher or lower).
# </div>

# In[44]:


label = df["TotalSleep"].dropna().astype(float)
result = normal_test(label, alpha=0.05)
print(result)


# <div style="background-color:#ADD8E6; padding: 18px;">
# <b>Danger,Exposure and Predation</b>
# </div>

# In[45]:


df.describe()


# <div style="background-color:#b22222; padding: 10px;">
# <b>Notice</b>: In our project, we explored various methods for cleaning our data due to its messy nature. Consequently, we opted to employ some distinct approaches for data cleaning. This resulted in the creation of two datasets, each processed using a different method. We will evaluate the best datasets during the modeling phase, ultimately selecting the most effective approach.

# In[46]:


import seaborn as sns
# Scatter plot of Danger vs Predation Plus
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Predation', y='Danger', hue='Exposure')
plt.title('Scatter Plot of Danger vs Predation')
plt.xlabel('Predation')
plt.ylabel('Danger')
plt.legend(title='Exposure')
plt.show()

# Scatter plot of Danger vs Exposure
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Exposure', y='Danger', hue='Predation')
plt.title('Scatter Plot of Danger vs Exposure')
plt.xlabel('Exposure')
plt.ylabel('Danger')
plt.legend(title='Predation')
plt.show()


# As we can see , danger and Predation have a good correlation and linear relationship so we can use that for filling the missing data

# In[47]:


# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=df['Danger'],
    y=df['Predation'],
    z=df['Exposure'],
    mode='markers',
    marker=dict(
        size=6,
        color=df['Exposure'],  # Color by Exposure
        colorscale='Viridis',  # Choose a colormap
        opacity=0.8
    )
)])

# Customize layout
fig.update_layout(
    scene=dict(
        xaxis_title='Danger',
        yaxis_title='Predation',
        zaxis_title='Exposure'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Show the plot
fig.show()


# In[48]:


# Calculate correlation between features and target variable
correlation_danger = df['Danger'].corr(df['Exposure'])
correlation_predation = df['Predation'].corr(df['Exposure'])

print("Correlation between Danger and Exposure:", correlation_danger)
print("Correlation between Predation and Exposure:", correlation_predation)


# In[49]:


df["Predation"] = df["Predation"].fillna(df.groupby("Order")["Predation"].transform("mean").round())
df["Exposure"] = df["Exposure"].fillna(df.groupby("Order")["Exposure"].transform("mean").round())
df["Danger"] = df["Danger"].fillna(df.groupby("Order")["Danger"].transform("mean").round())


# ##### It seems grouping by vore, gives some clues on the these features as well, so we can see that there is a low std dev of pred, exp, dang, grouped by vore
# Notes: Actually grouping by vore was doable at first, but it's a choice. Order and Vore are very much linked, they got the same vore for the same order (almost), sometimes herbi/omni but herbi/omni have almost the same mean. So it's fine and on parallel grouping by order can contain real information other than vore for predation exposure

# In[50]:


df["Predation"] = df["Predation"].fillna(df.groupby("Vore")["Predation"].transform("mean").round())
df["Exposure"] = df["Exposure"].fillna(df.groupby("Vore")["Exposure"].transform("mean").round())
df["Danger"] = df["Danger"].fillna(df.groupby("Vore")["Danger"].transform("mean").round())


# <div style="background-color:#ADD8E6; padding: 18px;">
# <b>BrainWt Column</b>
# </div>

# <span style="color:green;font-weight:700;font-size:15px">  
# Deal with wrong data
# </span>
# 
# <b>Lets look one more time into our dataset first! if we look at BrainWt column , we will notice something very wrong with it.</b>

# #### Here we will correct the 0 values found in the BrainWt feature that should not be 0

# In[51]:


# make a copy of our dataset in case of need first
df_copy = df.copy()


# In[53]:


df_copy[['Species','BrainWt']].head()


# <b>Look at these animals ! zero Brain weight ? it is absolutly wrong so , we calculate the number of zeros in this 
# column and replace it with NaN </b>

# In[55]:


df['BrainWt'] = df['BrainWt'].replace(0, np.nan)


# In[56]:


import plotly.io as pio
pio.renderers.default = "iframe"
# Assuming 'df'
selected_columns = df.columns[4:6]  # Select columns from the 6th column onwards

fig = px.histogram(data_frame=df, x=selected_columns)


# In[57]:


iplot(fig)


# <span style="color:red;font-size: 15px; font-style: italic; font-weight: bold;">--> we can see that we have 2 outliers too , now i am going to find them.</span>

# In[58]:


# Filter rows based on brainWt values
filtered_rows = df[(df['BrainWt'] >= 4250) & (df['BrainWt'] <= 5740)]

# Convert the filtered rows to a DataFrame
filtered_df = filtered_rows['BrainWt'].to_frame()

# Print the filtered DataFrame
print(filtered_rows)


# <span style="color:Blue;font-size: 15px; font-style: italic; font-weight: bold;">--> we can not eliminate these outliers cause both have important data about our objectives outputs but for the purpose of filling missing values of BrainWt with BodyWt (Brain:Body ratio) we can and should!</span>
# 

# <div style="background-color:#ADD8E6; padding: 10px;">
# Based on the summary above, The BrainWt and BodyWt have a great positive correlation.
# Plus, we know that always the BrainWt for each animal should be less than BodyWt.
# So, we can use the Binning method for BodyWt column and for Each interval we can take the mean of Non null values of BrainWt and 
# replace the Nan with it.

# In[59]:


# Define the column names
body_weight_col = 'BodyWt'
brain_weight_col = 'BrainWt'

# Plot the data
fig = px.scatter(df, x=body_weight_col, y=brain_weight_col, 
                 title="Scatter Plot of BodyWt vs. BrainWt",
                 labels={body_weight_col: "Body Weight", brain_weight_col: "Brain Weight"})

# Show the plot
fig.show()


# <div style="background-color:#ADD8E6; padding: 10px;">
# With a strong correlation between 'BodyWt' and 'BrainWt' and a considerable number of missing values in the 'BrainWt' column, you can impute the missing values using a method that takes into account the relationship between the two variables. Since there is a strong correlation, you can use linear regression to predict the missing 'BrainWt' values based on the corresponding 'BodyWt' values.
# </div>

# In[60]:


df_model = df[['BodyWt','BrainWt']].dropna()


# In[61]:


spearman_corr, p_value = spearmanr(df_model["BodyWt"],df_model["BrainWt"])
print(spearman_corr)
print(p_value)


# ##### We train a linear regression model to fill out the 0 values fo brain weight based on body weight

# In[62]:


feature = df_model["BodyWt"].values.reshape(-1,1)
target = df_model["BrainWt"].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size = 0.1, random_state = 11)


# In[63]:


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[64]:


mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
print(mse, r2)


# ##### We do the regression imputation

# In[65]:


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


# In[66]:


# Display the DataFrame with both columns side by side
print(df[['BodyWt', 'BrainWt']])


# In[67]:


# Check the condition that BodyWt should always be greater than BrainWt
condition_violations = df[df['BodyWt']*1000 <= df['BrainWt']]
if not condition_violations.empty:
    print("Warning: There are", len(condition_violations), "rows where BodyWt is not greater than BrainWt.")
    print("Here are the rows:")
    print(condition_violations)
else:
    print("No violations found.")


# <div style="background-color:#32cd32 ; padding: 10px;">
# In a article form wikipedia we can found this image and see that the mouse brain to body Ratio is about 1:40
#     
# [Link Brain–body mass ratio](https://en.wikipedia.org/wiki/Brain%E2%80%93body_mass_ratio)
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Brain-body_mass_ratio_for_some_animals_diagram.svg/1280px-Brain-body_mass_ratio_for_some_animals_diagram.svg.png" alt="Nature" width="500" height="450">
# </div>
# 

# In[68]:


# Calculate the ratio of BrainWt to BodyWt
ratio = 1 / 40
# Replace incorrect values with the ratio of 1:40 with respect to 'BodyWt'
df.loc[condition_violations.index, 'BrainWt'] = df.loc[condition_violations.index, 'BodyWt'] * ratio


# In[69]:


condition_violations = df[df['BodyWt']*1000 <= df['BrainWt']]
if not condition_violations.empty:
    print("Warning: There are", len(condition_violations), "rows where BodyWt is not greater than BrainWt.")
    print("Here are the rows:")
    print(condition_violations)
else:
    print("No violations found.")


# In[70]:


df.isna().sum()


# <div style="background-color:#ADD8E6; padding: 18px;">
# <b>Next go for deal with LifeSpan & Gestation Columns</b>
# </div>

# ##### We going to fill some LifeSpan rows where gestation is available using regression imputation again

# In[71]:


corr_func("LifeSpan",0)


# In[72]:


# We going to fill some LifeSpan rows where gestation is available
df_model = df.dropna()
df_model_features = df_model[["BrainWt", "TotalSleep","Gestation"]]
df_model_target = df_model[["LifeSpan"]]

X_train, X_test, y_train, y_test = train_test_split(df_model_features, df_model_target, test_size = 0.1, random_state = 11)
y_train = y_train.values.reshape(-1,1)
model = LinearRegression()
model.fit(X_train, y_train)

df_model_features_test = df[df["LifeSpan"].isnull() & -df["Gestation"].isnull()]
X_test = df_model_features_test[["BrainWt","TotalSleep","Gestation"]]

y_pred = model.predict(X_test)
df.loc[df["LifeSpan"].isnull() & -df["Gestation"].isnull(), "LifeSpan"] = y_pred
df["LifeSpan"] = np.round(df["LifeSpan"],1)


# #### We going to FULLY fill out Gestation missing values through regression imputation

# In[73]:


corr_func("Gestation",0.5)


# In[74]:


# We going to FULLY fill out Gestation missing values through regression imputation
df_model = df.dropna()
df_model_features = df_model[["BodyWt","BrainWt", "TotalSleep"]]
df_model_target = df_model[["Gestation"]]

X_train, X_test, y_train, y_test = train_test_split(df_model_features, df_model_target, test_size = 0.1, random_state = 11)
y_train = y_train.values.reshape(-1,1)
model = LinearRegression()
model.fit(X_train, y_train)

df_model_features_test = df[df["Gestation"].isnull()]
X_test = df_model_features_test[["BodyWt","BrainWt", "TotalSleep"]]

y_pred = model.predict(X_test)
gestation_minimum = df["Gestation"].min()
y_pred = np.round(np.maximum(y_pred, gestation_minimum),1)
df.loc[df["Gestation"].isnull(), "Gestation"] = y_pred
df["Gestation"] = np.round(df["Gestation"],1)


# #### We going to fill the remaining LifeSpan rows now that all gestation is available

# In[75]:


# We going to fill the remaining LifeSpan rows now that all gestation is available
df_model = df.dropna()
df_model_features = df_model[["BrainWt", "TotalSleep","Gestation"]]
df_model_target = df_model[["LifeSpan"]]

X_train, X_test, y_train, y_test = train_test_split(df_model_features, df_model_target, test_size = 0.1, random_state = 11)
y_train = y_train.values.reshape(-1,1)
model = LinearRegression()
model.fit(X_train, y_train)

df_model_features_test = df[df["LifeSpan"].isnull() & -df["Gestation"].isnull()]
X_test = df_model_features_test[["BrainWt","TotalSleep","Gestation"]]

y_pred = model.predict(X_test)
df.loc[df["LifeSpan"].isnull() & -df["Gestation"].isnull(), "LifeSpan"] = y_pred
df["LifeSpan"] = np.round(df["LifeSpan"],1)


# <div style="background-color:#ADD8E6; padding: 18px;">
# <b>Next go for deal with NonDreaming & Dreaming Column</b>
# </div>

# In[76]:


# we quickly test what Dreming correlates with
corr_func("Dreaming", 0.5)


# ##### Correlation for Dreaming : 
#  ##### Spearman correlation is: {'TotalSleep': 0.76, 'Awake': -0.76, 'NonDreaming': 0.56, 'Gestation': -0.69, 'Exposure': -0.63, 'Danger': -0.53} 
#  ##### So we will consider TotalSleep, Gestation and Danger

# In[77]:


df.isna().sum()


# <div style="background-color:#ADD8E6; padding: 18px;">
# <b>What About The Dreaming Column? </b>
#     
#     The Dreaming Column it is not just a simple column , is our Y or target to prediction , we can not drop or fill with mean or median! we must find the best Algorithm for predicting Dreaming using the other columns and predict the missing values ! 
# </div>

# In[79]:


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

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("mse =", mse, "r2 =", r2)


# In[80]:


# trying to find the heaviest predictor from the features above

# Assuming you have already trained your linear regression model as shown in your code
# model = LinearRegression()
# model.fit(X_train, y_train)

# Access the coefficients (weights) of the features
coefficients = model.coef_

# Create a dictionary to associate feature names with their coefficients
feature_coefficients = dict(zip(features.columns, coefficients[0]))

# Find the heaviest predictor (feature with the largest absolute coefficient)
heaviest_predictor = max(feature_coefficients, key=lambda k: abs(feature_coefficients[k]))

# Print the coefficients of all features and the heaviest predictor
print("Feature Coefficients:")
for feature, coefficient in feature_coefficients.items():
    print(f"{feature}: {coefficient:.4f}")

print("\nThe Heaviest Predictor is:", heaviest_predictor)


# In[81]:


plt.scatter(df["Dreaming"], df["Gestation"])


# In[83]:


corr_func("Dreaming",0.5)
kendal_corr = df["Dreaming"].corr(df["Gestation"], method="kendall")
print("\n\n Kendall correlation coefficient is ", kendal_corr)


# ##### we don't understand why but the model doesn't put any weight on Gestation despite the good negative correlation between both ? So we remove gestation in the model since it has no influence. We can dig further to understand it maybe comparing with xlstat , minitab or other...
# 
# ##### We tested a bunch of models and features to assess which is preferable for imputing values in Dreaming. It seems Linear regression offers the best score as far as the test was conducted. We trained it previoudly, now we use it to do the imputation.

# In[84]:


df_dreaming_null = df[df['Dreaming'].isnull()]

X_test = df_dreaming_null[["TotalSleep","Exposure"]]


# Make predictions on the test set
y_imputation = model.predict(X_test)


df.loc[df["Dreaming"].isnull(),"Dreaming"] = y_imputation


# In[85]:


df["Dreaming"].isna().sum()


# In[86]:


# Redo the Columns related to the BrainWt after dealing with missing data 
df['LogBrainWt'] = np.log(df['BrainWt'])
df['LogBodyWt'] = np.log(df["BodyWt"])
df["BrainBodyRatio"] = df["BrainWt"] / df["BodyWt"]


# In[87]:


df.describe()


# #### NonDreaming cleaning

# In[88]:


corr_func("NonDreaming",0.000001)


# <div style="background-color:#e2062c; padding: 10px;">
# <b>BUT : 
# Filling missing values based on a feature's correlation with the desired output (target variable) can introduce a risk of overfitting, especially if the correlation is high. Overfitting occurs when a model learns to capture noise or random fluctuations in the training data rather than the underlying relationships
# </b>
# </div.

# In[89]:


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

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("mse =", mse, "r2 =", r2)



# In[90]:


# Now we will to the regression imputation

x = df_ndreaming_null[["TotalSleep","BrainWt","BodyWt","Dreaming","Gestation"]]

y_imputation = model.predict(x)

df.loc[df["NonDreaming"].isnull(),"NonDreaming"] = y_imputation


# In[91]:


df["NonDreaming"].isna().sum()


# In[92]:


df["Dreaming"] = df["Dreaming"].round(2)
df["NonDreaming"] = df["NonDreaming"].round(2)


# <h2 id="data_standardization">Data Standardization</h2>
# <p>
# Data is usually collected from different agencies in different formats.
# (Data standardization is also a term for a particular type of data normalization where we subtract the mean and divide by the standard deviation.)
# </p>

# In[93]:


# Convert grams to kilograms
df['BrainWt'] = df['BrainWt'] / 1000


# In[95]:


df.head(8)


# <h2 id="data_normalization">Data Normalization</h2>
# 
# <b>Why normalization?</b>
# 
# <p>Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include scaling the variable so the variable average is 0, scaling the variable so the variance is 1, or scaling the variable so the variable values range from 0 to 1.
# </p>
# 

# In[97]:


# Define columns to standardize
col_std = ['BodyWt', 'BrainWt', 'TotalSleep', 'Awake', 'NonDreaming', 'Dreaming', 'LifeSpan', 'Gestation']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale and standardize each column
for col in col_std:
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

# Save the scaling parameters
scaling_params = scaler.data_min_, scaler.data_max_


# In[98]:


from scipy.special import boxcox1p

# Assuming df is your DataFrame and col_std is your list of columns to transform

for col in col_std:
    # Perform Johnson transformation using Box-Cox transformation
    df[col] = boxcox1p(df[col], 0.5)  # Adjust the second parameter as needed
    
# Now your data in the specified columns should be transformed using the Box-Cox transformation


# In[99]:


#from scipy.stats import johnsonsb
#import numpy as np

# Assuming df is your DataFrame and col_std is your list of columns to transform

#for col in col_std:
    # Fit Johnson SB distribution to the data
#    params = johnsonsb.fit(df[col])
    
    # Transform the data using the Johnson SB distribution
#    df[col] = johnsonsb(*params).cdf(df[col])

# Now your data in the specified columns should be transformed using the Johnson SB distribution


# ##### DF should contains only standardize data now, we will check.

# In[100]:


scaled = {}
for col in col_std:
    mean = df[col].mean()
    std = df[col].std()
    scaled[col] = [mean,std]


# In[101]:


scaled


# In[102]:


# List of numeric columns to plot
numeric_cols = [col for col in df.columns if df[col].dtype in ["float64", "int64"] and col not in ['Predation', 'Exposure', 'Danger']]

# Define a color palette for the plots
color_palette = sns.color_palette("Set2", len(numeric_cols))

# Create a figure for the subplots with a smaller size and non-white background
fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(8, 4 * len(numeric_cols)))  # Adjust figsize
fig.set_facecolor('#f0f0f0')  # Set background color to light gray

for i, col in enumerate(numeric_cols):
    # Plot the histogram on the left (first column)
    sns.histplot(data=df, x=col, bins="auto", kde=True, ax=axes[i, 0], color=color_palette[i])
    axes[i, 0].set_title(f"Histogram of {col}")
    axes[i, 0].set_xlabel("Values")
    axes[i, 0].set_ylabel("Frequency")
    
    # Plot the boxplot on the right (second column)
    sns.boxplot(data=df, x=col, ax=axes[i, 1], color=color_palette[i])
    axes[i, 1].set_title(f"Boxplot of {col}")
    axes[i, 1].set_xlabel("Variable")
    axes[i, 1].set_ylabel("Values")

# Add a title to the entire figure
plt.suptitle("Distribution Analysis of Numeric Variables", fontsize=14, y=1.02)  # Adjust fontsize

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# Seems like it's well standardize.

# <div style="background-color:#b22222; padding: 10px;">
# <b>We should as mentioned in instruction of project, drop the Awake column , its redundant!</b>
# </div>

# In[103]:


# Drop the 'awake' column
df = df.drop(columns=['Awake'])


# Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns are empty enough to drop entirely.
# We have some freedom in choosing which method to replace data; however, some methods may seem more reasonable than others. We will apply each method to many different columns:
# <div style="background-color:#b22222; padding: 10px;">
# <b>Drop the whole column:</b>
# 
# <ul>
#     <li><b>"Conservation": 29 missing data,no direct relation to our desired outputs, simply delete the whole column</b>
#         <ul>
#             <li>Reason: The conservation status of mammals, as categorized by the International Union for Conservation of Nature (IUCN), refers to their risk of extinction. It classifies species into various levels, such as “Least Concern,” “Endangered,” or “Critically Endangered” based on factors like population size, habitat loss, and threats. In summary, there isn’t direct evidence linking conservation status to mammal sleep</li>
#         </ul>
#     </li>
# </ul>
# </div>

# In[104]:


df.drop(columns=['Conservation'], inplace=True)
df.drop(columns=['Conservation_count'], inplace=True)

# Verify that the column has been dropped
print("Columns after dropping 'conservation':", df.columns)


# In[105]:


df.describe()


# ## Saving File - Data cleaned

# In[106]:


df = df.sort_values(by="Species", ascending=True)
df.to_csv("cleaned_sleep_merged.csv", index=False)


# # Statistical Analysis & Feature engineering

# In[107]:


from plotly.subplots import make_subplots
# Step 1: Compute the correlation matrix
df = df.select_dtypes(include=["float", "int"])

# Compute the correlation matrix
correlation_matrix = df.corr()

# Step 2: Visualize the correlation matrix using Plotly
fig = make_subplots(rows=1, cols=1)

heatmap = go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    colorscale='Viridis',
)

fig.add_trace(heatmap)

fig.update_layout(
    title="Correlation Matrix",
    xaxis_title="Features",
    yaxis_title="Features",
)

fig.show()


# <b>P-value</b>
# 
# <p>What is this P-value? The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.</p>
# 
# By convention, when the
# 
# <ul>
#     <li>p-value is $<$ 0.001: we say there is strong evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.05: there is moderate evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.1: there is weak evidence that the correlation is significant.</li>
#     <li>the p-value is $>$ 0.1: there is no evidence that the correlation is significant.</li>
# </ul>
# 

# In[108]:


from scipy import stats

# Assuming your normalized DataFrame is named df_normalized

# List to store correlation coefficients and p-values for TotalSleep
pearson_correlation_total_sleep = []
p_values_total_sleep = []

# List to store correlation coefficients and p-values for Dreaming
pearson_correlation_dreaming = []
p_values_dreaming = []

# Columns to calculate correlation for
columns_to_correlate = ['BodyWt', 'BrainWt', 'NonDreaming', 'LifeSpan', 'Gestation', 
                        'Predation', 'Exposure', 'Danger', 'Vore_omni', 'Vore_herbi', 'Vore_insecti']

# Calculate Pearson correlation coefficient and p-value for each pair of columns for TotalSleep and Dreaming
for column in columns_to_correlate:
    pearson_coef_total_sleep, p_value_total_sleep = stats.pearsonr(df[column], df['TotalSleep'])
    pearson_correlation_total_sleep.append(pearson_coef_total_sleep)
    p_values_total_sleep.append(p_value_total_sleep)
    
    pearson_coef_dreaming, p_value_dreaming = stats.pearsonr(df[column], df['Dreaming'])
    pearson_correlation_dreaming.append(pearson_coef_dreaming)
    p_values_dreaming.append(p_value_dreaming)

# Print the results for TotalSleep
print("Correlation results for TotalSleep:")
for i, column in enumerate(columns_to_correlate):
    print("The Pearson Correlation Coefficient between 'TotalSleep' and '{}' is {:.2f} with a P-value of P = {:.4f}".format(column, pearson_correlation_total_sleep[i], p_values_total_sleep[i]))

# Print the results for Dreaming
print("\nCorrelation results for Dreaming:")
for i, column in enumerate(columns_to_correlate):
    print("The Pearson Correlation Coefficient between 'Dreaming' and '{}' is {:.2f} with a P-value of P = {:.4f}".format(column, pearson_correlation_dreaming[i], p_values_dreaming[i]))


# In[109]:


# List to store significance interpretation for TotalSleep
significance_total_sleep = []

# List to store significance interpretation for Dreaming
significance_dreaming = []

# Define the significance levels
significance_levels = [0.001, 0.05, 0.1]

# Interpret the significance of the correlation for TotalSleep
for p_value in p_values_total_sleep:
    if p_value < significance_levels[0]:
        significance_total_sleep.append("Strong evidence of significant correlation")
    elif p_value < significance_levels[1]:
        significance_total_sleep.append("Moderate evidence of significant correlation")
    elif p_value < significance_levels[2]:
        significance_total_sleep.append("Weak evidence of significant correlation")
    else:
        significance_total_sleep.append("No evidence of significant correlation")

# Interpret the significance of the correlation for Dreaming
for p_value in p_values_dreaming:
    if p_value < significance_levels[0]:
        significance_dreaming.append("Strong evidence of significant correlation")
    elif p_value < significance_levels[1]:
        significance_dreaming.append("Moderate evidence of significant correlation")
    elif p_value < significance_levels[2]:
        significance_dreaming.append("Weak evidence of significant correlation")
    else:
        significance_dreaming.append("No evidence of significant correlation")

# Print the significance interpretation for TotalSleep
print("Significance interpretation for TotalSleep:")
for i, column in enumerate(columns_to_correlate):
    print("The significance interpretation for the correlation between 'TotalSleep' and '{}' is: {}".format(column, significance_total_sleep[i]))

# Print the significance interpretation for Dreaming
print("\nSignificance interpretation for Dreaming:")
for i, column in enumerate(columns_to_correlate):
    print("The significance interpretation for the correlation between 'Dreaming' and '{}' is: {}".format(column, significance_dreaming[i]))


# <div style="background-color:#ADD8E6; padding: 18px;">
#     Based on the significance interpretation for 'TotalSleep' and 'Dreaming', we can draw the following conclusions:
# 
# For TotalSleep:
# 
# There is moderate evidence of a significant correlation between TotalSleep and BodyWt, BrainWt, and Predation.
# There is strong evidence of a significant correlation between TotalSleep and NonDreaming, LifeSpan, Gestation, Exposure, and Danger.
# There is no evidence of a significant correlation between TotalSleep and Vore_carni.
# There is weak evidence of a significant correlation between TotalSleep and Vore_herbi.
# There is moderate evidence of a significant correlation between TotalSleep and Vore_insecti.
# For Dreaming:
# 
# There is weak evidence of a significant correlation between Dreaming and BodyWt, BrainWt, LifeSpan, and Vore_carni.
# There is strong evidence of a significant correlation between Dreaming and NonDreaming, Gestation, Predation, Exposure, and Danger.
# There is moderate evidence of a significant correlation between Dreaming and Vore_herbi.
# There is weak evidence of a significant correlation between Dreaming and Vore_insecti.
# These interpretations suggest varying levels of association between the features and the target variables 'TotalSleep' and 'Dreaming'. Features such as NonDreaming, LifeSpan, Gestation, Exposure, and Danger show stronger associations with TotalSleep and Dreaming, while features like Vore_carni and Vore_insecti show weaker or no evidence of significant correlation.
# </div>

# <h3>ANOVA: Analysis of Variance</h3>
# <p>The Analysis of Variance  (ANOVA) is a statistical method used to test whether there are significant differences between the means of two or more groups. ANOVA returns two parameters:</p>
# 
# <p><b>F-test score</b>: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means.</p>
# 
# <p><b>P-value</b>:  P-value tells how statistically significant our calculated score value is.</p>
# 
# 
# 

# In[110]:


from scipy.stats import f_oneway

# Exclude the first column
features = df.columns[1:]

# Define your target variables
targets = ['TotalSleep', 'Dreaming']

# Perform ANOVA for each feature with respect to each target variable
for target in targets:
    print(f"ANOVA results for '{target}':")
    for column in features:
        f_statistic, p_value = f_oneway(df[column], df[target])
        if f_statistic > 1:
            effect_size = "Larger effect size (larger difference between means)"
        else:
            effect_size = "Smaller effect size (smaller difference between means)"
        print(f"The ANOVA for '{column}' has an F-statistic of {f_statistic:.2f} and a p-value of {p_value:.4f}. {effect_size}")


# <div style="background-color:#ADD8E6; padding: 18px;">
# For 'TotalSleep':
# 
# 'BodyWt', 'BrainWt', 'NonDreaming', 'LifeSpan', and 'Gestation' have significantly different means compared to 'TotalSleep', as evidenced by their very low p-values (p < 0.001) and large effect sizes (F-statistic > 1). This indicates a larger difference between the means.
# 'Predation' and 'Vore_herbi' have p-values greater than 0.1, suggesting that there is no significant difference in means between these features and 'TotalSleep'. The effect size is smaller (F-statistic < 1).
# 'Exposure' and 'Danger' have p-values between 0.05 and 0.1, indicating weak evidence of significant differences in means compared to 'TotalSleep'. However, the effect size is still relatively large.
# 'Vore_carni' has a p-value less than 0.05, suggesting a moderate evidence of a significant difference in means compared to 'TotalSleep'. The effect size is larger.
# For 'Dreaming':
# 
# 'BodyWt', 'BrainWt', 'NonDreaming', 'LifeSpan', 'Gestation', 'Predation', 'Exposure', 'Danger', and 'Vore_insecti' have significantly different means compared to 'Dreaming', as evidenced by their very low p-values (p < 0.001) and large effect sizes (F-statistic > 1). This indicates a larger difference between the means.
# 'Vore_herbi' has a p-value less than 0.05, suggesting a weak evidence of a significant difference in means compared to 'Dreaming'. The effect size is larger.
# 'Vore_carni' has a p-value greater than 0.1, indicating no significant difference in means compared to 'Dreaming'. The effect size is smaller.
# Overall, the ANOVA results suggest that certain features have significantly different means compared to 'TotalSleep' and 'Dreaming', while others do not. The effect sizes also vary, indicating the magnitude of the differences between the means.

# <div style="background-color:#ffd700; padding: 20px;">
# 
# ## Conclusion
# For predicting TotalSleep:
# 
# Consider using features that have shown moderate to strong evidence of significant correlation with TotalSleep, such as 'BodyWt', 'BrainWt', 'NonDreaming', 'LifeSpan', 'Gestation', 'Exposure', and 'Danger'.
# Features like 'Predation' and 'Vore_herbi' may not be useful for predicting TotalSleep as they did not show significant correlation.
# Pay attention to the magnitude of effect sizes when selecting features. Features with larger effect sizes are likely to have more substantial influence on TotalSleep.
# You can experiment with different regression models such as linear regression, decision trees, random forests, or gradient boosting to predict TotalSleep. Evaluate the performance of these models using appropriate metrics like mean squared error (MSE), mean absolute error (MAE), or Rred-u
# 
# red.
# For predicting Dreaming:
# 
# Focus on features that have shown moderate to strong evidence of significant correlation with Dreaming, such as 'BodyWt', 'BrainWt', 'NonDreaming', 'LifeSpan', 'Gestation', 'Predation', 'Exposure', 'Danger', and 'Vore_herbi'.
# Features like 'Vore_carni' and 'Vore_insecti' may not be useful for predicting Dreaming as they did not show significant correlation.
# Similar to TotalSleep, consider the effect sizes of the features when making your selection.
# Experiment with various regression models as mentioned above to predict Dreaming, and evaluate their performance using appropriate metrics.

# # Model training and Evaluation

# #### Loading working file

# In[111]:


file = "cleaned_sleep_merged.csv"

data = pd.read_csv(file)

df = copy.deepcopy(data)


# #### Model selection

# In[112]:


# Feature Selection based on f_regression
features = ['Gestation', 'Exposure', 'LifeSpan', 'Danger', 'BrainWt', 'BodyWt', 'Vore_herbi', 'Vore_insecti','Vore_omni', 'Predation']
X = df[features]
y_total_sleep = df['TotalSleep']
y_dreaming = df['Dreaming']

# Initialize dictionaries to store the cumulative MSE and R2 scores for averaging
cumulative_results_dreaming = {"Linear Regression": {"MSE": [], "R2": []},
                               "Random Forest": {"MSE": [], "R2": []},
                               "Gradient Boosting": {"MSE": [], "R2": []},
                               "SVR" : {"MSE" : [], "R2": []},
                               "KNN-Regressor": {"MSE" : [], "R2": []}}

cumulative_results_total_sleep = {"Linear Regression": {"MSE": [], "R2": []},
                                  "Random Forest": {"MSE": [], "R2": []},
                                  "Gradient Boosting": {"MSE": [], "R2": []},
                                  "SVR" : {"MSE" : [], "R2": []},
                                  "KNN-Regressor": {"MSE" : [], "R2": []}}

def model_apply(X_train, X_test, y_train, y_test, cumulative_results, target: str):
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cumulative_results[name]["MSE"].append(mse)
        cumulative_results[name]["R2"].append(r2)
        
        if name == "Linear Regression":
            # Coefficients for Linear Regression
            coef_abs = np.abs(model.coef_)
            max_indices = np.argsort(coef_abs)[::-1][:3]  # Select indices of top 3 features
            print(f"\nTop 3 predictors for {name} - Target feature: {target}")
            for idx in max_indices:
                print(f"{features[idx]}: {model.coef_[idx]}")
                
        elif name == "Random Forest":
            # Feature importances for Random Forest
            importances = model.feature_importances_
            max_indices = np.argsort(importances)[::-1][:3]  # Select indices of top 3 features
            print(f"\nTop 3 predictors for {name} - Target feature: {target}")
            for idx in max_indices:
                print(f"{features[idx]}: {importances[idx]}")
        
    return cumulative_results

# We pick a seed
random_state = 11
# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=random_state),
    "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    "SVR": SVR(),
    "KNN-Regressor" : KNeighborsRegressor()
}

# Splitting the dataset for both targets
X_train_total_sleep, X_test_total_sleep, y_train_total_sleep, y_test_total_sleep = train_test_split(X, y_total_sleep, test_size=0.2, random_state=random_state)
X_train_dreaming, X_test_dreaming, y_train_dreaming, y_test_dreaming = train_test_split(X, y_dreaming, test_size=0.2, random_state=random_state)

cumulative_results_dreaming = model_apply(X_train_dreaming, X_test_dreaming, y_train_dreaming, y_test_dreaming, cumulative_results_dreaming, "Dreaming")
cumulative_results_total_sleep = model_apply(X_train_total_sleep, X_test_total_sleep, y_train_total_sleep, y_test_total_sleep, cumulative_results_total_sleep, "TotalSleep")

# Calculate and print the mean MSE and R2 for each model
def print_mean_results(cumulative_results, title):
    print(f"\n Results for {title}:")
    for model, metrics in cumulative_results.items():
        mean_mse = np.mean(metrics["MSE"])
        mean_r2 = np.mean(metrics["R2"])
        print(f"{model} - MSE: {mean_mse:.4f}, R2: {mean_r2:.4f}")

print_mean_results(cumulative_results_dreaming, "Target - Dreaming")
print_mean_results(cumulative_results_total_sleep, "Target - Total Sleep")


# <font color='green'>
# 
# **Results for Target - Dreaming:**
# - Linear Regression:
#   - MSE: 0.0201
#   - R2: 0.4010
# - Random Forest:
#   - MSE: 0.0120
#   - R2: 0.6430
# - Gradient Boosting:
#   - MSE: 0.0136
#   - R2: 0.5943
# - SVR (Support Vector Regression):
#   - MSE: 0.0185
#   - R2: 0.4475
# - KNN-Regressor (K-Nearest Neighbors Regressor):
#   - MSE: 0.0187
#   - R2: 0.4428
# 
# **Results for Target - Total Sleep:**
# - Linear Regression:
#   - MSE: 0.0290
#   - R2: 0.5218
# - Random Forest:
#   - MSE: 0.0222
#   - R2: 0.6342
# - Gradient Boosting:
#   - MSE: 0.0240
#   - R2: 0.6033
# - SVR:
#   - MSE: 0.0333
#   - R2: 0.4503
# - KNN-Regressor:
#   - MSE: 0.0287
#   - R2: 0.5272
# 
# </font>
# 

# <span style="background-color:lightgreen"><b>These results provide insights into the performance of each model for predicting the target variables. Among the models evaluated, Random Forest and Gradient Boosting consistently demonstrate lower MSE and higher R2 values, indicating better predictive performance compared to Linear Regression, SVR, and KNN-Regressor. However, the choice of the best-performing model may depend on specific requirements and constraints of the application.
#                                                                                                                                                                                                                                                                            </b></span>
# 

# In[120]:


results_dreaming = pd.DataFrame(cumulative_results_dreaming)
results_totalsleep = pd.DataFrame(cumulative_results_total_sleep)


# In[121]:


results_dreaming


# In[122]:


results_totalsleep


# In[ ]:


results_dreaming.to_csv("data/result/results_dreaming.csv")
results_totalsleep.to_csv("data/result/results_totalsleep.csv")


# - <b>Support Vector Machines (SVM)</b>: SVMs are versatile models that can be used for classification and regression tasks. They work well for both linear and non-linear data and can handle high-dimensional spaces effectively.
# - <b>Gradient Boosting Machines (GBM)</b>: GBM is an ensemble learning technique that builds a series of weak learners (typically decision trees) sequentially, with each tree correcting the errors of its predecessor. It often yields very accurate results and is robust to overfitting.
# - <b>K-Nearest Neighbors (KNN)</b>: KNN is a simple and intuitive algorithm that works well for both regression and classification tasks. It predicts the value of a new data point based on the average of its nearest neighbors in the feature space.
# - <b>Bayesian Regression</b>: Bayesian regression techniques, such as Bayesian linear regression and Gaussian process regression, offer a probabilistic approach to regression modeling. They can handle uncertainty in both the model parameters and predictions.
# 
# 

# ## Multi-Output Regression: using simple linear Regression Vs. Non Linear Regression

# In[134]:


from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
# Select features based on correlation with 'TotalSleep' and 'Dreaming'
total_sleep_features = ['NonDreaming', 'LifeSpan', 'Exposure', 'Danger', 'BodyWt', 'BrainWt', 'Predation','Gestation']
dreaming_features = ['NonDreaming', 'BrainWt', 'BodyWt', 'Gestation']

# Split the data into training and testing sets
X_total_sleep = df[total_sleep_features]
X_dreaming = df[dreaming_features]
y_total_sleep = df['TotalSleep']
y_dreaming = df['Dreaming']

X_total_sleep_train, X_total_sleep_test, y_total_sleep_train, y_total_sleep_test = train_test_split(X_total_sleep, y_total_sleep, test_size=0.2, random_state=42)
X_dreaming_train, X_dreaming_test, y_dreaming_train, y_dreaming_test = train_test_split(X_dreaming, y_dreaming, test_size=0.2, random_state=42)

# Train linear regression models
model_total_sleep = LinearRegression()
model_total_sleep.fit(X_total_sleep_train, y_total_sleep_train)

model_dreaming = LinearRegression()
model_dreaming.fit(X_dreaming_train, y_dreaming_train)

# Predict on the testing set
y_total_sleep_pred = model_total_sleep.predict(X_total_sleep_test)
y_dreaming_pred = model_dreaming.predict(X_dreaming_test)

# Calculate RMSE for evaluation
rmse_total_sleep = root_mean_squared_error(y_total_sleep_test, y_total_sleep_pred)
rmse_dreaming = root_mean_squared_error(y_dreaming_test, y_dreaming_pred)

print("RMSE for TotalSleep prediction:", rmse_total_sleep)
print("RMSE for Dreaming prediction:", rmse_dreaming)

# Perform k-fold cross-validation for 'TotalSleep' prediction
total_sleep_scores = cross_val_score(model_total_sleep, X_total_sleep, y_total_sleep, cv=5, scoring='neg_mean_squared_error')
total_sleep_rmse_cv = np.sqrt(-total_sleep_scores.mean())

# Perform k-fold cross-validation for 'Dreaming' prediction
dreaming_scores = cross_val_score(model_dreaming, X_dreaming, y_dreaming, cv=5, scoring='neg_mean_squared_error')
dreaming_rmse_cv = np.sqrt(-dreaming_scores.mean())

print("Cross-validated RMSE for TotalSleep prediction:", total_sleep_rmse_cv)
print("Cross-validated RMSE for Dreaming prediction:", dreaming_rmse_cv)
#---------------------------------------------------------------------------------------------------------------------------------
# Define SVR models
model_total_sleep_svr = SVR(kernel='rbf')
model_dreaming_svr = SVR(kernel='rbf')

# Train SVR models
model_total_sleep_svr.fit(X_total_sleep_train, y_total_sleep_train)
model_dreaming_svr.fit(X_dreaming_train, y_dreaming_train)

# Predict using SVR models
y_total_sleep_pred_svr = model_total_sleep_svr.predict(X_total_sleep_test)
y_dreaming_pred_svr = model_dreaming_svr.predict(X_dreaming_test)

# Calculate RMSE for SVR models
rmse_total_sleep_svr = root_mean_squared_error(y_total_sleep_test, y_total_sleep_pred_svr)
rmse_dreaming_svr = root_mean_squared_error(y_dreaming_test, y_dreaming_pred_svr)

print("RMSE for TotalSleep prediction using SVR:", rmse_total_sleep_svr)
print("RMSE for Dreaming prediction using SVR:", rmse_dreaming_svr)

# Perform k-fold cross-validation for TotalSleep prediction using SVR
total_sleep_scores_svr = cross_val_score(model_total_sleep_svr, X_total_sleep, y_total_sleep, cv=5, scoring='neg_mean_squared_error')
total_sleep_rmse_cv_svr = np.sqrt(-total_sleep_scores_svr.mean())

# Perform k-fold cross-validation for Dreaming prediction using SVR
dreaming_scores_svr = cross_val_score(model_dreaming_svr, X_dreaming, y_dreaming, cv=5, scoring='neg_mean_squared_error')
dreaming_rmse_cv_svr = np.sqrt(-dreaming_scores_svr.mean())

print("Cross-validated RMSE for TotalSleep prediction using SVR:", total_sleep_rmse_cv_svr)
print("Cross-validated RMSE for Dreaming prediction using SVR:", dreaming_rmse_cv_svr)


# ##  Using Polynomial regression Vs. Random Forest

# In[133]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Define degree of polynomial features
degree = 2  # You can adjust the degree as needed

# Create polynomial regression pipeline for TotalSleep prediction
poly_total_sleep = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_total_sleep.fit(X_total_sleep_train, y_total_sleep_train)

# Create polynomial regression pipeline for Dreaming prediction
poly_dreaming = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_dreaming.fit(X_dreaming_train, y_dreaming_train)

# Predict using polynomial regression models
y_total_sleep_pred_poly = poly_total_sleep.predict(X_total_sleep_test)
y_dreaming_pred_poly = poly_dreaming.predict(X_dreaming_test)

# Calculate RMSE for polynomial regression models
# Calculate RMSE for polynomial regression models
rmse_total_sleep_poly = mean_squared_error(y_total_sleep_test, y_total_sleep_pred_poly, squared=False)
rmse_dreaming_poly = mean_squared_error(y_dreaming_test, y_dreaming_pred_poly, squared=False)

print("RMSE for TotalSleep prediction using Polynomial Regression:", rmse_total_sleep_poly)
print("RMSE for Dreaming prediction using Polynomial Regression:", rmse_dreaming_poly)

# Perform k-fold cross-validation for TotalSleep prediction using Polynomial Regression
total_sleep_scores_poly = cross_val_score(poly_total_sleep, X_total_sleep, y_total_sleep, cv=5, scoring='neg_mean_squared_error')
total_sleep_rmse_cv_poly = np.sqrt(-total_sleep_scores_poly.mean())

# Perform k-fold cross-validation for Dreaming prediction using Polynomial Regression
dreaming_scores_poly = cross_val_score(poly_dreaming, X_dreaming, y_dreaming, cv=5, scoring='neg_mean_squared_error')
dreaming_rmse_cv_poly = np.sqrt(-dreaming_scores_poly.mean())

print("Cross-validated RMSE for TotalSleep prediction using Polynomial Regression:", total_sleep_rmse_cv_poly)
print("Cross-validated RMSE for Dreaming prediction using Polynomial Regression:", dreaming_rmse_cv_poly)
##-----------------------------------------------------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

# Define Random Forest models
rf_total_sleep = RandomForestRegressor(n_estimators=100, random_state=42)
rf_dreaming = RandomForestRegressor(n_estimators=100, random_state=42)

# Train Random Forest models
rf_total_sleep.fit(X_total_sleep_train, y_total_sleep_train)
rf_dreaming.fit(X_dreaming_train, y_dreaming_train)

# Predict using Random Forest models
y_total_sleep_pred_rf = rf_total_sleep.predict(X_total_sleep_test)
y_dreaming_pred_rf = rf_dreaming.predict(X_dreaming_test)

# Calculate RMSE for Random Forest models
rmse_total_sleep_rf = mean_squared_error(y_total_sleep_test, y_total_sleep_pred_rf, squared=False)
rmse_dreaming_rf = mean_squared_error(y_dreaming_test, y_dreaming_pred_rf, squared=False)

print("RMSE for TotalSleep prediction using Random Forest:", rmse_total_sleep_rf)
print("RMSE for Dreaming prediction using Random Forest:", rmse_dreaming_rf)

# Perform k-fold cross-validation for TotalSleep prediction using Random Forest
total_sleep_scores_rf = cross_val_score(rf_total_sleep, X_total_sleep, y_total_sleep, cv=5, scoring='neg_mean_squared_error')
total_sleep_rmse_cv_rf = np.sqrt(-total_sleep_scores_rf.mean())

# Perform k-fold cross-validation for Dreaming prediction using Random Forest
dreaming_scores_rf = cross_val_score(rf_dreaming, X_dreaming, y_dreaming, cv=5, scoring='neg_mean_squared_error')
dreaming_rmse_cv_rf = np.sqrt(-dreaming_scores_rf.mean())

print("Cross-validated RMSE for TotalSleep prediction using Random Forest:", total_sleep_rmse_cv_rf)
print("Cross-validated RMSE for Dreaming prediction using Random Forest:", dreaming_rmse_cv_rf)


# ## Decision Tree 

# In[129]:


from sklearn.tree import DecisionTreeRegressor

# Define Regression Tree models
tree_total_sleep = DecisionTreeRegressor(random_state=42)
tree_dreaming = DecisionTreeRegressor(random_state=42)

# Train Regression Tree models
tree_total_sleep.fit(X_total_sleep_train, y_total_sleep_train)
tree_dreaming.fit(X_dreaming_train, y_dreaming_train)

# Predict using Regression Tree models
y_total_sleep_pred_tree = tree_total_sleep.predict(X_total_sleep_test)
y_dreaming_pred_tree = tree_dreaming.predict(X_dreaming_test)

# Calculate RMSE for Regression Tree models
rmse_total_sleep_tree = mean_squared_error(y_total_sleep_test, y_total_sleep_pred_tree, squared=False)
rmse_dreaming_tree = mean_squared_error(y_dreaming_test, y_dreaming_pred_tree, squared=False)

print("RMSE for TotalSleep prediction using Regression Trees:", rmse_total_sleep_tree)
print("RMSE for Dreaming prediction using Regression Trees:", rmse_dreaming_tree)

# Perform k-fold cross-validation for TotalSleep prediction using Regression Trees
total_sleep_scores_tree = cross_val_score(tree_total_sleep, X_total_sleep, y_total_sleep, cv=5, scoring='neg_mean_squared_error')
total_sleep_rmse_cv_tree = np.sqrt(-total_sleep_scores_tree.mean())

# Perform k-fold cross-validation for Dreaming prediction using Regression Trees
dreaming_scores_tree = cross_val_score(tree_dreaming, X_dreaming, y_dreaming, cv=5, scoring='neg_mean_squared_error')
dreaming_rmse_cv_tree = np.sqrt(-dreaming_scores_tree.mean())

print("Cross-validated RMSE for TotalSleep prediction using Regression Trees:", total_sleep_rmse_cv_tree)
print("Cross-validated RMSE for Dreaming prediction using Regression Trees:", dreaming_rmse_cv_tree)


# ## K-Nearest Neighbors (KNN) Vs. Bayesian Regression Vs.Gradient Boosting Machines (GBM)

# In[130]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge

# Define GBM models
model_total_sleep_gbm = GradientBoostingRegressor()
model_dreaming_gbm = GradientBoostingRegressor()

# Train GBM models
model_total_sleep_gbm.fit(X_total_sleep_train, y_total_sleep_train)
model_dreaming_gbm.fit(X_dreaming_train, y_dreaming_train)

# Predict using GBM models
y_total_sleep_pred_gbm = model_total_sleep_gbm.predict(X_total_sleep_test)
y_dreaming_pred_gbm = model_dreaming_gbm.predict(X_dreaming_test)

# Calculate RMSE for GBM models
rmse_total_sleep_gbm = mean_squared_error(y_total_sleep_test, y_total_sleep_pred_gbm, squared=False)
rmse_dreaming_gbm = mean_squared_error(y_dreaming_test, y_dreaming_pred_gbm, squared=False)

print("RMSE for TotalSleep prediction using GBM:", rmse_total_sleep_gbm)
print("RMSE for Dreaming prediction using GBM:", rmse_dreaming_gbm)

# Perform k-fold cross-validation for TotalSleep prediction using GBM
total_sleep_scores_gbm = cross_val_score(model_total_sleep_gbm, X_total_sleep, y_total_sleep, cv=5, scoring='neg_mean_squared_error')
total_sleep_rmse_cv_gbm = np.sqrt(-total_sleep_scores_gbm.mean())

# Perform k-fold cross-validation for Dreaming prediction using GBM
dreaming_scores_gbm = cross_val_score(model_dreaming_gbm, X_dreaming, y_dreaming, cv=5, scoring='neg_mean_squared_error')
dreaming_rmse_cv_gbm = np.sqrt(-dreaming_scores_gbm.mean())

print("Cross-validated RMSE for TotalSleep prediction using GBM:", total_sleep_rmse_cv_gbm)
print("Cross-validated RMSE for Dreaming prediction using GBM:", dreaming_rmse_cv_gbm)


# Define KNN models
model_total_sleep_knn = KNeighborsRegressor()
model_dreaming_knn = KNeighborsRegressor()

# Train KNN models
model_total_sleep_knn.fit(X_total_sleep_train, y_total_sleep_train)
model_dreaming_knn.fit(X_dreaming_train, y_dreaming_train)

# Predict using KNN models
y_total_sleep_pred_knn = model_total_sleep_knn.predict(X_total_sleep_test)
y_dreaming_pred_knn = model_dreaming_knn.predict(X_dreaming_test)

# Calculate RMSE for KNN models
rmse_total_sleep_knn = mean_squared_error(y_total_sleep_test, y_total_sleep_pred_knn, squared=False)
rmse_dreaming_knn = mean_squared_error(y_dreaming_test, y_dreaming_pred_knn, squared=False)

print("RMSE for TotalSleep prediction using KNN:", rmse_total_sleep_knn)
print("RMSE for Dreaming prediction using KNN:", rmse_dreaming_knn)

# Perform k-fold cross-validation for TotalSleep prediction using KNN
total_sleep_scores_knn = cross_val_score(model_total_sleep_knn, X_total_sleep, y_total_sleep, cv=5, scoring='neg_mean_squared_error')
total_sleep_rmse_cv_knn = np.sqrt(-total_sleep_scores_knn.mean())

# Perform k-fold cross-validation for Dreaming prediction using KNN
dreaming_scores_knn = cross_val_score(model_dreaming_knn, X_dreaming, y_dreaming, cv=5, scoring='neg_mean_squared_error')
dreaming_rmse_cv_knn = np.sqrt(-dreaming_scores_knn.mean())

print("Cross-validated RMSE for TotalSleep prediction using KNN:", total_sleep_rmse_cv_knn)
print("Cross-validated RMSE for Dreaming prediction using KNN:", dreaming_rmse_cv_knn)


# Define Bayesian Regression models
model_total_sleep_bayesian = BayesianRidge()
model_dreaming_bayesian = BayesianRidge()

# Train Bayesian Regression models
model_total_sleep_bayesian.fit(X_total_sleep_train, y_total_sleep_train)
model_dreaming_bayesian.fit(X_dreaming_train, y_dreaming_train)

# Predict using Bayesian Regression models
y_total_sleep_pred_bayesian = model_total_sleep_bayesian.predict(X_total_sleep_test)
y_dreaming_pred_bayesian = model_dreaming_bayesian.predict(X_dreaming_test)

# Calculate RMSE for Bayesian Regression models
rmse_total_sleep_bayesian = mean_squared_error(y_total_sleep_test, y_total_sleep_pred_bayesian, squared=False)
rmse_dreaming_bayesian = mean_squared_error(y_dreaming_test, y_dreaming_pred_bayesian, squared=False)

print("RMSE for TotalSleep prediction using Bayesian Regression:", rmse_total_sleep_bayesian)
print("RMSE for Dreaming prediction using Bayesian Regression:", rmse_dreaming_bayesian)

# Perform k-fold cross-validation for TotalSleep prediction using Bayesian Regression
total_sleep_scores_bayesian = cross_val_score(model_total_sleep_bayesian, X_total_sleep, y_total_sleep, cv=5, scoring='neg_mean_squared_error')
total_sleep_rmse_cv_bayesian = np.sqrt(-total_sleep_scores_bayesian.mean())

# Perform k-fold cross-validation for Dreaming prediction using Bayesian Regression
dreaming_scores_bayesian = cross_val_score(model_dreaming_bayesian, X_dreaming, y_dreaming, cv=5, scoring='neg_mean_squared_error')
dreaming_rmse_cv_bayesian = np.sqrt(-dreaming_scores_bayesian.mean())

print("Cross-validated RMSE for TotalSleep prediction using Bayesian Regression:", total_sleep_rmse_cv_bayesian)
print("Cross-validated RMSE for Dreaming prediction using Bayesian Regression:", dreaming_rmse_cv_bayesian)


# <div style="background-color:#ffd700; padding: 20px;">
# 
# ## final Thoughts
# ### Linear Regression vs. SVR vs. Polynomial Regression:
# - **TotalSleep Prediction:**
#   - Linear Regression and SVR exhibit similar RMSE values, indicating comparable predictive performance. However, Polynomial Regression shows significantly higher RMSE and cross-validated RMSE, suggesting overfitting or inadequacy of the polynomial features.
# - **Dreaming Prediction:**
#   - Similar to TotalSleep prediction, Linear Regression and SVR perform similarly in terms of RMSE. However, Polynomial Regression again demonstrates higher RMSE values, indicating potential overfitting.
# 
# ### Random Forest vs. Regression Trees vs. GBM:
# - **TotalSleep Prediction:**
#   - Random Forest and GBM demonstrate lower RMSE values compared to Regression Trees, suggesting better predictive performance. However, GBM exhibits slightly better performance than Random Forest in terms of both RMSE and cross-validated RMSE.
# - **Dreaming Prediction:**
#   - Similar to TotalSleep prediction, Random Forest and GBM outperform Regression Trees in terms of RMSE. However, GBM shows slightly better performance than Random Forest in terms of RMSE and cross-validated RMSE.
# 
# ### KNN vs. Bayesian Regression:
# - **TotalSleep Prediction:**
#   - KNN and Bayesian Regression exhibit comparable performance in terms of RMSE, with Bayesian Regression showing slightly lower RMSE values. However, KNN demonstrates higher cross-validated RMSE compared to Bayesian Regression.
# - **Dreaming Prediction:**
#   - Similarly, KNN and Bayesian Regression demonstrate comparable performance in terms of RMSE, with Bayesian Regression showing slightly lower RMSE values. However, KNN exhibits higher cross-validated RMSE compared to Bayesian Regression.
# 
# ### Overall Perspective:
# - **For TotalSleep Prediction:**
#   - GBM appears to be the top-performing model, followed closely by Random Forest and SVR. Polynomial Regression shows poor performance due to potential overfitting.
# - **For Dreaming Prediction:**
#   - GBM also emerges as the top-performing model, with Random Forest and SVR showing competitive performance. Polynomial Regression again exhibits poor performance due to overfitting.
# 
# ### Conclusion:
# - Based on the analysis, it is evident that ensemble methods like Random Forest and GBM, along with SVR, tend to outperform other models in predicting both 'TotalSleep' and 'Dreaming' variables. However, the choice of the best-performing model may depend on specific requirements such as computational complexity, interpretability, and the trade-off between bias and variance. Further optimization and fine-tuning of the models may be necessary to achieve optimal predictive performance.
# 
# te metrics.

# ### Model justification & Evaluation

# #### Residual Analysis for Linear Regression

# In[135]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[136]:


data = copy.deepcopy(df)


# In[137]:


# 'TotalSleep' as target and other columns are features
X = data[['LogBodyWt', 'LogBrainWt', 'BrainBodyRatio','Gestation','Danger','LifeSpan']]  # Include other relevant features
y = data['TotalSleep']


# In[138]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[139]:


# Initialize the model
model = LinearRegression()


# In[140]:


# Fit the model on the training data
model.fit(X_train, y_train)


# In[141]:


# Predict on the test data
y_pred = model.predict(X_test)


# In[142]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[143]:


print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# In[144]:


min_sleep = data['TotalSleep'].min()
max_sleep = data['TotalSleep'].max()

print(f"Minimum Total Sleep: {min_sleep} hours")
print(f"Maximum Total Sleep: {max_sleep} hours")


# In[145]:


# 'data' is the DataFrame and 'model' is the trained LinearRegression model

# Scatter plot of the actual data
plt.scatter(data['LogBodyWt'], data['TotalSleep'], color='blue', label='Actual Data')


# In[146]:


# Calculate the mean values for the other features
mean_brainbody_ratio = data['BrainBodyRatio'].mean()
mean_logbrainwt = data['LogBrainWt'].mean()
mean_lifespan = data['LifeSpan'].mean()
mean_gestation = data["Gestation"].mean()
mean_danger = data["Danger"].mean()


# In[147]:


# Creating a DataFrame for predictions, ensuring the column order matches the training data
log_bodywt_range = np.linspace(data['LogBodyWt'].min(), data['LogBodyWt'].max(), 100)
prediction_df = pd.DataFrame({
    'LogBodyWt': log_bodywt_range,
    'LogBrainWt': [mean_logbrainwt] * len(log_bodywt_range),  # mean_logbrainwt defined earlier
    'BrainBodyRatio': [mean_brainbody_ratio] * len(log_bodywt_range),
    'Gestation' : [mean_gestation] * len(log_bodywt_range),
    'Danger' : [mean_danger] * len(log_bodywt_range),
        'LifeSpan' : [mean_lifespan] * len(log_bodywt_range)   # mean_brainbody_ratio defined earlie
})


# In[148]:


# Predict 'TotalSleep' using the model for each value in the range
predicted_sleep = model.predict(prediction_df)


# In[149]:


# Create the scatter plot
plt.scatter(data['LogBodyWt'], data['TotalSleep'], color='blue', label='Actual Data')

# Create the regression line plot
plt.plot(log_bodywt_range, predicted_sleep, color='red', label='Regression Line')

# Adding labels, title, and legend
plt.xlabel('Log Body Weight')
plt.ylabel('Total Sleep (hours)')
plt.title('Total Sleep vs Log Body Weight with Regression Line')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# In[150]:


from sklearn.preprocessing import PolynomialFeatures


# In[151]:


# Confirm that a linear relationship applies 
# Scatter plot with polynomial regression lines
sns.regplot(x='LogBodyWt', y='TotalSleep', data=data, order=1, label='Linear')
sns.regplot(x='LogBodyWt', y='TotalSleep', data=data, order=2, label='Quadratic', scatter=False)
plt.legend()
plt.show()


# In[152]:


# Further test regression model using residuals analysis
residuals = y_test - y_pred


# In[153]:


# Residuals vs. Predicted Values
plt.scatter(y_pred, residuals)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()


# In[154]:


#!pip install statsmodels


# In[155]:


# Residuals vs. Predicted Values with Lowess Line
from statsmodels.nonparametric.smoothers_lowess import lowess

# Calculate lowess line
fraction = 0.3  # Fraction of data to use for smoothing
lowess_line = lowess(residuals, y_pred, frac=fraction)

# Plot
plt.scatter(y_pred, residuals)
plt.plot(lowess_line[:, 0], lowess_line[:, 1], color='red', lw=1)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values with Lowess Line')
plt.show()


# In[156]:


# Histogram of Residuals
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.show()


# ### Residual analysis for Random Forest

# In[157]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 'TotalSleep' as target and other columns are features
X = data[['LogBodyWt', 'LogBrainWt', 'BrainBodyRatio','Gestation','Danger','LifeSpan']]  # Include other relevant features
y = data['TotalSleep']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=11)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

min_sleep = data['TotalSleep'].min()
max_sleep = data['TotalSleep'].max()

print(f"Minimum Total Sleep: {min_sleep} hours")
print(f"Maximum Total Sleep: {max_sleep} hours")




# In[158]:


# 'data' is the DataFrame and 'model' is the trained LinearRegression model

# Scatter plot of the actual data
plt.scatter(data['LogBodyWt'], data['TotalSleep'], color='blue', label='Actual Data')


# In[159]:


# Calculate the mean values for the other features
mean_brainbody_ratio = data['BrainBodyRatio'].mean()
mean_logbrainwt = data['LogBrainWt'].mean()

# Creating a DataFrame for predictions, ensuring the column order matches the training data
log_bodywt_range = np.linspace(data['LogBodyWt'].min(), data['LogBodyWt'].max(), 100)
prediction_df = pd.DataFrame({
    'LogBodyWt': log_bodywt_range,
    'LogBrainWt': [mean_logbrainwt] * len(log_bodywt_range),  # mean_logbrainwt defined earlier
    'BrainBodyRatio': [mean_brainbody_ratio] * len(log_bodywt_range)  # mean_brainbody_ratio defined earlier
})


# In[161]:


# Further test regression model using residuals analysis
residuals = y_test - y_pred


# In[162]:


# Residuals vs. Predicted Values
plt.scatter(y_pred, residuals)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()


# In[163]:


# Residuals vs. Predicted Values with Lowess Line
from statsmodels.nonparametric.smoothers_lowess import lowess

# Calculate lowess line
fraction = 0.3  # Fraction of data to use for smoothing
lowess_line = lowess(residuals, y_pred, frac=fraction)


# In[164]:


# Plot
plt.scatter(y_pred, residuals)
plt.plot(lowess_line[:, 0], lowess_line[:, 1], color='red', lw=1)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values with Lowess Line')
plt.show()


# In[165]:


# Histogram of Residuals
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.show()

