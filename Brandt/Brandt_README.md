First steps

I've started some analysis on the data which I thought I would share, attached is a copy of my juypter notebook, the excel, and my environment settings.

( conda env create -f brandt_sleep.yml #to recreate this environment otherwise, am using Python 3.10, packages called are identified in the notebook)

I may do some more on work on this over the weekend, if yes, I will send it around.

Bon week-end,

Brandt

09/01/24 - I've created a new verision my my notebook, sleep_study_v.03.ipynb, data sourice --> cleaned_sleep_data_v.03.xlsx, env-->brandt_sleep_v03.yml
Note, here I have changed all the brain weights to Kg from gram and removed the species with zero brain weight and I have performed a regression analysis.

16/01 - I've tried running the model with the 2007 data, but that did not provide an improved result, sleep_study_2007

23/01 - Reverted back to orginal work and code, adding Residuals Analysis to confirm that a non-linear relationship does not exist, i.e., bigger animals sleep less
Order Cingulata, armidillos, are an outlier in that they sleep a lot for their size, confirmed in the literature, but no reason as for 'why'
