import pandas as pd 
import numpy as np
from pandas import plotting
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from statsmodels.formula.api import ols
import urllib
import os
import seaborn

## loading in the cvs data file
brain = pd.read_csv(r'data\brain_size.csv', sep=';', na_values=".")
brain

## creating dataframes [df] through numpy arrays
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

## exposing the df created from numpy arrays
pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})  

####### Manipulating Data #######

## reviewing basic information about the csv data 
brain.shape ## reveals # of columns and # of rows 
brain.columns ## reveal name oof columns 
brain.describe() ## quick overview of the dataset
print(brain['Gender']) ## printing only the values of the 'Gender' column 
brain[brain['Gender'] == 'Female']['VIQ'].mean() ## finding all the females within data then getting the mean of the VIQ values

## splitting data based on categorical values
groupby_gender = brain.groupby('Gender') ## grouping the data based on the 'Gender' column 
for gender, value in groupby_gender['VIQ']: 
    print((gender, value.mean())) ## finding all the female and male within data then getting the mean of the VIQ values
groupby_gender.mean() ## getting the mean of each column based on the 'Gender' values 

##### Exercise ######

## What is the mean value for VIQ for the full population?
brain['VIQ'].mean()
## How many males/females were included in this study?
## Hint use ‘tab completion’ to find out the methods that can be called, instead of ‘mean’ in the above example.
groupby_gender = brain.groupby('Gender')
groupby_gender.count()
## What is the average value of MRI counts expressed in log units, for males and females? 
for gender, value in groupby_gender['MRI_Count']:
    print((gender, value.mean()))

##### Ploting Data #####

## plotting the data based on certain columns 
plotting.scatter_matrix(brain[['Weight', 'Height', 'MRI_Count']])   
plotting.scatter_matrix(brain[['PIQ', 'VIQ', 'FSIQ']])   

##### Exercise #####

## Plot the scatter matrix for males only, and for females only. Do you think that the 2 sub-populations correspond to gender?
plotting.scatter_matrix(brain[['VIQ', 'MRI_Count', 'Height']],
                        c=(brain['Gender'] == 'Female'), marker='o',
                        alpha=1, cmap='winter')
fig = plt.gcf()
fig.suptitle("blue: male, green: female", size=13)
plt.show()

##### Hypothesis Testing #####

## 1-sample t-test: testing the value of a population mean 
stats.ttest_1samp(brain['VIQ'], 0)   
## 2-sample t-test: testing for difference across population 
female_viq = brain[brain['Gender'] == 'Female']['VIQ']
male_viq = brain[brain['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq) ## We have seen above that the mean VIQ in the male and female populations were different. To test if this is significant
## Paired tests: repeated measurements on the same individuals
stats.ttest_ind(brain['FSIQ'], brain['PIQ']) ## testing if FISQ and PIQ are significantly different
## approach of line 66 forgets the links between observations: 'FSIQ' and 'PIQ' are measured on same individual
stats.ttest_rel(brain['FSIQ'], brain['PIQ']) ## this appoarch has the removes the variance due to inter-subject variability since its  confounding via a “paired test”, or “repeated measures test”
## line 68 approach is equivalent to a 1-sample test on the difference
stats.ttest_1samp(brain['FSIQ'] - brain['PIQ'], 0)
## T-tests assume Gaussian errors. We can use a Wilcoxon signed-rank test, that relaxes this assumption:
stats.wilcoxon(brain['FSIQ'], brain['PIQ'])   
## corresponding test in the non paired case is the Mann–Whitney U test, scipy.stats.mannwhitneyu() 

##### Exercise ##### 

## Test the difference between weights in males and females.
stats.ttest_1samp(brain.dropna()['Weight'], 0) ## dropna() drops the null values
female_weight = brain.dropna()[brain['Gender'] == 'Female']['Weight']
male_weight = brain.dropna()[brain['Gender'] == 'Male']['Weight']
stats.ttest_ind(female_weight, male_weight)
## Use non parametric statistics to test the difference between VIQ in males and females.
female_viq = brain.dropna()[brain['Gender'] == 'Female']['VIQ']
male_viq = brain.dropna()[brain['Gender'] == 'Male']['VIQ']
scipy.stats.mannwhitneyu(female_viq, male_viq)
## Conclusion: we find that the data does not support the hypothesis that males and females have different VIQ.

##### Linear models, multiple factors, and analysis of variance #####

## generating simulated data according to model 
x = np.linspace(-5, 5, 20)
np.random.seed(1)
## normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
## Create a data frame containing all the relevant variables
data = pd.DataFrame({'x': x, 'y': y})
## specifying an OLS model and fitting it 
model = ols("y ~ x", data).fit()
## inspecting the various statistics derived from the fit 
print(model.summary())  

##### Exercise ##### 
## Retrieve the estimated parameters from the model above. Hint: use tab-completion to find the relevent attribute.
x = np.linspace(-5, 5, 20)
np.random.seed(1)
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
data = pd.DataFrame({'x': x, 'y': y})
model = ols("y ~ x", data).fit()
print(model.summary()) 

## categorical variables: comparing groups or multiple groups 
## a comparison between IQ of male and female using a linear model 
model = ols("VIQ ~ Gender + 1", brain).fit()
print(model.summary()) 

### Tips on Specifying Model ### 
## Forcing categorical: the ‘Gender’ is automatically detected as a categorical variable, and thus each of its different values are treated as different entities.
## An integer column can be forced to be treated as categorical using:
### model = ols('VIQ ~ C(Gender)', data).fit() ### 
## Intercept: We can remove the intercept using - 1 in the formula, or force the use of an intercept using + 1.

## Link to t-tests between different FSIQ and PIQ

## To compare different types of IQ, we need to create a “long-form” table, listing IQs, where the type of IQ is indicated by a categorical variable:
data_fisq = pd.DataFrame({'iq': brain['FSIQ'], 'type': 'fsiq'})
data_piq = pd.DataFrame({'iq': brain['PIQ'], 'type': 'piq'})
data_long = pd.concat((data_fisq, data_piq))
print(data_long)
model = ols("iq ~ type", data_long).fit()
print(model.summary())  
stats.ttest_ind(brain['FSIQ'], brain['PIQ'])   
## the same values for t-test and corresponding p-values for the effect of the type of iq than the previous t-test were retrieved

##### Multiple Regression: including multiple factors ##### 
iris = pd.read_csv(r'data\iris.csv')
model = ols('sepal_width ~ name + petal_length', iris).fit()
print(model.summary()) 

##### Post-hoc hypothesis testing: analysis of variance (ANOVA) ##### 

##### Exercise #####
## Going back to the brain size + IQ data, test if the VIQ of male and female are different after removing the effect of brain size, height and weight.
model = ols('VIQ ~ Gender + MRI_Count + Height', brain).fit()
print(model.summary())
print(model.f_test([0, 1, 0, 0]))

##### More visualization: seaborn for statistical exploration ##### 

if not os.path.exists('wages.txt'):
    # Download the file if it is not present
    urllib.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages',
                       'wages.txt')
# Give names to the columns
names = [
    'EDUCATION: Number of years of education',
    'SOUTH: 1=Person lives in South, 0=Person lives elsewhere',
    'SEX: 1=Female, 0=Male',
    'EXPERIENCE: Number of years of work experience',
    'UNION: 1=Union member, 0=Not union member',
    'WAGE: Wage (dollars per hour)',
    'AGE: years',
    'RACE: 1=Other, 2=Hispanic, 3=White',
    'OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other',
    'SECTOR: 0=Other, 1=Manufacturing, 2=Construction',
    'MARR: 0=Unmarried,  1=Married',
]
short_names = [n.split(':')[0] for n in names]
data = pd.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None,
                       header=None)
data.columns = short_names
# Log-transform the wages, because they typically are increased with
# multiplicative factors
data['WAGE'] = np.log10(data['WAGE'])
print(data)

## an intuition on the interactions between continuous variables using seaborn.pairplot() to display a scatter matrix
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg') 
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg', hue='SEX') 
## resetting the display back to default
plt.rcdefaults()

##### Implot: plotting a univariate regression #####

seaborn.lmplot(y='WAGE', x='EDUCATION', data=data)

##### Testing for Interactiosn ##### 
result = sm.ols(formula='WAGE ~ EDUCATION + GENDER + EDUCATION * GENDER',
                data=data).fit()
print(result.summary())