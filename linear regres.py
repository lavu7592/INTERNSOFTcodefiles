# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:12:42 2022

@author: 91831
"""
#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt

#Reading the data from your files
lav = pd.read_csv('advertising.csv')
lav.head()

#visualizing the data
fig , axs = plt.subplots(1,3,sharey = True)
lav.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
lav.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
lav.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


#Creating x and y for linear regression
feature_cols = ['TV']
X = lav[feature_cols]
y = lav.Sales

#importing linear regression algo for simple linear reg
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

result = 6.974821488229891+0.05546477*50
print(result)


#creating a dataframe with min and max value of the table
X_new = pd.DataFrame({'TV':[lav.TV.min(),lav.TV.max()]})
X_new.head()


preds = lr.predict(X_new)
preds


lav.plot(kind='scatter',x='TV',y='Sales')

plt.plot(X_new,preds,c='red',linewidth = 3)


import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales ~ TV', data=lav).fit()
lm.conf_int()

#finding the probability values
lm.pvalues

#finding the r squared values
lm.rsquared

#multi linear regression
feature_cols = ['TV','Radio','Newspaper']
X = lav[feature_cols]
y = lav.Sales



lr = LinearRegression()
lr.fit(X,y)


print(lr.intercept_)
print(lr.coef_)


import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales ~ TV+Radio+Newspaper', data=lav).fit()
lm.conf_int()
lm.summary()


import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales ~ TV+Radio', data=lav).fit()
lm.conf_int()
lm.summary()






























