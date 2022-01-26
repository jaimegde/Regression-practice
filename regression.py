#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:35:52 2021

@author: jaimegde
"""

from plotnine import *
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
boston.keys()
boston.DESCR

#Data
X = pd.DataFrame(boston.data)
X.columns = boston.feature_names
X.head(3)

Y = pd.DataFrame(boston.target)
Y.head(3)

#Train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, shuffle = True, random_state = 42)

#Model
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
regr.coef_
regr.intercept_ 

coefs = pd.DataFrame()
coefs["Variables"] = X_train.columns
coefs["Coeff"] = regr.coef_[0]
coefs

#Predictions
predictions = regr.predict(X_test)
predictions.shape

#Accuracy
mean_absolute_error(Y_test, predictions) #how many units error
mean_squared_error(Y_test, predictions)

#tree
#%%
#####################
### REG.3
######################
from plotnine import *
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston = load_boston()
boston.keys()
boston.DESCR

# Data:
X = pd.DataFrame(boston.data)
X.columns = boston.feature_names
Y = pd.DataFrame(boston.target)

# Train/Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, shuffle = True)


# Model
regr = DecisionTreeRegressor(criterion="mse",max_depth=10,min_samples_leaf=5)
regr.fit(X_train, Y_train)

# Predictions:
predictions = regr.predict(X_test)

# Accuracy:
allinfo = pd.DataFrame()
allinfo["Real"] = Y_test.iloc[:,0]
allinfo["Prediction"] = predictions
allinfo["ErrorRatio"] = np.abs(allinfo.Real - allinfo.Prediction)/allinfo.Real
allinfo.head(3)

allinfo["KindOfPrediction"] = "Bad"
allinfo.loc[(allinfo.ErrorRatio < 0.05) & (allinfo.Real > 25), 'KindOfPrediction'] = "Good"
allinfo.loc[(allinfo.ErrorRatio < 0.10) & (allinfo.Real <= 25), 'KindOfPrediction'] = "Good"
allinfo.KindOfPrediction.value_counts()
allinfo.KindOfPrediction.value_counts(normalize=True)


# Graph1
most_imp_variables = pd.DataFrame()
most_imp_variables["Variable"] = X_test.columns
most_imp_variables["Importance"] = regr.feature_importances_
ggplot(aes(x="Variable",y="Importance"),most_imp_variables)+geom_bar(stat="identity")+coord_flip()

# Graph2
ggplot(aes(x="Real",y="Prediction",color="KindOfPrediction"),allinfo) + geom_point() + geom_abline(slope=1,intercept=0)

# Graph3
allinfo["KindOfHouse"] = "Expensive"
allinfo.loc[allinfo.Real < 25, 'KindOfHouse'] = "Cheap"
ggplot(aes(x="ErrorRatio",color="KindOfHouse"),allinfo) + geom_density()
# or with histogram and facet_wrap

sol = pd.DataFrame(columns = ["cheap bound", "expensive bound", "accuracy"])

for i in np.arange(0,1,0.05):
    for j in np.arange(0,1,0.05):
        allinfo["KindOfPrediction"] = "Bad"
        allinfo.loc[(allinfo.ErrorRatio < i) & (allinfo.Real > 25), "KindOrPrediction"] = "Good"
        allinfo.loc[(allinfo.ErrorRatio < j) & (allinfo.Real <= 25), "KindOfPrediction"] = "Good"
        acc = sum(allinfo.KindOfPrediction=="Good")/len(allinfo.KindOfPrediction)
        sol.loc[len(sol)]=[i,j,acc]
        

#1

ggplot(aes(x="cheap bound", y="expensive bound", color="accuracy"), sol) + geom_point()

ggplot(aes(x="cheap bound", y="expensive bound", fill="accuracy"), sol) + geom_tile()



sol90 = sol.loc[sol.accuracy >= 0.90,:]

ggplot(aes(x="cheap bound", y="expensive bound", color="accuracy"), sol90) + geom_point()

ggplot(aes(x="cheap bound", y="expensive bound", fill="accuracy"), sol90) + geom_tile()



#%%
from plotnine import *
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

poke = pd.read_csv("pokemon_description.csv")

poke.shape
poke.columns
X = poke.loc[:, ['HP', 'Attack', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']]
Y = poke.loc[:,'Defense']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size = 0.2, shuffle = True)

regr = DecisionTreeRegressor(criterion="mse",max_depth=10,min_samples_leaf=5)
regr.fit(X_train, Y_train)

predictions = regr.predict(X_test)

mean_absolute_error(Y_test, predictions) #how many units error
mean_squared_error(Y_test, predictions)

allinfo = pd.DataFrame()
allinfo["Real"] = Y_test.iloc[0:]
allinfo["Prediction"] = predictions
allinfo["ErrorRatio"] = np.abs(allinfo.Real - allinfo.Prediction)/allinfo.Real
allinfo.head(3)

allinfo["KindOfHouse"] = "Bad"
allinfo.loc[allinfo.ErrorRatio < 0.10, 'KindOfHouse'] = "Good"

allinfo.KindOfHouse.value_counts()
allinfo.KindOfHouse.value_counts(normalize=True)



























