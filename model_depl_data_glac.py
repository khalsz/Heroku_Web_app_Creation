# -*- coding: utf-8 -*-
"""Kaggle House Price Comp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aEw6sZoRMhPtjV6WK96C2Xr_YI8-lgFT

#Loading Libraries
"""
import category_encoders

import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
import sys
import xgboost
from xgboost import XGBRegressor
import os 

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

"""#Loading Data"""
os.chdir('C:/Users/khalsz/Documents/Leicester Uni Actvt/glacier internship')
train_df = pd.read_csv('data/train_house.csv')


"""#Data EDA"""



#ID column no needed for the prediction. Index is ok
train_df.drop('Id', axis = 1, inplace = True)




train_df.columns

"""#Data Cleaning """

num_val = ['float64', 'int64']
num_col = train_df.select_dtypes(include = num_val)




X_col = num_col[num_col.isna().sum()[num_col.isna().sum() < 4].index]


X_col = X_col.loc[:, ['LotArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'GrLivArea', 'GarageArea']]
X_col = X_col.apply(lambda x: x.fillna(x.mean()), axis = 0)

X_col.columns

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_col, num_col['SalePrice'], test_size=0.33, random_state=42)

"""#Modelling """

xgbr = XGBRegressor(verbosity=0) 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

xgbr.fit(X_train, y_train)


import pickle

pickle.dump(xgbr, open('model_house.pkl', 'wb'))





