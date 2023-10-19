# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:25:32 2023

@author: DELL
"""

import pandas as pd
import numpy as np
data = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Big%20Sales%20Data.csv')
data.head()
data.describe
data.columns
data.shape
data['Item_Weight'].fillna(data.groupby(['Item_Type'])['Item_Weight'].transform('mean'), inplace=True)
import seaborn as sns
sns.pairplot(data)
data[['Item_Identifier']].value_counts()
data.columns
data[['Item_Fat_Content']].value_counts()
data.replace({'Item_Fat_Content': {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}}, inplace = True)
data[['Item_Fat_Content']].value_counts()
data.replace({'Low Fat': 0, 'Regular': 1}, inplace = True)
data[['Item_Fat_Content']].value_counts()
data[['Item_Identifier']].value_counts()
data.replace({'Item_Type': {'Fruits and Vegetables': 0, 'Snack Foods': 0, 'Household':1, 'Frozen Foods': 0, 'Dairy': 0, 'Baking Goods': 0, 
                            'Canned': 0, 'Health and Hygiene': 1, 'Meat': 0, 'Soft Drinks': 0, 'Breads':0, 'Hard Drinks': 0, 'Others': 2,
                            'Starchy Foods': 0, 'Breakfast': 0, 'Seafood': 0}}, inplace = True)
data[['Item_Type']].value_counts()
#data[['Outlet_Identifier']].value_counts()
data.replace({'Outlet_Identifier':{'OUT027': 0,'OUT013': 1, 'OUT035': 2, 'OUT046': 3, 'OUT049': 4, 'OUT045': 5, 'OUT018': 6, 'OUT017': 7,
                                   'OUT010': 8, 'OUT019': 9}}, inplace = True)
data[['Outlet_Identifier']].value_counts()
data['Outlet_Size'].value_counts()
data.replace({'Outlet_Size': {'Small': 0, 'Medium': 1, 'High': 2}}, inplace = True)
data['Outlet_Size'].value_counts()
data.replace({'Outlet_Location_Type':{'Tier 1' :0, 'Tier 2' : 1, 'Tier 3' :2}}, inplace = True)
data[['Outlet_Location_Type']].value_counts()
data.replace({'Outlet_Type' : {'Grocery Store' : 0, 'Supermarket Type1': 1,'Supermarket Type2': 2,'Supermarket Type3': 3}}, inplace = True)
data[['Outlet_Type']].value_counts()
data.head()
data.columns

y = data['Item_Outlet_Sales']
x = data.drop(['Item_Identifier', 'Item_Outlet_Sales'], axis = 1)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_std = data[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']]
x_std = ss.fit_transform(x_std)
x_std

x[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']] = pd.DataFrame(x_std, columns = [['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']])
x

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 2529)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 2529)
model.fit(x_train, y_train)  
y_pred = model.predict(x_test)
y_pred  

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
mean_absolute_error(y_test, y_pred)
mean_absolute_percentage_error(y_test, y_pred) 
r2_score(y_test, y_pred) 

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted price')
plt.title('Actual Price vs Predicted Price')
plt.show()                                                               

