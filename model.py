# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:47:43 2021

@author: shilp
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
os.chdir('C:/Users/shilp/Desktop/Internship')
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index = True, sort = True)
print (train.shape, test.shape, data.shape)

data.head()
data.shape
data.describe()
data.info()

data.isnull().sum()
data.nunique()




#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
categorical_columns # all are not required 
categorical_columns = [x for x in categorical_columns if x not in ["Item_Identifier","Outlet_Identifier","source"]]
categorical_columns

for col in categorical_columns:
    print ("frequency of categorical variables")
    print(col)
    print(data[col].value_counts())

import seaborn as sns
sns.distplot(data.Item_Outlet_Sales, color = "c")
plt.show()



s_profit = pd.DataFrame({'Item_Type': data.Item_Type, 'Profit': data.Item_Outlet_Sales})
s_profit.head()
# summing profit of each state
s_data = s_profit.groupby(['Item_Type'], as_index = False).sum()
s_data
sns.countplot(x = "Item_Type", data = data)
plt.xticks(rotation = 90)
plt.show()

# Outlet _Identifier
data.Outlet_Identifier.value_counts().plot(kind = "bar")


# plotting a histogram for Item_MRP variable
data['Item_MRP'].plot.hist()


# plotting a boxplot for Item_MRP variable
data['Item_MRP'].plot.box()


#Here, we can see that there is no outlier in Item_MRP variable.


#Bivariate Analysis
#For continuous - continuous variable
plot.scatter['Item_Weight','Item_MRP']


data['Item_Weight'].corr(data['Item_MRP']) 
#Correlation
# Correlation between Item_Weight and Item_MRP

data[['Item_Weight','Item_MRP']].corr()
# seaborn: Statistical data visulization. Seaborn ia a python data visulization library based on matplotlib.
# it provides a high-level interface for drawing attrative and informative statistical graphics. 

import seaborn as sns


data.corr()

# Plotting correlation between different features

cor = data.corr()
plt.figure(figsize=(16,10))
sns.heatmap(cor)

#We can see that there is a good correlation between Item_MRP and Item_Outlet_Sales

#Data Cleaning
# Check missing values:
data.apply(lambda x: sum(x.isnull()))
#Filling missing values
data.Item_Outlet_Sales = data.Item_Outlet_Sales.fillna(data.Item_Outlet_Sales.mean())
data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())
data['Outlet_Size'].value_counts()

data.Outlet_Size = data.Outlet_Size.fillna('Medium')
data.apply(lambda x: sum(x.isnull()))

# Item type combine:
data['Item_Identifier'].value_counts()
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

# Fat_Content showing redudancy of differnt types

data.Item_Fat_Content.value_counts()

# Now replace LF by Low Fat ,low fat by lf,reg by Regular

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat':'Low Fat','reg':'Regular','LF':'Low Fat'})
data.Item_Fat_Content.value_counts()

# No of years outlet is working conditon

data['Outlet_Years'] = 2018 - data['Outlet_Establishment_Year']
Mean_Visibility = data['Item_Visibility'].mean()
data['Item_Visibility_MeanRatio'] = data.apply(lambda x:x['Item_Visibility'] / Mean_Visibility,axis=1)
# As Item Id and Outlet Id


#Convert categorical into numerical
from sklearn.preprocessing import LabelEncoder

var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type']
number = LabelEncoder()
# Item_Identifier and outlet_Identifier are also useful for making prediction

data['Outlet'] = number.fit_transform(data['Outlet_Identifier'])
data['Identifier'] = number.fit_transform(data['Item_Identifier'])
for i in var_mod:
      data[i]=number.fit_transform(data[i])
data.head()
#One-Hot Coding
#Numerical and One-Hot Coding of Categorical variables
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
data.head()

data.dtypes


# Exporting Data

import warnings
warnings.filterwarnings('ignore')

# Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

# Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'], axis = 1, inplace=True)
train.drop(['source'], axis = 1, inplace = True)

# Export files as modified versions
train.to_csv("C:/Users/Hinal/Desktop/train_modified.csv", index = False)
test.to_csv("C:/Users/Hinal/Desktop/test_modified.csv", index = False)
#Model Building
# Reading modified data 
os.chdir('C:/Users/shilp/Desktop/Internship')
train2 = pd.read_csv("train_modified.csv")
test2 = pd.read_csv("test_modified.csv")
import statsmodels.api as sm
x = train2[[ 'Item_MRP', 'Item_Fat_Content_0', 'Outlet_Type_0', 'Outlet_Type_1',
       'Outlet_Type_2', 'Outlet_Type_3']]
y = train2['Item_Outlet_Sales']
x.info()

x.shape, y.shape
((8523, 6), (8523,))
 
y = np.array(y).reshape(8523,1)
#x = sm.add_constant(x)
lm = sm.OLS(y,x)
results = lm.fit()
results.summary()


import pickle
filename = 'new_model.pkl'
pickle.dump(results, open(filename, 'wb'))

# # X_train = train2.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis = 1)
# y_train = train2.Item_Outlet_Sales
# X_test = test2.drop(['Outlet_Identifier','Item_Identifier'], axis = 1)
# X = train2[['Item_MRP','Item_Weight','Identifier']]
x.head()




#Linear Regression Model
# Fitting Multiple Linear Regression to the training set

from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
LinearRegression()


import pickle
filename = 'python file 5.pkl'
pickle.dump(regressor, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))


# Predicting the test set results
y_pred = regressor.predict(x)

y_pred

# Measuring Accuracy
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import metrics
lr_accuracy = round(regressor.score(x,y) * 100,2)
lr_accuracy
r2_score(y, regressor.predict(x))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y, regressor.predict(x))))




#Random Forest Model
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=50, n_jobs=4)
regressor.fit(x, y)

RandomForestRegressor(max_depth=6, min_samples_leaf=50, n_jobs=4)

from sklearn.externals import joblib 
joblib.dump(regressor, 'filename.pkl') 

# Predicting the test set results

y_pred = regressor.predict(x)
y_pred
rf_accuracy = round(regressor.score(x,y),2)
rf_accuracy
r2_score(y, regressor.predict(x))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(y, regressor.predict(x))))


