# Internship---Mini-Project

## Introduction about the project
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.
Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
 The data has missing values as some stores do not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.
 
## Hypothesis 
Are the items with less visibility having more sales.

## Loading Packages
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

## Importing dataset
train = pd.read_csv(r"C:\Users\shilp\Desktop\Internship\Train.csv")

test = pd.read_csv(r"C:\Users\shilp\Desktop\Internship\Test.csv")


## Data structure and content
train.head()

train.tail()

print(train.shape)

print(test.shape)

#The train data consists of 8,523 training examples with 12 features.
#The test data consists of 5,681 training examples with 11 features

train.columns

test.columns

train.info()

#there are 4 float type 6 object type and 1 int type.

## Univariate analysis
Univariate analysis is the simplest form of analyzing data. “Uni” means “one”, so in other words your data has only one variable. It doesn’t deal with causes or relationships (unlike regression ) and it’s major purpose is to describe; It takes data, summarizes that data and finds patterns in the data.


### univariate : numerical
pd.set_option('display.max_columns', None)

print('Number of trainings examples:', len(train),'\n')

train.describe()

#So, here we can see that Total count of Item_Weight is 7060 which is less than the length of the training dataset, therefore it may contains some missing values.

#The average weight of all items is 12.85kg and the maximum weight of the item is 21.3 kg. So it is clear that the stores are not selling heavy weight items.

#The maximum price of the items is 266, so we can say that the stores does not selling costly items like TV, mobile phones, laptops etc.

#Most recent store was established in 2009 and the oldest store was established in 1985.

#Average sales of items is Rs 2181 and the maximum sale is Rs 13,086.

#### making a list of numerical columns
numerical = train.select_dtypes(include = [np.number])

numerical

#### distribution of target variable

plt.figure(figsize=(12,7))

sns.distplot(train.Item_Outlet_Sales, bins = 25)

plt.xlabel("Item_Outlet_Sales")

plt.ylabel("Number of Sales")

plt.title("Item_Outlet_Sales Distribution")

print ("Skew is:", train.Item_Outlet_Sales.skew())

print("Kurtosis: %f" % train.Item_Outlet_Sales.kurt())

#Most of the stores has a maximum sales in between 450 - 3900. Only few of the stores having sales more than 6000.

correlation = numerical.corr()

correlation

correlation['Item_Outlet_Sales'].sort_values(ascending=False)

#From the above result, we can see that Item_MRP have the most positive correlation and the Item_Visibility have the lowest correlation with our target variable. 


 
