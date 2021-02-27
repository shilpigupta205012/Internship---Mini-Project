# Internship---Mini-Project

## Introduction about the project
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.
Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
 The data has missing values as some stores do not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.
 
## Hypothesis 

## Loading Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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

###The train data consists of 8,523 training examples with 12 features.
###The test data consists of 5,681 training examples with 11 features

train.columns
test.columns

train.info()
### there are 4 float type 6 object type and 1 int type.
train.describe()

 
