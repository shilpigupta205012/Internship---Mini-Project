## Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
%matplotlib inline #(to visualise graph instantly)
warnings.filterwarnings('ignore')
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso




## Importing Datasets

os.chdir('C:/Users/shilp/Desktop/Technolabs')
df = pd.read_csv('Train.csv')
df.head()

### Statistical information
df.describe()

### datatype of attributes
df.info()

### Check unique values in dataset
df.apply(lambda x: len(x.unique()))

## Preprocessing the dataset

### check for null values
df.isnull().sum()

### Check for cateogorical attributes
cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)
cat_col

cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
cat_col

### Print the cateogorical columns
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()

## Fill missing values
#pivot table function in pandas library
item_weight_mean = df.pivot_table(values = 'Item_Weight', index = 'Item_Identifier')
item_weight_mean

#get info of missing values
miss_bool = df['Item_Weight'].isnull()
miss_bool


#Lets fill missing values
for i, item in enumerate(df['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            df['Item_Weight'][i] = 0
df['Item_Weight'].isnull().sum()


outlet_size_mode = df.pivot_table(values = 'Outlet_Size',columns = 'Outlet_Type',aggfunc= (lambda x: x.mode()[0]))
outlet_size_mode

miss_bool = df['Outlet_Size'].isnull()
df.loc[miss_bool,'Outlet_Size'] = df.loc[miss_bool, 'Outlet_Type'].apply(lambda x:outlet_size_mode[x])

df['Outlet_Size'].isnull().sum()

sum(df['Item_Visibility']==0)

df.loc[:,'Item_Visibility'].replace([0],[df['Item_Visibility'].mean()],inplace = True)
sum(df['Item_Visibility']==0)


### Combine item fat content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat':'Low Fat','reg':'Regular','LF':'Low Fat'})
df.Item_Fat_Content.value_counts()

## Creation of new attributes
#New item type
df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
df['New_Item_Type']

df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
df['New_Item_Type'].value_counts()

df.loc[df['New_Item_Type']=='Non-Consumable','Item_Fat_Content'] = 'Non-Edible'
df['Item_Fat_Content'].value_counts()

### Create small values for establishment year
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
df['Outlet_Years']

df.head()

## Exploratory data analysis\
sns.distplot(df['Item_Weight'])

sns.distplot(df['Item_Visibility'])
#all the values is at mean- becz we filled missing values with mean

sns.distplot(df['Item_MRP'])

sns.distplot(df['Item_Outlet_Sales'])
#We need to normalised the skewed data
#Log transformation
df['Item_Outlet_Sales'] = np.log(1+df['Item_Outlet_Sales'])
sns.distplot(df['Item_Outlet_Sales'])
#Now its almost normalized

##Cateogorical Attributes
#using COuntplot

sns.countplot(df['Item_Fat_Content'])

l = list(df['Item_Type'].unique())
sns.countplot(df['Item_Type'])
chart.set_xticklabels(labels = 1, rotation = 90)

sns.countplot(df['Outlet_Establishment_Year'])

sns.countplot(df['Outlet_Size'])

sns.countplot(df['Outlet_Location_Type'])

sns.countplot(df['Outlet_Type'])

## Correation Matrix
corr = df.corr()
sns.heatmap(corr,annot = True, cmap = 'coolwarm')

## Label Encoding
le = LabelEncoder()
#Convert cateogorical into numerical for better prediction and model can process this column also
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
df.columns
cat_col = ['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type', 'New_Item_Type']
#We will convert each of them
for col in cat_col:
    df[col] = le.fit_transform(df[col])

## One Hot Encoding
#it will create a new column for each cateogory
#will improve accuracy of model but it takes time for model to be run
#instead of labeling you can go directly for one hot encoding

df = pd.get_dummies(df, columns = ['Item_Fat_Content',   'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'])
#we will not include item type because there are many items in item type and it will make model really big which may have so many features
df.head()
#now we have 26 columns

## Train - Test Split
x = df.drop(columns = ['Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']

## Model Training

def train(model, x, y):
    #train the model
    model.fit(x,y)
    
    #predict the training set
    pred = model.predict(x)
    
    #perform cross va;idation 
    cv_score = cross_val_score(model , x, y, scoring = 'neg_mean_squared_error')
    
    print("Model report")
    print("MSE:",mean_squared_error(y,pred))
    print("CV Score:",cv_score)

model = LinearRegression(normalize=True)
train(model, x, y)
coef = pd.Series(model.coef_, x.columns).sort_values()
coef.plot(kind = 'bar', title = 'Model Coefficients')

#saving model to disk
pickle.dump(model,open('model.pkl','wb'))
