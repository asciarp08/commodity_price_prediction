# Random Forest Regression

# Importing the libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
data_df = pd.read_csv('India_Key_Commodities_Retail_Prices_1997_2015.csv')
data_df.drop(['Date'], axis=1, inplace=True)
data_df.drop(['Country'], axis=1, inplace=True)

# Encoding categorical data(region)

encode1 = LabelEncoder()
encode1.fit(data_df.Region.unique())
data_df['Region_Encoded'] = encode1.transform(data_df.Region)
data_df.drop(['Region'], axis=1, inplace=True)

cols = data_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_df = data_df[cols]

# Encoding categorical data(commodity)

encode2 = LabelEncoder()
encode2.fit(data_df.Commodity.unique())
data_df['Commodity_Encoded'] = encode2.transform(data_df.Commodity)
data_df.drop(['Commodity'], axis=1, inplace=True)

cols = data_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_df = data_df[cols]

# Encoding categorical data(center)

encode3 = LabelEncoder()
encode3.fit(data_df.Centre.unique())
data_df['Centre_Encoded'] = encode3.transform(data_df.Centre)
data_df.drop(['Centre'], axis=1, inplace=True)

cols = data_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
data_df = data_df[cols]

X = data_df.iloc[:, 0:3].values
y = data_df.iloc[:, 3:4].values
# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
y_train = y_train.ravel()
# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))
pickle.dump(encode1, open('encode1.pkl', 'wb'))
pickle.dump(encode2, open('encode2.pkl', 'wb'))
pickle.dump(encode3, open('encode3.pkl', 'wb'))
pickle.dump(sc_X,open('sc_X.pkl','wb'))
pickle.dump(sc_y,open('sc_y.pkl', 'wb'))