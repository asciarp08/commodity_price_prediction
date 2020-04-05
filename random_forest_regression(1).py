# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('commodity1.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4:5].values

# Encoding categorical data(region)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder1 = LabelEncoder()
X[:, 3] = labelencoder1.fit_transform(X[:, 3])
onehotencoder1 = OneHotEncoder(categories = 'auto')
X = onehotencoder1.fit_transform(X,y=None).toarray

# Encoding categorical data(commodity)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder2 = LabelEncoder()
X[:, 1] = labelencoder2.fit_transform(X[:, 1])
onehotencoder2 = OneHotEncoder(categories = 'auto')
X = onehotencoder2.fit_transform(X,y=None).toarray()


# Encoding categorical data(center)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder3 = LabelEncoder()
X[:, 2] = labelencoder3.fit_transform(X[:, 2])
onehotencoder3 = OneHotEncoder(categories = 'auto')
X = onehotencoder3.fit_transform(X,y=None).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

'''# Predicting a new result
z_pred = ["20111997","MUMBAI","Tur dal","NORTH"]
z_pred = sc_X.fit_transform(z_pred)
y_pred = regressor.predict(z_pred)'''
# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
#To compare y_test and y_pred
y_test = sc_y.inverse_transform(y_test)


# Saving model to disk
pickle.dump(regressor, open('price.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('price.pkl','rb'))
print(model.predict([["MUMBAI","Tur dal","NORTH"]]))

'''# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()'''
