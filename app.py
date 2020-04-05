import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():


    # Random Forest Regression

    # Importing the libraries 
    import pandas as pd

    # Importing the dataset
    data_df = pd.read_csv('India_Key_Commodities_Retail_Prices_1997_2015.csv')
    data_df.drop(['Date'],axis=1,inplace=True)
    data_df.drop(['Country'], axis = 1, inplace = True)
    # Encoding categorical data(region)
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    encode1 = LabelEncoder()
    encode1.fit(data_df.Region.unique())
    data_df['Region_Encoded'] = encode1.transform(data_df.Region)
    data_df.drop(['Region'], axis=1, inplace=True)

    cols = data_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data_df = data_df[cols]

    # Encoding categorical data(commodity)
    # from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    encode2 = LabelEncoder()
    encode2.fit(data_df.Commodity.unique())
    data_df['Commodity_Encoded'] = encode2.transform(data_df.Commodity)
    data_df.drop(['Commodity'], axis=1, inplace=True)

    cols = data_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data_df = data_df[cols]


    # Encoding categorical data(center)
    # from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    y_train = y_train.ravel()
    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)

    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    #1
    # final_features = []
    # center = request.args.get('Center')
    # final_features.append(int(encode3.transform([center])))

    # # final_features = onehotencoder1.fit_transform(final_features).toarray()
    #  #1
    # commodity = request.args.get('Commodity')
    # final_features.append(int(encode2.transform([commodity])))
    # # final_features = onehotencoder2.fit_transform(final_features).toarray()
    #  #1
    # region = request.args.get('Region')
    # final_features.append(int(encode1.transform([region])))
    final_feature = []

    center = "LUCKNOW"
    final_feature.append(int(encode3.transform([center])))
  
    commodity = "Tea Loose"
    final_feature.append(int(encode2.transform([commodity])))
    
    region = "WEST"
    final_feature.append(int(encode1.transform([region])))

    # final_features = onehotencoder3.fit_transform(final_features).toarray()
    final_feature = sc_X.transform(np.array(final_feature).reshape(1,-1))
    prediction = regressor.predict(final_feature)
 
    prediction = sc_y.inverse_transform(prediction)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Expected Commodity Price should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run( debug=True)