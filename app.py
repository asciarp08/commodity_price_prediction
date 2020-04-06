import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
regressor = pickle.load(open('model.pkl', 'rb'))
encode1 = pickle.load(open('encode1.pkl', 'rb'))
encode2 = pickle.load(open('encode2.pkl', 'rb'))
encode3 = pickle.load(open('encode3.pkl', 'rb'))
sc_X = pickle.load(open('sc_X.pkl','rb'))
sc_y = pickle.load(open('sc_y.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    '''
    For rendering results on HTML GUI
    '''
    
    final_feature = []

    center = request.form["Center"]
    # center = "BHOPAL"
    final_feature.append(int(encode3.transform([center])))
  
    commodity = request.form["Commodity"]
    # commodity = "Sugar"
    final_feature.append(int(encode2.transform([commodity])))
    
    region = request.form["Region"]
    # region = "SOUTH"
    final_feature.append(int(encode1.transform([region])))

    # final_features = onehotencoder3.fit_transform(final_features).toarray()
    final_feature = sc_X.transform(np.array(final_feature).reshape(1,-1))
    prediction = regressor.predict(final_feature)
 
    prediction = sc_y.inverse_transform(prediction)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Expected Price : ₹ {}'.format(output))
    
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