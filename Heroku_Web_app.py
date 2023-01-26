# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model_house.pkl', 'rb'))

@app.route('/')
def home(): 
    render_template('index.html')
    
@app.route('/predict', methods = ['POST'])
def predict():
    X_features = [int(x) for x in request.form.values()]
    f_X_features = [np.array(X_features)]
    prediction = model.predict(f_X_features)
    result = round(prediction[0], 2)
    return render_template('index.html', 
                           prediction_t = 'House price should be $ {}'.format(result))

if __name__ == '__main__': 
    app.run()