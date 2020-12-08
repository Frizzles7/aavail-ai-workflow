#!/usr/bin/env python
"""
flask application
"""

import os
import numpy as np
from flask import Flask
from flask import jsonify, request
from flask import render_template
from src.model import model_train, model_load, model_predict
from src.model import MODEL_VERSION, MODEL_VERSION_NOTE

app = Flask(__name__)

@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    # confirm request data is received
    if not request.json:
        print('request data not received')
        return jsonify([])
    
    if 'country' not in request.json:
        print('country not included in request')
        return jsonify([])
        
    if 'year' not in request.json:
        print('year not included in request')
        return jsonify([])

    if 'month' not in request.json:
        print('month not included in request')
        return jsonify([])

    if 'day' not in request.json:
        print('day not included in request')
        return jsonify([])
    
    # set the test flag
    test = False
    if 'test' in request.json and request.json['test'] == 'test':
        test = True
    
    # get input parameters
    country = request.json['country']
    year = request.json['year']
    month = request.json['month']
    day = request.json['day']
    
    # get model for prediction
    data_dir = os.path.join(".","data","cs-train")
    preds = model_predict(country, year, month, day, test=test)
    
    result = {}
    for key,item in preds.items():
        if isinstance(item, np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item    
    
    return jsonify(result['y_pred'])


@app.route('/train', methods=['GET', 'POST'])
def train():

    # confirm request data is received
    if not request.json:
        print('request data not received')
        return jsonify([])

    # set the test flag
    test = False
    if 'test' in request.json and request.json['test'] == 'test':
        test = True

    data_dir = os.path.join(".","data","cs-train")
    model_train(data_dir,test=test)
    
    return jsonify(True)


@app.route('/logfile', methods=['GET'])
def logfile():
    
    # confirm request data is received
    if not request.json:
        print('request data not received')
        return jsonify([])

    logname = request.json['log']
    log_dir = os.path.join(".","logs")
    if not os.path.exists(os.path.join(log_dir, logname)):
        print('log file does not exist')
        return jsonify([])
    
    return send_from_directory(log_dir, logname, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)

