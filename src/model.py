#!/usr/bin/env python
"""
model training
"""

import time,os,re,joblib
#from datetime import date
#from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from logger import update_predict_log, update_train_log
from helper import fetch_ts, engineer_features

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time series"

def _model_train(df, tag, test=False):
    """
    example function to train model
    
    the 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies the use of the 'test' log file 
    """


    # start timer for runtime
    time_start = time.time()
    
    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        y=y[-n_samples:]
        X=X.iloc[-n_samples:]
        dates=dates[-n_samples:]
        
    # perform a train-test split
    # since this is time series, last rows should be test set
    n_test = int(np.round(0.25 * X.shape[0]))
    X_train = X.iloc[:-n_test]
    X_test = X.iloc[-n_test:]
    y_train = y[:-n_test]
    y_test = y[-n_test:]
    
    # train a random forest model
    param_grid_rf = {
    'rf__criterion': ['mse','mae'],
    'rf__n_estimators': [10,20,30,40,50,60,70],
    'rf__max_depth': [5,3,2],
    'rf__max_samples': [0.7,0.8,0.9]
    }

    pipe_rf = Pipeline(steps=[('rf', RandomForestRegressor())])
    
    # create time series split for cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=tscv, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse = round(np.sqrt(mean_squared_error(y_test,y_pred)))
    
    ## retrain using all data
    grid.fit(X, y)
    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join("..",MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join("..",MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(grid,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    update_train_log(tag,(str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
  

def model_train(data_dir, test=False):
    """
    function to train model given a df
    
    'test' - can be used to subset data simulating a train
    """
    
    if not os.path.isdir(os.path.join("..",MODEL_DIR)):
        os.mkdir(os.path.join("..",MODEL_DIR))

    if test:
        print("... test flag on")
        print("...... subsetting data")
        print("...... subsetting countries")
        
    ## fetch time series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data set
    for country, df in ts_data.items():
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        _model_train(df, country, test=test)


def model_load(prefix='sl', data_dir=None, training=True):
    """
    example function to load model
    
    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join("..","data","cs-train")
    
    models = [f for f in os.listdir(os.path.join("..","models")) if re.search(prefix,f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join("..","models",model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X,y,dates = engineer_features(df,training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X":X,"y":y,"dates": dates}
        
    return(all_data, all_models)


def model_predict(country, year, month, day, all_models=None, test=False):
    """
    example function to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        all_data, all_models = model_load(training=False)
    
    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    ## load data
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))
    print(target_date)

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,data['dates'][0],data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]
    
    ## sanity check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)


    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
    update_predict_log(country,y_pred,y_proba,target_date,
                       runtime, MODEL_VERSION, test=test)
    
    return({'y_pred':y_pred,'y_proba':y_proba})


if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    ## train the model
    print("TRAINING MODELS")
    data_dir = os.path.join("..","data","cs-train")
    model_train(data_dir,test=True)

    ## load the model
    print("LOADING MODELS")
    all_data, all_models = model_load()
    print("... models loaded: ",",".join(all_models.keys()))

    ## test predict
    print("PREDICTING")
    country='all'
    year='2018'
    month='01'
    day='05'
    result = model_predict(country,year,month,day)
    print(result)
