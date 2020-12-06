#!/usr/bin/env python
"""
logging functions
"""

import time,os,re,csv,sys,uuid,joblib
from datetime import date




def update_train_log(tag,period,rmse,runtime,MODEL_VERSION,MODEL_VERSION_NOTE,test=False):
    """
    function to update training log file
    """

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    # naming for log files
    today = date.today()
    if test:
        logfile = os.path.join("logs", 
                  "train-test-{}-{}-{}.log").format(today.year, today.month, today.day)
    else:
        logfile = os.path.join("logs", 
                  "train-{}-{}-{}.log").format(today.year, today.month, today.day)

    # update log file with data
    header = ["unique_id", "timestamp", "rmse", "runtime", 
              "model_version", "model_version_note"]
    write_header = False
    logdata = [uuid.uuid4(), time.time(), rmse, runtime, MODEL_VERSION, MODEL_VERSION_NOTE]
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=",")
        if write_header:
            mywriter.writerow(header)
        mywriter.writerow(logdata)


def update_predict_log(country, y_pred, y_proba, target_date, runtime, MODEL_VERSION, test=False):
    """
    function to update prediction log file
    """

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    # naming for log files
    today = date.today()
    if test:
        logfile = os.path.join("logs", 
                  "predict-test-{}-{}-{}.log").format(today.year, today.month, today.day)
    else:
        logfile = os.path.join("logs", 
                  "predict-{}-{}-{}.log").format(today.year, today.month, today.day)

    # update log file with data
    header = ["unique_id", "timestamp", "country", "y_pred", "y_proba", 
              "target_date", "runtime", "model_version"]
    write_header = False
    logdata = [uuid.uuid4(), time.time(), country, y_pred, y_proba, 
               target_date, runtime, MODEL_VERSION]
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=",")
        if write_header:
            mywriter.writerow(header)
        mywriter.writerow(logdata)

if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    from model import MODEL_VERSION, MODEL_VERSION_NOTE
    
    # logging for training
    update_train_log('eire', ('2017-11-29', '2019-05-31'), {'rmse':1.2345}, '00:01:02', MODEL_VERSION, MODEL_VERSION_NOTE, test=True)
    
    # logging for predicting
    update_predict_log('eire', 1.23, 4.56, '2019-05-31', '00:01:02', MODEL_VERSION, test=True)

