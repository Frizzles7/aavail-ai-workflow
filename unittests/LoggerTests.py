#!/usr/bin/env python
"""
logging unit tests
"""

import unittest
import os, sys
from datetime import date
from src.logger import update_train_log, update_predict_log

class LoggerTest(unittest.TestCase):

    def test_01_train(self):
        """
        test train logging
        """
        
        today = date.today()
        logfile = os.path.join(".","logs", 
                  "train-test-{}-{}-{}.log").format(today.year, today.month, today.day)
        if os.path.exists(logfile):
            os.remove(logfile) 
        
        tag = 'eire'
        period = ('2017-11-29', '2019-05-31')
        rmse = {'rmse':1.2345}
        runtime = '00:01:02'
        MODEL_VERSION = '0.1'
        MODEL_VERSION_NOTE = 'testing note'
        
        update_train_log(tag,period,rmse,runtime,MODEL_VERSION,MODEL_VERSION_NOTE,test=True)
        self.assertTrue(os.path.exists(logfile))
    
    
    def test_02_predict(self):
        """
        test predict logging
        """
        
        today = date.today()
        logfile = os.path.join(".","logs", 
                  "predict-test-{}-{}-{}.log").format(today.year, today.month, today.day)
        if os.path.exists(logfile):
            os.remove(logfile)        
        
        country = 'eire'
        y_pred = 1.23
        y_proba = 4.56
        target_date = '2019-05-31'
        runtime = '00:01:02'
        MODEL_VERSION = '0.1'
        
        update_predict_log(country, y_pred, y_proba, target_date, runtime, MODEL_VERSION, test=True)
        self.assertTrue(os.path.exists(logfile))
        
        
if __name__ == '__main__':
    unittest.main()
   
