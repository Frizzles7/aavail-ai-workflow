#!/usr/bin/env python
"""
model unit tests
"""

import unittest
import os, sys
from src.model import model_train, model_load, model_predict

class ModelTest(unittest.TestCase):

    def test_01_train(self):
        """
        test model training
        """
        
        data_dir = os.path.join(".","data","cs-train")
        model_train(data_dir, test=True)
        saved_model = os.path.join("models","test-united_kingdom-0_1.joblib")
        self.assertTrue(os.path.exists(saved_model))

       
    def test_02_load(self):
        """
        test model loading
        """
        
        all_data, all_models = model_load()
        model = all_models["eire"]
        # check that both fit and predict methods are available for model
        self.assertTrue('fit' in dir(model))
        self.assertTrue('predict' in dir(model))

    
    def test_03_predict(self):
        """
        test model predicting
        """
        
        pred = model_predict('all','2018','01','05')
        y_pred = pred['y_pred']
        self.assertTrue(y_pred >= 0)


if __name__ == '__main__':
    unittest.main()
    
