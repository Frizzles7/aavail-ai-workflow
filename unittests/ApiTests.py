#!/usr/bin/env python
"""
api unit tests
"""

import unittest
import os, sys
import json
import requests

# confirm flask server is running for tests
try:
    requests.post('http://0.0.0.0:8080/')
    server_is_running = True
except:
    server_is_running = False


class ApiTest(unittest.TestCase):

    @unittest.skipUnless(server_is_running, "local flask server not running")
    def test_01_train(self):
        """
        test train api
        """
        
        request_json = {'test':'test'}
        r = requests.post('http://0.0.0.0:8080/train', json=request_json)
        response = json.loads(r.text)
        self.assertTrue(response == True)
        
    
    @unittest.skipUnless(server_is_running, "local flask server not running")
    def test_02_predict(self):
        """
        test predict api
        """
        
        request_json = {'country':'all','year':'2018','month':'01','day':'05', 'test':'test'}
        r = requests.post('http://0.0.0.0:8080/predict', json=request_json)
        response = json.loads(r.text)
        self.assertTrue(float(response[0]) >= 0)
        

    @unittest.skipUnless(server_is_running, "local flask server not running")        
    def test_03_predict_bads(self):
        """
        test predict api when no data or bad data is given
        """
        
        # no data provided
        r = requests.post('http://0.0.0.0:8080/predict')
        response = json.loads(r.text)
        self.assertTrue(response == [])
        
        # bad data given
        r = requests.post('http://0.0.0.0:8080/predict', json={'bad':'data'})
        response = json.loads(r.text)
        self.assertTrue(response == [])


    @unittest.skipUnless(server_is_running, "local flask server not running")        
    def test_04_logfile(self):
        """
        test logfile api
        """
        
        log = 'predict-test-2020-12-6.log'
        request_json = {'log':log}
        r = requests.post('http://0.0.0.0:8080/logfile', json=request_json)
        
        with open(log, 'wb') as f:
            f.write(r.content)
        self.assertTrue(os.path.exists(log))
        if os.path.exists(log):
            os.remove(log)
        
        
if __name__ == '__main__':
    unittest.main()

