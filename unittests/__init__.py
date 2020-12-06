import os
import sys
import unittest

sys.path.append(os.path.realpath(os.path.dirname(__file__)))

# model tests
from ModelTests import *
ModelTestSuite = unittest.TestLoader().loadTestsFromTestCase(ModelTest)

# logger tests
from LoggerTests import *
LoggerTestSuite = unittest.TestLoader().loadTestsFromTestCase(LoggerTest)

MainSuite = unittest.TestSuite([ModelTestSuite, LoggerTestSuite])

