from unittest import TestCase
import yoloapnea.evaluate as evaluate
from yoloapnea.apneas import ApneaType

import numpy as np


class TestEvaluate(TestCase):
    
    def setUp(self):
        apnea_types = [ApneaType.ObstructiveApnea]
        pred = np.array([0,1,0,0])
        truth = np.array([0,1,0,0])
        self.evaluate = evaluate.Evaluate(pred,truth,apnea_types)
        
    def test_metrics(self):
        ...
        #self.evaluate.metrics

    def test_f1(self):
         value = self.evaluate.f1

    def test_predictedAHI(self):
        value = self.evaluate.predictionAHI
        print(value)

    def test_groundTruth_AHI(self):
        value = self.evaluate.groundTruthAHI
        print(value)

