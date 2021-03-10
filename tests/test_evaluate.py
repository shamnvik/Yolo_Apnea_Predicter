from unittest import TestCase
import yoloapnea.evaluate as evaluate
from yoloapnea.apneas import ApneaType

import numpy as np
from sklearn.metrics import f1_score

class TestEvaluate(TestCase):
    
    def setUp(self):
        apnea_types = [ApneaType.ObstructiveApnea]
        self.pred  = np.array([0,1,0,0,0,1,1,0])
        self.truth = np.array([0,1,0,0,0,1,0,0])
        self.evaluate = evaluate.Evaluate(self.pred,self.truth,apnea_types)

    def test_f1(self):
        value = self.evaluate.f1
        self.assertEqual(value,f1_score(y_true=self.truth,y_pred=self.pred))

    def test_predictedAHI(self):
        value = self.evaluate.predictionAHI
        self.assertGreaterEqual(value,0)

    def test_groundTruth_AHI(self):
        value = self.evaluate.groundTruthAHI
        self.assertGreaterEqual(value,0)

