from unittest import TestCase
from yolo_apnea_predicter import ApneaPredictor, Predictions
import os
import numpy as np



class TestApneaPredictor(TestCase):
    def setUp(self):
        test_signal = np.load("shhs1-200703-signal.npz")
        self.abdo_signal = test_signal["abdo_res"]
        self.thor_signal = test_signal["thor_res"]
        self.apnea_predictor = ApneaPredictor()


    def test_append_signal(self):
        for i in range(0,1000,60):
            returned_prediction = self.apnea_predictor.append_signal(self.abdo_signal[i:i+30])
            self.assertIsInstance(returned_prediction,Predictions)
