from unittest import TestCase
from yolo_apnea_predicter.apnea_detector import ApneaDetector
import os
import numpy as np


class TestApneaPredictor(TestCase):
    def setUp(self):
        test_signal = np.load("shhs1-200753-signal.npz")
        self.abdo_signal = test_signal["abdo_res"]
        self.thor_signal = test_signal["thor_res"]
        self.apnea_predictor = ApneaDetector()

    def test_append_signal(self):
        self.apnea_predictor.append_signal(self.abdo_signal[0:900])
        self.apnea_predictor.append_signal(self.abdo_signal[900:1500])
        true_signal = self.abdo_signal[0:1500]
        appended_signal = self.apnea_predictor.signal
        np.testing.assert_almost_equal(true_signal,appended_signal,decimal=5)


