from unittest import TestCase
from yolo_apnea_predicter.apnea_detector import ApneaDetector
from yolo_apnea_predicter.predictions import Predictions
import os
import numpy as np

class TestApneaDetector(TestCase):

    def setUp(self):
        test_signal = np.load("shhs1-200753-signal.npz")
        self.abdo_signal = test_signal["abdo_res"]
        self.thor_signal = test_signal["thor_res"]

    #def test_predict_unchecked_data_enough_data(self):
    #    apnea_predictor = ApneaDetector()
    #    apnea_predictor.append_signal(self.abdo_signal[0:1700])
    #    apnea_predictor._predict_unchecked_data()
    #    predictions = apnea_predictor.predictions.get_all_predictions()
    #    print(predictions)
    #    self.fail()

    def test_predict_unchecked_data_too_little_data(self):
        apnea_predictor = ApneaDetector()
        apnea_predictor.append_signal(self.abdo_signal[0:700])
        predictions = apnea_predictor.predictions.get_all_predictions()
        self.assertEqual(np.max(predictions[700:]), 0)


    def test_append_signal(self):
        apnea_predictor = ApneaDetector()
        apnea_predictor.append_signal(self.abdo_signal[0:900])
        apnea_predictor.append_signal(self.abdo_signal[900:1500])
        true_signal = self.abdo_signal[0:1500]
        appended_signal = apnea_predictor.signal
        np.testing.assert_almost_equal(true_signal,appended_signal,decimal=5)
        self.assertEqual(1500,apnea_predictor.signal_length)
