from unittest import TestCase

import numpy as np

from yolo_apnea_predicter.apnea_detector import ApneaDetector


class TestApneaDetector(TestCase):

    def setUp(self):
        test_signal = np.load("shhs1-200753-signal.npz")
        self.abdo_signal = test_signal["abdo_res"]
        self.thor_signal = test_signal["thor_res"]

    def test_predict_unchecked_data_too_little_data(self):
        apnea_predictor = ApneaDetector()
        apnea_predictor.append_signal(self.abdo_signal[0:700])
        predictions = apnea_predictor.predictions.get_predictions_as_np_array()
        self.assertEqual(np.max(predictions[700:]), 0)

    def test_append_signal(self):
        apnea_predictor = ApneaDetector()
        apnea_predictor.append_signal(self.abdo_signal[0:900])
        apnea_predictor.append_signal(self.abdo_signal[900:1500])
        true_signal = self.abdo_signal[0:1500]
        appended_signal = apnea_predictor.signal
        np.testing.assert_almost_equal(true_signal, appended_signal, decimal=5)
        self.assertEqual(1500, apnea_predictor.signal_length)


    def test_append_signal_long(self):
        apnea_predictor = ApneaDetector()
        apnea_predictor.append_signal(self.abdo_signal[0:900])
        apnea_predictor.append_signal(self.abdo_signal[900:1500])
        apnea_predictor.append_signal(self.abdo_signal[1500:10000])
        true_signal = self.abdo_signal[0:10000]
        appended_signal = apnea_predictor.signal
        np.testing.assert_almost_equal(true_signal, appended_signal, decimal=5)
        self.assertEqual(10000, apnea_predictor.signal_length)