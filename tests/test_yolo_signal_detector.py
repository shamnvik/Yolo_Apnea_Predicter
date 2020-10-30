from unittest import TestCase
import yolo_apnea_predicter.yolo_signal_detector as Yolo
import numpy as np

from yolo_apnea_predicter.config import Image_config


class TestYoloSignalDetector(TestCase):

    def setUp(self):
        test_signal = np.load("shhs1-200753-signal.npz")
        self.abdo_signal = test_signal["abdo_res"]

        self.prediction_duration = Image_config.sliding_window_duration

        self.yolo = Yolo.YoloSignalDetector()

    def test_detect(self):
        print("test")
        self.yolo.detect(self.abdo_signal[89193:89193+self.prediction_duration])
        self.fail()
