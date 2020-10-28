from unittest import TestCase
import yolo_apnea_predicter.yolo_signal_detector as Yolo
import numpy as np
import configparser


class TestYoloSignalDetector(TestCase):

    def setUp(self):
        test_signal = np.load("shhs1-200753-signal.npz")
        self.abdo_signal = test_signal["abdo_res"]

        config = configparser.ConfigParser()
        config.read("../yolo_apnea_predicter/config.ini")
        self.prediction_duration = int(config["DEFAULT"]["SlidingPredictionWindowDuration"])

        self.yolo = Yolo.YoloSignalDetector()

    def test_detect(self):
        print("test")
        self.yolo.detect(self.abdo_signal[89193:89193+self.prediction_duration])
        self.fail()
