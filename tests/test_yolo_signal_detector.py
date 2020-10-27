from unittest import TestCase
import yolo_apnea_predicter.yolo_signal_detector as Yolo
import numpy as np
import configparser


class TestYoloSignalDetector(TestCase):

    def setUp(self):
        test_signal = np.load("shhs1-200703-signal.npz")
        self.abdo_signal = test_signal["abdo_res"]

        config = configparser.ConfigParser()
        config.read("../yolo_apnea_predicter/config.ini")
        self.sliding_window = int(config["DEFAULT"]["SlidingPredictionWindowOverlap"])

        self.yolo = Yolo.YoloSignalDetector()

    def test_detect(self):
        self.yolo.detect(self.abdo_signal[0:self.sliding_window])
        self.fail()
