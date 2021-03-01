from unittest import TestCase

import numpy as np

import yoloapnea.yolo_signal_detector as Yolo
from yoloapnea.config import ImageConfig, YoloConfig
from yoloapnea.signalplotter import SignalPlotter
from pathlib import Path
import os

class TestYoloSignalDetector(TestCase):

    def setUp(self):
        test_signal = np.load("shhs1-200753-signal.npz")
        self.abdo_signal = test_signal["abdo_res"]

        self.prediction_duration = ImageConfig.sliding_window_duration
        self.prediction_overlap = ImageConfig.sliding_window_overlap

        self.weights_path = str(Path(os.getcwd(),"yolo-obj_last.weights"))
        self.config_path = str(Path(os.getcwd(),"yolo-obj.cfg"))
        self.yolo = Yolo.YoloSignalDetector(self.weights_path, YoloConfig.size,YoloConfig.iou,YoloConfig.score,self.config_path)

        self.signalplotter = SignalPlotter(self.prediction_overlap,self.prediction_overlap)

    def test_detect(self):
        img = self.signalplotter.signal_to_image(self.abdo_signal[89193:89193 + self.prediction_duration])
        predictions = self.yolo.detect(img)
        self.assertGreater(len(predictions), 0)  # NB! This is with current model. May not detect apnea on other models
