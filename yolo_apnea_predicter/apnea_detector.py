import numpy as np
import configparser

class ApneaDetector:

    def __init__(self,signal=None):
        self.signal = signal if signal is not None else []
        self.signal_index = 0
        self.signal_size = 0

        config = configparser.ConfigParser()
        config.read("../yolo_apnea_predicter/config.ini")
        self.sliding_window_duration = config["DEFAULT"]["SlidingPredictionWindowDuration"]
        self.sliding_window_overlap = config["DEFAULT"]["SlidingPredictionWindowOverlap"]

    def get_predictions(self):
        ...

    def predict_signal(self,signal):
        print("PREDICTING SIGNAL")
        self.signal_size += len(signal)
        self.signal = np.append(self.signal,signal)
        print(self.signal)
        print(self.signal_size)


    def predict_from_index(self,index):
        unchecked_duration = self.signal_size - self.signal_index
        if unchecked_duration < self.sliding_window_duration:
            



