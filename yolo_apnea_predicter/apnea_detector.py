import numpy as np
import configparser
import uuid
from yolo_apnea_predicter.yolo_signal_detector import YoloSignalDetector

class ApneaDetector:

    def __init__(self,signal=None):
        config = configparser.ConfigParser()
        config.read("../yolo_apnea_predicter/config.ini")
        self.sliding_window_duration = config["DEFAULT"]["SlidingPredictionWindowDuration"]
        self.sliding_window_overlap = config["DEFAULT"]["SlidingPredictionWindowOverlap"]

        self.signal_index = 0
        self.id = uuid.uuid1()
        self.signal = np.zeros(0) # TODO: Set to a higher value that can contain a complete nights recording, and copy into that array when appending instead

        #self.yolo = YoloSignalDetector()


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
            ...
            

    def append_signal(self,signal):
        """
        Appends newly recieved sensor data to the data already stored.
        Only used for predictions in real time

        :param signal: np array of new signal

        :return: Start and end location of apnea relative to start of recording
        """
        print("appending signal")
        self.signal = np.concatenate((self.signal,signal))

    def generate_report_from_signal(self):
        """
        Generates a report from the whole signal
        :return: Report object
        """
        ...
        return

    def predict_signal(self):
        """
        Run the ML algorithm on the whole recording stored in self.signal
        :return: dataframe of all prediction results, unfiltered
        """
        ...
        return