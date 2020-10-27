import uuid
import os
from yolo_apnea_predicter.apnea_detector import ApneaDetector

class ApneaPredictor:

    def __init__(self):
        self.id = uuid.uuid1()
        self.predictor = ApneaDetector()

        # self.signal = self._set_signal(signal_file) if (signal_file is not None) else []

    def test(self):
        print("this is a test")
        return True

    def append_signal(self,signal):
        """
        Appends newly recieved sensor data to the data already stored.
        Only used for predictions in real time

        :param signal: np array of new signal

        :return: Start and end location of apnea relative to start of recording
        """
        print(signal)
        new_predictions = self.predictor.predict_signal(signal)
        return new_predictions
        ...

    def predict_signal(self):
        """
        Run the ML algorithm on the whole recording stored in self.signal
        :return: dataframe of all prediction results, unfiltered
        """
        ...
        return

    def generate_report_from_signal(self):
        """
        Generates a report from the whole signal
        :return: Report object
        """
        ...
        return


    def _set_signal(self):
        """
        Helper function to set signal file to self.signal
        """
        ...