import numpy as np
import uuid
import matplotlib.pyplot as plt
import cv2

from yolo_apnea_predicter.config import Image_config
from yolo_apnea_predicter.yolo_signal_detector import YoloSignalDetector
from yolo_apnea_predicter.predictions import Predictions

class ApneaDetector:

    def __init__(self):
        self.sliding_window_duration = Image_config.sliding_window_duration
        self.sliding_window_overlap = Image_config.sliding_window_overlap

        self.signal_index = 0
        self.signal_length = 0
        self.id = uuid.uuid1()
        self.signal = np.zeros(0) # TODO: Set to a higher value that can contain a complete nights recording, and copy into that array when appending instead

        self.predictions = Predictions()
        self.yolo = YoloSignalDetector()

    def append_signal(self,signal):
        """
        Appends newly recieved sensor data to the data already stored.
        Only used for predictions in real time

        :param signal: np array of new signal

        :return: Start and end location of apnea relative to start of recording
        """
        self.signal_length += len(signal)
        self.signal = np.concatenate((self.signal,signal))
        self._predict_unchecked_data()

    def get_predictions(self):
        """
        Returns a list of all events that has happened since the last time this method was called.
        """
        return self.predictions.get_all_predictions()

    def get_last_predictions(self):
        raise NotImplementedError("Will eventually return the last 90 seconds of predictions (or val from config")


    def _predict_unchecked_data(self):
        unchecked_duration = self.signal_length - self.signal_index

        signal_to_check = np.zeros(self.sliding_window_duration)

        if self.signal_index + self.signal_length < self.sliding_window_duration:
            signal_to_check[-unchecked_duration:] = self.signal
            self._predict_image(signal_to_check, unchecked_duration - self.sliding_window_duration)
            self.signal_index += min(unchecked_duration,self.sliding_window_overlap)
            unchecked_duration -= unchecked_duration

        elif unchecked_duration >= self.sliding_window_duration:
            signal_to_check[:] = self.signal[self.signal_index:self.signal_index + self.sliding_window_duration]
            self._predict_image(signal_to_check, self.signal_index)
            self.signal_index += self.sliding_window_overlap
            unchecked_duration -= self.sliding_window_overlap

        elif self.signal_length < self.sliding_window_duration:
            print("DOES THIS HAPPEN?")
            signal_to_check[-self.signal_length:] = self.signal[:]
            self.signal_index += min(unchecked_duration, self.sliding_window_overlap)

        elif unchecked_duration < self.sliding_window_duration: #This should be the remainder of the signal
            print("LAST ELIF?")
            print("Unchecked duration is:")
            print(unchecked_duration)
            print("signal index")
            print(self.signal_index)
            print("signal_length")
            print(self.signal_length)

            signal_to_check[:] = self.signal[-self.sliding_window_duration:]
            self._predict_image(signal_to_check, -self.sliding_window_duration)
            self.signal_index += unchecked_duration
            unchecked_duration -= unchecked_duration
            print("Now unchecked is:")
            print(unchecked_duration)
        else:
            raise NotImplementedError("Ran through all if-else statements")

        if unchecked_duration > 0:
            print("Unchecked duration is")
            print(unchecked_duration)
            print(f"{(self.signal_index/self.signal_length)*100:.2f}%")
            self._predict_unchecked_data()


    def _predict_image(self, signal, start_index):
        detections = self.yolo.detect(signal,show_bbox=False)
        self.predictions.append_predictions(detections,start_index)




