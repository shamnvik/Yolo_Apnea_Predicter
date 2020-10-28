import numpy as np
import configparser
import uuid
import matplotlib.pyplot as plt
import cv2

from yolo_apnea_predicter.yolo_signal_detector import YoloSignalDetector
from yolo_apnea_predicter.predictions import Predictions

class ApneaDetector:

    def __init__(self,signal=None):
        config = configparser.ConfigParser()
        config.read("../yolo_apnea_predicter/config.ini")
        self.sliding_window_duration = int(config["DEFAULT"]["SlidingPredictionWindowDuration"])
        self.sliding_window_overlap = int(config["DEFAULT"]["SlidingPredictionWindowOverlap"])

        self.signal_index = 0
        self.signal_length = 0
        self.id = uuid.uuid1()
        self.signal = np.zeros(0) # TODO: Set to a higher value that can contain a complete nights recording, and copy into that array when appending instead

        self.predictions = Predictions()
        self.yolo = YoloSignalDetector()


    def predict_signal_from_start(self):
        """
        Run the ML algorithm on the whole recording stored in self.signal
        :return: dataframe of all prediction results, unfiltered
        """
        ...
        raise NotImplementedError("not implemented yet")


    def get_new_predictions(self):
        """
        Returns a list of all events that has happened since the last time this method was called.
        """
        return self.predictions.get_all_predictions()
        # TODO not implemented yet

    def predict_unchecked_data(self):
        unchecked_duration = self.signal_length - self.signal_index

        signal_to_check = np.zeros(self.sliding_window_duration)

        if self.signal_index + self.signal_length < self.sliding_window_duration:
            signal_to_check[-unchecked_duration:] = self.signal
            self.predict_image(signal_to_check,unchecked_duration - self.sliding_window_duration)
            self.signal_index += min(unchecked_duration,self.sliding_window_overlap)

        elif unchecked_duration >= self.sliding_window_duration:
            signal_to_check[:] = self.signal[self.signal_index:self.signal_index + self.sliding_window_duration]
            self.predict_image(signal_to_check,self.signal_index)
            self.signal_index += self.sliding_window_overlap
            unchecked_duration -= self.sliding_window_overlap

        elif self.signal_length < self.sliding_window_duration:
            signal_to_check[-self.signal_length:] = self.signal[:]
            self.signal_index += min(unchecked_duration, self.sliding_window_overlap)

        elif unchecked_duration < self.sliding_window_duration: #This should be the remainder of the signal
            signal_to_check[:] = self.signal[-self.sliding_window_duration:]
            self.predict_image(signal_to_check,-self.sliding_window_duration)
            self.signal_index += unchecked_duration
            unchecked_duration -= unchecked_duration
        else:
            raise NotImplementedError("Ran through all if-else statements")

        if unchecked_duration > 0:
            self.predict_unchecked_data()

        print()
        #Remember to increase signal_index
        return

    def predict_image(self,signal,start_index):
        detections = self.yolo.detect(signal,show_bbox=False)
        print("Predicting image results")
        print("start index:", start_index)
        self.predictions.append_predictions(detections,start_index)

    def generate_image(self,signal):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(signal)
        ax.set_ylim(-1, 1)

        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.grid(False)
        plt.axis('off')

        ax.set_xlim(0, 900)

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image",img)
        cv2.waitKey(0)
        return img

    def append_signal(self,signal):
        """
        Appends newly recieved sensor data to the data already stored.
        Only used for predictions in real time

        :param signal: np array of new signal

        :return: Start and end location of apnea relative to start of recording
        """
        self.signal_length += len(signal)
        self.signal = np.concatenate((self.signal,signal))

    def generate_report_from_signal(self):
        """
        Generates a report from the whole signal
        :return: Report object
        """
        ...
        return

