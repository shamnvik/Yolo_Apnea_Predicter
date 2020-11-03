import uuid

import numpy as np
import progressbar

from yoloapnea.config import ImageConfig,YoloConfig
from yoloapnea.predictions import Predictions
from yoloapnea.yolo_signal_detector import YoloSignalDetector


class ApneaDetector:

    def __init__(self,weights_path=YoloConfig.weights): #Todo, remove default weights
        self.sliding_window_duration = ImageConfig.sliding_window_duration
        self.sliding_window_overlap = ImageConfig.sliding_window_overlap

        self.signal_index = 0
        self.signal_length = 0
        self.id = uuid.uuid1() #TODO id from parameters maybe? Could be used when making the detector concurrent
        self.signal = np.zeros(12 * 60 * 60 * 10)

        self.predictions = Predictions()
        self.yolo = YoloSignalDetector(weights_path)

    def append_signal(self, signal):
        """
        Appends newly received sensor data to the data already stored. Runs yolo on signal to detect apnea.

        :param signal: np array of new signal

        :return: None. Predictions can be accessed from predictions object (self.predictions).
        """

        self._signal[self.signal_length:self.signal_length + len(signal)] = signal
        self.signal_length += len(signal)
        print("Predicting newly added signal")
        progress = progressbar.ProgressBar(max_value=self.signal_length)
        progress.update(self.signal_index)
        self._predict_unchecked_data(progress)

    @property
    def signal(self):
        return self._signal[:self.signal_length]

    @signal.setter
    def signal(self, signal):
        self._signal = signal

    def _predict_unchecked_data(self, progress):
        """
        Iterates through the data that has not been analyzed by yolo yet.
        Appends np array of 0's if there is to little data, otherwise recursively predicts apneas
        on the remaining data with a stride of {self.sliding_window_overlap}.

        If newly added data is less than {self.sliding_window_overlap} it predicts all the new data
        and whatever is needed before to reach {self.sliding_window_duration}
        """
        unchecked_duration = self.signal_length - self.signal_index
        signal_to_check = np.zeros(self.sliding_window_duration)

        if self.signal_index + self.signal_length < self.sliding_window_duration:
            signal_to_check[-self.signal_length:] = self.signal
            self._predict_image(signal_to_check, unchecked_duration - self.sliding_window_duration)
            self.signal_index += min(unchecked_duration, self.sliding_window_overlap)
            unchecked_duration -= unchecked_duration

        elif unchecked_duration >= self.sliding_window_duration:
            signal_to_check[:] = self.signal[self.signal_index:self.signal_index + self.sliding_window_duration]
            self._predict_image(signal_to_check, self.signal_index)
            self.signal_index += self.sliding_window_overlap
            unchecked_duration -= self.sliding_window_overlap

        elif self.signal_length < self.sliding_window_duration:
            signal_to_check[-self.signal_length:] = self.signal[:]
            self.signal_index += min(unchecked_duration, self.sliding_window_overlap)

        elif unchecked_duration < self.sliding_window_duration:  # This should be the remainder of the signal
            signal_to_check[:] = self.signal[-self.sliding_window_duration:]
            self._predict_image(signal_to_check, -self.sliding_window_duration)
            self.signal_index += unchecked_duration
            unchecked_duration -= unchecked_duration
        else:
            raise NotImplementedError("Ran through all if-else statements")

        # print(f"Analyzed: {(self.signal_index / self.signal_length) * 100:.2f}%")
        print("Signal index is")
        progress.update(self.signal_index)
        if unchecked_duration > 0:
            self._predict_unchecked_data(progress)

    def _predict_image(self, signal, start_index):
        """
        Local helper function for running yolo on a signal of already correct length and inserts the predictions
        into the prediction object

        :param signal: signal to detect: Should always be {self.sliding_window_duration} length
        :param start_index: index of signal[0] in {self.signal} for knowing when predictions start
                since start of recording
        """
        detections = self.yolo.detect(signal, show_bbox=False)
        self.predictions.append_predictions(detections, start_index)