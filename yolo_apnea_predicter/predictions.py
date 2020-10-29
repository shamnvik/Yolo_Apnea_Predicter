import configparser
import numpy as np

class Predictions:

    def __init__(self):
        self.predictions = np.zeros(1500) #Todo initalize with larger array and copy intoinstead
        config = configparser.ConfigParser()
        config.read("../yolo_apnea_predicter/config.ini")
        self.sliding_window_duration = int(config["DEFAULT"]["SlidingPredictionWindowDuration"])


    def insert_new_prediction(self,prediction):
        print("inserting new prediction")
        print(self.predictions.shape)
        np.maximum(self.predictions[prediction["start"]:prediction["end"]], prediction["confidence"], out=self.predictions[prediction["start"]:prediction["end"]])


    def get_unread_predictions(self):
        raise NotImplementedError("unread predictions not implemented yet")
        # TODO not implemented

    def get_all_predictions(self):
        print("All predictions:")
        return self.predictions

    def append_predictions(self, detections,start_index):
        for detection in detections:
            confidence = detection["confidence"]
            start_percentage = detection["left"]
            end_percentage = detection["right"]

            start = start_index + int(start_percentage * self.sliding_window_duration )
            end = start_index + int(end_percentage * self.sliding_window_duration )

            new_prediction = {"start":start,
                              "end":end,
                              "confidence":confidence}

            self.insert_new_prediction(new_prediction)


    def _sort_predictions(self):
        self.predictions.sort(key=lambda x : x["start"])





