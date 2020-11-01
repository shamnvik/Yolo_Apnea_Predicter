import numpy as np
from yattag import Doc, indent
from .config import Image_config


class Predictions:

    def __init__(self):
        self.predictions = np.zeros(12 * 60 * 60 * 10)  # Todo initalize with larger array and copy intoinstead
        self.sliding_window_duration = Image_config.sliding_window_duration

    def get_unread_predictions(self):
        raise NotImplementedError("unread predictions not implemented yet")
        # TODO not implemented

    def get_predictions_as_np_array(self):
        """
        All stored predicions as numpy array
        :return: numpy array of predictions. Values represent confidence of prediction. 0 < value < 1
        """
        return self.predictions

    def _insert_new_prediction(self, prediction):
        """
        Inserts prediction into predictions array
        :param prediction: Prediction dictionary with keys: start, end & confidence
        """
        np.maximum(self.predictions[prediction["start"]:prediction["end"]], prediction["confidence"],
                   out=self.predictions[prediction["start"]:prediction["end"]])

    def append_predictions(self, detections, start_index):
        """
        Converts detection object from yolo into timestamps and copies the confidence of the prediction into the prediction array

        :param detections: Predictions coming from yolo with values in relative numbers based on the image it was detected on
        :param start_index: index in the signal of the leftmost pixel in the image yolo was run on.
        """
        for detection in detections:
            confidence = detection["confidence"]
            start_percentage = detection["left"]
            end_percentage = detection["right"]

            start = start_index + int(start_percentage * self.sliding_window_duration)
            end = start_index + int(end_percentage * self.sliding_window_duration)

            new_prediction = {"start": start,
                              "end": end,
                              "confidence": confidence}

            self._insert_new_prediction(new_prediction)

    def get_xml(self, threshold=0):
        """
        Return NSRR XML of all predictions
        :param threshold: Threshold of yolo's confidence. Discards confidence lower than threshold if thresholds is set.
        :return: string of XML
        """
        doc, tag, text = Doc().tagtext()
        doc.asis('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')

        with tag('PSGAnnotation'):
            with tag('SoftwareVersion'):
                text("Compumedics")
            with tag('EpochLength'):
                text("30")

            with tag("ScoredEvents"):
                start = 0
                end = 0
                for i, confidence in enumerate(self.predictions):

                    if start != 0:
                        if confidence == 0:
                            end = i
                            with tag("ScoredEvent"):
                                with tag("EventType"):
                                    text("Respiratory|Respiratory")
                                with tag("EventConcept"):
                                    text("Obstructive apnea|ObstructiveApnea")
                                with tag("Start"):
                                    text(start)
                                with tag("Duration"):
                                    text(end - start)
                                with tag("SignalLocation"):
                                    text("ABDO RES")
                            start = 0
                            end = 0

                    elif confidence > threshold:
                        start = i

        result = indent(
            doc.getvalue(),
            indentation=' ' * 4,
            newline='\r\n'
        )
        return result
