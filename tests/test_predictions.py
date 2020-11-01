from unittest import TestCase
from xml.dom import minidom

from yolo_apnea_predicter.config import Image_config
from yolo_apnea_predicter.predictions import Predictions


class TestPredictions(TestCase):

    def setUp(self):
        self.predictions = Predictions()
        self.sliding_window_duration = Image_config.sliding_window_duration

        self.non_overlap_predictions = [{"left": 0.2,
                                         "right": 0.4,
                                         "confidence": 70},
                                        {"left": 0.5,
                                         "right": 0.7,
                                         "confidence": 65}]

        self.overlap_predictions = [{"left": 0.2,
                                     "right": 0.6,
                                     "confidence": 70},
                                    {"left": 0.5,
                                     "right": 0.7,
                                     "confidence": 65}]

    def test_insert_new_prediction(self):
        first_prediction = {"start": 30,
                            "end": 400,
                            "confidence": 70}
        self.predictions._insert_new_prediction(first_prediction)

        second_prediction = {"start": 450,
                             "end": 700,
                             "confidence": 65}
        self.predictions._insert_new_prediction(second_prediction)
        pred_array = self.predictions.predictions

        self.assertEqual(pred_array[29], 0)
        self.assertEqual(pred_array[30], 70)
        self.assertEqual(pred_array[200], 70)
        self.assertEqual(pred_array[399], 70)
        self.assertEqual(pred_array[400], 0)
        self.assertEqual(pred_array[401], 0)

        self.assertEqual(pred_array[420], 0)
        self.assertEqual(pred_array[449], 0)
        self.assertEqual(pred_array[450], 65)
        self.assertEqual(pred_array[451], 65)
        self.assertEqual(pred_array[563], 65)
        self.assertEqual(pred_array[699], 65)
        self.assertEqual(pred_array[700], 0)
        self.assertEqual(pred_array[800], 0)

    def test_get_all_predictions(self):
        self.fail()

    def test_append_predictions(self):
        self.predictions.append_predictions(self.non_overlap_predictions, 0)
        pred_array = self.predictions.predictions

        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.2)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.199)], 0)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.23)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.39)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.40)], 0)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.41)], 0)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.47)], 0)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.5)], 65)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.499)], 0)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.51)], 65)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.699)], 65)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.7)], 0)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.85)], 0)

    def test_append_predictions_with_overlap(self):
        self.predictions.append_predictions(self.overlap_predictions, 0)
        pred_array = self.predictions.predictions

        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.2)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.199)], 0)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.23)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.39)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.40)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.41)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.47)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.59)], 70)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.6)], 65)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.61)], 65)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.699)], 65)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.7)], 0)
        self.assertEqual(pred_array[int(self.sliding_window_duration * 0.85)], 0)

    def test_get_xml(self):
        self.predictions.append_predictions(self.non_overlap_predictions, 0)

        xml = self.predictions.get_xml(0)
        minidom.parseString(xml)
