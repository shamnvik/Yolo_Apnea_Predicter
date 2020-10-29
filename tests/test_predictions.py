from unittest import TestCase
from yolo_apnea_predicter.predictions import Predictions


class TestPredictions(TestCase):

    def setUp(self):
        self.predictions = Predictions()

    def test_insert_new_prediction(self):
        first_prediction = {"start": 30,
                            "end": 400,
                            "confidence": 70}
        self.predictions.insert_new_prediction(first_prediction)
        self.assertIn(first_prediction, self.predictions.predictions)

        second_prediction = {"start": 450,
                             "end": 700,
                             "confidence": 70}
        self.predictions.insert_new_prediction(second_prediction)
        self.assertIn(second_prediction, self.predictions.predictions)

    def test_get_unread_predictions(self):
        self.fail()

    def test_get_all_predictions(self):
        self.fail()

    def test_append_predictions(self):
        predictions = [{"left": 0.2,
                        "right": 0.4,
                        "confidence": 70},
                       {"left": 0.5,
                        "right": 0.7,
                        "confidence": 70}]
        self.predictions.append_predictions(predictions,0)
        self.assertEqual(len(self.predictions.predictions),2)

    def test_clean_predictions_no_overlap(self):
        predictions = [{"left": 0.2,
                        "right": 0.4,
                        "confidence": 70},
                       {"left": 0.5,
                        "right": 0.7,
                        "confidence": 70}]

        self.predictions.append_predictions(predictions,0)
        self.predictions.clean_predictions()
        self.assertEqual(len(self.predictions.predictions), 2)

    def test_clean_predictions_no_overlap_test_sorting(self):
        predictions = [{"left": 0.5,
                        "right": 0.7,
                        "confidence": 50},
                       {"left": 0.2,
                        "right": 0.4,
                        "confidence": 70}]

        self.predictions.append_predictions(predictions,0)
        self.predictions.clean_predictions()
        self.assertEqual(self.predictions.predictions[0]["confidence"], 70)
        self.assertEqual(self.predictions.predictions[1]["confidence"], 50)

    def test_clean_predictions_with_overlap(self):
        predictions = [{"left": 0.2,
                        "right": 0.6,
                        "confidence": 70},
                       {"left": 0.5,
                        "right": 0.7,
                        "confidence": 70}]

        self.predictions.append_predictions(predictions,0)
        self.predictions.clean_predictions()
        self.assertEqual(len(self.predictions.predictions), 1)