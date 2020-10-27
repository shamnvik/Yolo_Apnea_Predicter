import tensorflow as tf
import configparser
import os
import cv2
import numpy as np
from PIL import Image

from tensorflow.python.saved_model import tag_constants
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import yolo_apnea_predicter.tensorflow_yolov4.core.utils as utils


class YoloSignalDetector:

    def __init__(self):

        config = configparser.ConfigParser()
        config.read("../yolo_apnea_predicter/config.ini")
        self.input_size = int(config["YOLO"]["size"])
        self.iou = float(config["YOLO"]["iou"])
        self.score = float(config["YOLO"]["score"])

        self.weights = f'{os.getcwd()}{os.sep}..{os.sep}{config["YOLO"]["weights"]}'
        print(self.weights)

        print("Loading model")
        self.saved_model_loaded = tf.saved_model.load(self.weights, tags=[tag_constants.SERVING])



    def detect(self,signal):
        print("detecting")
        demo_image = f"{os.getcwd()}{os.sep}296807.png"
        self.infer_image(demo_image)


    def infer_image(self,image_path):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (self.input_size, self.input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = self.saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        image = utils.draw_bbox(original_image, pred_bbox)
        image = Image.fromarray(image.astype(np.uint8))
        image.show()