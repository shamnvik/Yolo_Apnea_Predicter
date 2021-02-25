import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from pandas.core.common import flatten
from .config import YoloConfig
from .tensorflow_yolov4.core import utils
import os

CONF_THRESH, NMS_THRESH = 0.0, 0.5


class YoloSignalDetector:

    loaded_model = None

    def __init__(self, weights_path : str, input_size, iou, score, config_path : str):
        self.weights = weights_path
        self.input_size = input_size
        self.iou = iou
        self.score = score

        if YoloSignalDetector.loaded_model is None:

            print("loading model")

            print(f"Config path:{config_path} weights path: {weights_path}")
            print(os.path.exists(weights_path))
            print(os.path.exists(config_path))
            print(type(weights_path))
            YoloSignalDetector.loaded_model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            print("model loaded")

    def detect(self, signal, show_bbox=False):
        print("running detect")
        loaded_model = YoloSignalDetector.loaded_model
        layers = loaded_model.getLayerNames()
        output_layers = [layers[i[0] - 1] for i in loaded_model.getUnconnectedOutLayers()]

        image = self.signal_to_image(signal)


        # from https://cloudxlab.com/blog/object-detection-yolo-and-python-pydarknet/
        #image = cv2.imread(r"C:\Users\Sondre Hamnvik\Downloads\dog.jpg")
        (H, W) = image.shape[:2]

        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)

        loaded_model.setInput(blob)
        layer_outputs = loaded_model.forward(output_layers)
        boxes = []
        confidences = []
        classIDs = []

        predictions = []


        for output in layer_outputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONF_THRESH:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH,
                                NMS_THRESH)


        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                confidence = confidences[i]
                left_start = x/W
                right_end = (x+w)/W

                left_start = 0 if left_start < 0 else left_start
                right_end = 1 if right_end > 1 else right_end

                pred = {"confidence": confidence,
                                    "left": left_start,
                                    "right": right_end}
                predictions.append(pred)
                print(pred)
        return predictions

        # image = self.signal_to_image(signal)
        # scores, boxes = self.infer_image(image, show_bbox=show_bbox)
        #
        # predictions = []
        # for confidence, prediction in zip(scores, boxes):
        #     if confidence > 0:
        #         (_, left_start, _, right_end) = prediction
        #         pred = {"confidence": confidence,
        #                 "left": left_start,
        #                 "right": right_end}
        #
        #         predictions.append(pred)
        # return predictions

    #Maybe write this to it's own class? SignalPlotter
    @staticmethod
    def signal_to_image(signal):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(signal)
        ax.set_ylim(-1, 1)

        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.grid(False)
        plt.axis('off')
        ax.set_xlim(0, 900)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        return img

    def infer_image(self, image, show_bbox=False):
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (self.input_size, self.input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = YoloSignalDetector.loaded_model.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)

