import configparser
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

use_pickle_data = False

if use_pickle_data:
    import pickle
else:
    import tensorflow as tf
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

        if not use_pickle_data:
            self.saved_model_loaded = tf.saved_model.load(self.weights, tags=[tag_constants.SERVING])



    def detect(self,signal, show_bbox=False):
        test = self.signal_to_image(signal)
        if use_pickle_data:
            scores = pickle.load(open( "scores.p", "rb" ))[0].numpy() # Remove [0] when not using pickle
            boxes = pickle.load(open( "boxes.p", "rb" ))[0].numpy()
        else:
            scores,boxes = self.infer_image(test,show_bbox=show_bbox)

        predictions = []



        for confidence,prediction in zip(scores,boxes):
            if confidence > 0:
                (_, left_start,_, right_end) = prediction
                pred = {"confidence":confidence,
                        "left":left_start,
                        "right":right_end}

                predictions.append(pred)
        return predictions

    def signal_to_image(self,signal):
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
        return img


    def infer_image(self,image,show_bbox=False):
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        if False:
            print("scores")
            print(scores) # Array of confidence, shape=(1, 50)

            print("boxes") # Array of bounding boxes?  shape=(1, 50, 4)
            print(boxes) #I think 0: is center y position(or distance from top to pred), 1: percantege to min x pos(left side of pred) , 2:height(discard) 3: max x position(Right side of pred box)
            # Probably corners on the form distance from top left to (top/bottom, left, top/bottom, right)
            print("classes") #Classes, shape=(1, 50) (only 0 for apnea)
            print(classes)

            print("valid_detections")
            print(valid_detections) #tf.Tensor([2], shape=(1,), dtype=int32)
            print(type(valid_detections))

        if show_bbox and not use_pickle_data:
            image = utils.draw_bbox(original_image, pred_bbox)
            image = Image.fromarray(image.astype(np.uint8))
            image.show()

        return(scores.numpy()[0],boxes.numpy()[0])


