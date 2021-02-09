import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from pandas.core.common import flatten
from .config import YoloConfig
from .tensorflow_yolov4.core import utils

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


CONF_THRESH, NMS_THRESH = 00, 0.5


class YoloSignalDetector:

    loaded_model = None

    def __init__(self,weights_path,input_size,iou,score,config_path=None):
        self.weights = weights_path
        self.input_size = input_size
        self.iou = iou
        self.score = score

        if YoloSignalDetector.loaded_model is None:
            YoloSignalDetector.loaded_model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    def detect(self, signal, show_bbox=False):
        print("running detect")
        loaded_model = YoloSignalDetector.loaded_model
        layers = loaded_model.getLayerNames()
        output_layers = [layers[i[0] - 1] for i in loaded_model.getUnconnectedOutLayers()]

        image = self.signal_to_image(signal)
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)

        loaded_model.setInput(blob)
        layer_outputs = loaded_model.forward(output_layers)
        class_ids, confidences, b_boxes = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
        class_ids, confidences, b_boxes
        print(class_id)
        print(confidences)
        print(b_boxes)
        for confidence,(x_pos, _, width, _) in zip(confidences,b_boxes):
            print(f"Prediction conf:{confidence} x_pos:{x_pos} width:{width}")


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

        boxes, pred_conf = None, None

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

        if show_bbox:
            image = utils.draw_bbox(original_image, pred_bbox)
            image = Image.fromarray(image.astype(np.uint8))
            image.show()

        return scores.numpy()[0], boxes.numpy()[0]
