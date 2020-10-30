import os

class Image_config():
    sliding_window_duration = 900
    sliding_window_overlap = 450

class YOLO_config():
    size = 416
    weights = "weights/yolov3-416"
    weights = os.path.abspath(os.path.join(os.path.dirname( __file__ ),weights))
    iou = 0.45
    score = 0.25
    pred_names = "tensorflow_yolov4/data/classes/coco.names"
    pred_names = os.path.abspath(os.path.join(os.path.dirname( __file__ ),pred_names))

