from PIL import Image
from pathlib import Path

import os
from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Boxes, Results
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any
Img = str|Path|int|Image.Image|list|tuple|np.ndarray|torch.Tensor

@dataclass
class DetectionResults:
    boxes:  np.ndarray[Any, Any]
    conf:   np.ndarray[Any, Any]
    cls:    np.ndarray[Any, Any]
    clslgd: dict[int, str]

    def __str__(self):
        ret = ""
        for a,b in self.__dict__.items(): ret = f"{ret}{a}: {b}\n"
        return ret

    def get_key(self, val: str):
        return list(self.clslgd.keys())[list(self.clslgd.values()).index(val)]

    def tuple(self): return self.__dict__.values()


class NumberPlateRecognition():
    def __init__(self, file_path: str = "."):
        self.file_path = file_path
        plate_model_path = os.path.join(os.path.abspath("."),"models","first10ktrain","weights","best.pt")
        self.plate_model = YOLO(plate_model_path, task='detect')
        yolo_model_path = os.path.join(os.path.abspath("."), "models", "yolo11n.pt")
        self.yolo_model = YOLO(yolo_model_path, task='detect')
        self.conf_interval = 0.5

    def analyze(self, image: Img, model: YOLO) -> DetectionResults:
        result: Results = model(image)[0]
        data = result.boxes.cpu().numpy()
        return DetectionResults(data.xyxy,data.conf,data.cls,result.names)

    def blur_image(self, image: Img, bboxes: Boxes):
        int_bboxes = bboxes.astype('int')
        image = image.copy()
        for box in int_bboxes:
            cropped_img = image[box[1]:box[3], box[0]:box[2]]
            blurred_img = cv2.GaussianBlur(cropped_img, (25,25),0)
            x_offset, y_offset = box[0], box[1]
            x_end = x_offset + blurred_img.shape[1]
            y_end = y_offset + blurred_img.shape[0]
            image[y_offset:y_end, x_offset:x_end] = blurred_img
        return image

    def chained_detection(self, image: Img, cls: str, first_conf_thresh: float = 0) -> DetectionResults:
        data = self.analyze(image, self.yolo_model)
        car_class_id = data.get_key(cls)
        car_boxes_coordinates = data.boxes[data.cls == car_class_id]
        car_boxes_conf = data.conf[data.cls == car_class_id]

        number_of_cars = len(car_boxes_coordinates)
        store_box = np.zeros((number_of_cars, 4))
        store_conf = np.zeros((number_of_cars,))
        current_index = 0

        for i in range(number_of_cars):
            if car_boxes_conf[i] < first_conf_thresh: continue
            x1, y1, x2, y2 = map(int, car_boxes_coordinates[i])
            cropped_image = image[y1:y2, x1:x2]

            # looking for plates
            plate_rec = self.analyze(cropped_image, self.plate_model)

            num_plates = len(plate_rec.boxes)
            if num_plates > 0:
                # transform back to original coordinates
                transformed_boxes =  np.array(plate_rec.boxes) + [x1, y1, x1, y1]
                transformed_conf = np.array(plate_rec.conf)

                store_box[current_index:current_index + num_plates] = transformed_boxes
                store_conf[current_index:current_index + num_plates] = transformed_conf
                current_index += num_plates

        store_box = store_box[:current_index]
        store_conf = store_conf[:current_index]

        store_box = store_box[store_conf > self.conf_interval]
        store_conf = store_conf[store_conf > self.conf_interval]

        # output store_box contains the coordinates of the plates which needs to be blurred
        return DetectionResults(store_box, store_conf, None, None)
