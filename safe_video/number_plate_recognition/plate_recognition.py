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

    def blur_image(self, image: Img, boxes: np.ndarray[Boxes]):
        image = image.copy()
        for box in boxes:
            cropped_img = self.crop_image(image, box)
            blurred_img = cv2.GaussianBlur(cropped_img, (25,25),0)
            x1,y1,x2,y2 = box.astype("int")
            image[y1:y2,x1:x2] = blurred_img
            
            # TODO Delete later
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 4)
        return image

    def crop_image(self, image: Img, xyxy: np.ndarray):
        assert len(xyxy) == 4, "Array must have exactly 4 entries"
        x1,y1,x2,y2 = xyxy.astype("int")
        return image[y1:y2,x1:x2]

    def chained_detection(self, image: Img, cls: str) -> DetectionResults:
        data = self.analyze(image, self.yolo_model)
        car_class_id = data.get_key(cls)
        car_boxes_coordinates = data.boxes[data.cls == car_class_id]
        car_boxes_conf = data.conf[data.cls == car_class_id]

        store_box = []
        store_conf = []

        for box,conf in zip(car_boxes_coordinates,car_boxes_conf):
            if conf < self.conf_interval: continue
            x1,y1,_,_ =  box.astype("int")
            cropped_image = self.crop_image(image, box)

            # looking for plates
            plate_rec = self.analyze(cropped_image, self.plate_model)
            num_plates = len(plate_rec.boxes)

            if num_plates > 0:
                # transform back to original coordinates
                transformed_boxes =  np.array(plate_rec.boxes) + [x1, y1, x1, y1]
                transformed_conf = np.array(plate_rec.conf)

                transformed_boxes = transformed_boxes[transformed_conf > self.conf_interval]
                transformed_conf = transformed_conf[transformed_conf > self.conf_interval]
                store_box.extend(transformed_boxes)
                store_conf.extend(transformed_conf)

        # output store_box contains the coordinates of the plates which needs to be blurred
        return DetectionResults(np.array(store_box), np.array(store_conf), None, None)
