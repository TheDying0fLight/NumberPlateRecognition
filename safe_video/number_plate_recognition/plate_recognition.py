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

    def tuple(self): return self.__dict__.values()

def get_key(dct: dict, val: str):
    return list(dct.keys())[list(dct.values()).index(val)]

class ObjectDetection():
    def __init__(self, file_path: str = "."):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.file_path = file_path
        self.models: list[YOLO] = []
        self.add_model(os.path.join(os.path.abspath("."),"models","first10ktrain","weights","best.pt"))
        self.add_model(os.path.join(os.path.abspath("."), "models", "yolo11n.pt"))
        self.conf_interval = 0.5

    def add_model(self, path: str):
        model = YOLO(path, task="detect")
        model.to(self.device)
        self.models.append(model)

    def analyze(self, image: Img, classes: list[str], verbose = False) -> DetectionResults:
        mdl_clss: dict[YOLO, list] = {}
        for mdl_idx in range(len(self.models)):
            clss = self.models[mdl_idx].names
            mdl_clss[mdl_idx] = []
            for cls in classes.copy():
                if len(classes) == 0: break
                try:
                    cls_idx = get_key(clss, cls)
                    classes.remove(cls)
                    mdl_clss[mdl_idx].append(cls_idx)
                except Exception as e:
                    if verbose: print(e)
            else: continue
            break
        print(mdl_clss)
        for mdl,clss in mdl_clss.items():
            if len(clss) == 0: continue
            self.result: Results = self.models[mdl](image, classes = clss)[0]
        # result: Results = model(image)[0]
        data = self.result.boxes.cpu().numpy()
        return DetectionResults(data.xyxy,data.conf,data.cls,self.result.names)

    def get_classes(self): return [m.names for m in self.models]

    def blur_image(self, image: Img, boxes: np.ndarray[Boxes]):
        image = image.copy()
        for box in boxes:
            cropped_img = self.crop_image(image, box)
            blurred_img = cv2.GaussianBlur(cropped_img, (25,25),0)
            x1,y1,x2,y2 = box.astype("int")
            image[y1:y2,x1:x2] = blurred_img
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
