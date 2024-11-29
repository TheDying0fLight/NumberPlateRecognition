from PIL import Image
from pathlib import Path
from typing import Dict, List, Union

import os
from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Boxes, Results
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

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


class NumberPlateRecognition():
    def __init__(self, file_path: str = "."):
        self.file_path = file_path
        model_path = os.path.join(os.path.abspath("."),"models","first10ktrain","weights","best.pt")
        self.model = YOLO(model_path, task='detect')
        yolo_model_path = os.path.join(os.path.abspath("."), "models", "yolo11n.pt")
        self.yolo_model = YOLO(yolo_model_path, task='detect')
        self.conf_interval = 0.5

    def analyze(self,
                image: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor]) -> DetectionResults:
        result: Results = self.model(image)[0]
        data: Boxes = result.boxes.cpu().numpy()
        return DetectionResults(data.xyxy,data.conf,data.cls,result.names)
    
    def analyze_car(self, image: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor]) -> DetectionResults:
        result = self.yolo_model(image)[0]
        data: Boxes = result.boxes.cpu().numpy()
        return DetectionResults(data.xyxy,data.conf,data.cls,result.names)

    def blur_image(self, image, bboxes):
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
    
    def detect_plate(self,image):
        # TODO: kommt das Bild schon so 
        # image = cv2.imread("example.jpg")
        # image = image[:, :, ::-1]
        
        car_class_id = 2 # maybe switch to searching algo to find car == 2 if the library changes the class id
        data = self.analyze_car(image)
        car_boxes_coordinates = data.boxes[data.cls == car_class_id]
        car_boxes_conf = data.conf[data.cls == car_class_id]
        
        number_of_cars = len(car_boxes_coordinates)
        store_box = np.zeros((number_of_cars, 4))
        store_conf = np.zeros((number_of_cars,))
        current_index = 0

        for i in range(number_of_cars):
            x1, y1, x2, y2 = map(int, car_boxes_coordinates[i])
            cropped_image = image[y1:y2, x1:x2]

            # looking for plates
            plate_rec = self.analyze(cropped_image)

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
        return store_box, store_conf
