from PIL import Image
from pathlib import Path
from typing import Dict, List, Union

import os
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import cv2
import numpy as np
import torch

class DetectionResults():
    def __init__(self, boxes, conf, cls, classlegend):
        self.boxes = boxes
        self.conf = conf
        self.cls = cls
        self.clslgd = classlegend

    def __str__(self):
        ret = ""
        for a,b in self.__dict__.items(): ret = f"{ret}{a}: {b}\n"
        return ret


class NumberPlateRecognition():
    def __init__(self):
        model_path = os.path.join(os.path.abspath("."),"models","first10ktrain","weights","best.onnx")
        self.model = YOLO(model_path, task='detect')

    def analyze(self,
                image: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor]) -> DetectionResults:
        result = self.model(image)[0]
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
