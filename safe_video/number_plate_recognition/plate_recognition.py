from PIL import Image
from pathlib import Path
from typing import Dict, List, Union

import os
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results
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

    def blur_image(self):
        raise NotImplementedError("TODO")

    def filtered_model(self, image, filters = None):
        results = self.model(image)
        # for filter in filters:
        #     if 
        for result in results:
            result.show()