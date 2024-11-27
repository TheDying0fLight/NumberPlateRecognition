from PIL import Image
from pathlib import Path
from typing import Dict, List, Union

import os
from ultralytics import YOLO
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
    def __init__(self):
        model_path = os.path.join(os.path.abspath("."),"models","first10ktrain","weights","best.onnx")
        self.model = YOLO(model_path, task='detect')

    def analyze(self,
                image: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor]) -> DetectionResults:
        result: Results = self.model(image)[0]
        data: Boxes = result.boxes.cpu().numpy()
        return DetectionResults(data.xyxy,data.conf,data.cls,result.names)

    def blur_image(self):
        raise NotImplementedError("TODO")