import os
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

class NumberPlateRecognition():
    def __init__(self, file_path: str = "."):
        self.file_path = file_path
        model_path = os.path.join(os.path.abspath("."),"models","first10ktrain","weights","best.onnx")
        self.model = YOLO(model_path, task='detect')

    def analyze(self, image):
        result = self.model(image)[0]
        data: Boxes = result.boxes.cpu().numpy()
        return {'boxes': data.xyxy,
                'conf': data.conf,
                'cls': data.cls,
                'clsleg': result.names}

    def blur_image(self):
        print(f'Image at {self.file_path} has to be blurred')