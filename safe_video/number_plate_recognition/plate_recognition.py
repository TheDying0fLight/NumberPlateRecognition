import os
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import cv2

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
    def __init__(self, file_path: str = "."):
        self.file_path = file_path
        model_path = os.path.join(os.path.abspath("."),"models","first10ktrain","weights","best.pt")
        self.model = YOLO(model_path, task='detect')

    def analyze(self, image):
        self.result = self.model(image)[0]
        data: Boxes = self.result.boxes.cpu().numpy()
        return DetectionResults(data.xyxy,data.conf,data.cls,self.result.names)

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