from .utils import *

from PIL import Image
from pathlib import Path
from copy import deepcopy
from IPython.display import clear_output

import os
from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Boxes, Results
import numpy as np
import torch
ImageInput = str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor


class ObjectDetection():
    def __init__(self, file_path: str = "."):
        self.result = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.file_path = file_path
        self.models: list[YOLO] = []
        self.add_model(os.path.join(os.path.abspath("."), "models", "first10ktrain", "weights", "best.pt"))
        self.add_model(os.path.join(os.path.abspath("."), "models", "yolo11n.pt"))

    def add_model(self, path: str):
        model = YOLO(path, task="detect")
        model.to(self.device)
        self.models.append(model)

    def map_classes_to_models(self, classes: list[str], verbose: bool = False) -> dict[int, list[int]]:
        # select the right model for each class
        model_class_dict: dict[int, list] = {}
        for model_idx in range(len(self.models)):
            model_classes = self.models[model_idx].names
            model_class_dict[model_idx] = []
            for target_class in classes.copy():
                if len(classes) == 0: break
                try:
                    class_idx = find_key_by_value(model_classes, target_class)
                    classes.remove(target_class)
                    model_class_dict[model_idx].append(class_idx)
                except Exception as e:
                    if verbose: print(e)
            else: continue
            break
        if len(classes) > 0: print(f"Could not find models for the following classes: {classes}")
        return model_class_dict

    def detect_objects(self, image: ImageInput, model_class_dict: dict[int, list[int]], verbose: bool = False) -> Results:
        # detect and combine results
        self.result = None
        for model_idx, class_indices in model_class_dict.items():
            if len(class_indices) == 0: continue
            detection_results = self.models[model_idx](image, classes=class_indices)[0].cpu().numpy()
            if self.result is None: self.result = detection_results
            else: self.result = merge_results(self.result, detection_results)
        if not verbose: clear_output()
        return self.result

    def get_classes(self) -> list[str]: return np.concatenate([list(model.names.values()) for model in self.models])

    def apply_blur_to_image(self, image: ImageInput, detection_results: Results, target_classes: list[str] | str) -> np.ndarray:
        if type(target_classes) is str: target_classes = [target_classes]
        image_copy = image.copy()
        for bbox, class_id in zip(detection_results.boxes.xyxy, detection_results.boxes.cls):
            if detection_results.names[class_id] not in target_classes: continue
            cropped_region = self.crop_image(image_copy, bbox)
            blurred_region = cv2.GaussianBlur(cropped_region, (25, 25), 0)
            x1, y1, x2, y2 = bbox.astype("int")
            image_copy[y1:y2, x1:x2] = blurred_region
        return image_copy

    def crop_image(self, image: ImageInput, bbox: np.ndarray) -> np.ndarray:
        assert len(bbox) == 4, "Array must have exactly 4 entries"
        x1, y1, x2, y2 = bbox.astype("int")
        return image[y1:y2, x1:x2]

    def chain_detection(self, image: ImageInput, primary_class_dict: dict[int, list[int]], secondary_class_dict: dict[int, list[int]], verbose: bool = False) -> Results:
        primary_results = self.detect_objects(image, primary_class_dict, verbose)
        merged_results = deepcopy(primary_results)

        for bbox in primary_results.boxes.xyxy:
            x1, y1, _, _ = bbox.astype("int")
            cropped_image = self.crop_image(image, bbox)

            secondary_results = self.detect_objects(cropped_image, secondary_class_dict, verbose)
            if secondary_results.boxes.data.size > 0:
                secondary_results.boxes.data[:, :4] += [x1, y1, x1, y1]

            merged_results = merge_results(merged_results, secondary_results)
        self.result = merged_results
        return self.result

    def process_image(self, image: ImageInput, primary_classes: list[str] | str, secondary_classes: list[str] | str = None, verbose: bool = False) -> Results:
        if issubclass(type(primary_classes), str): primary_classes = [primary_classes]
        if issubclass(type(secondary_classes), str): secondary_classes = [secondary_classes]

        primary_mapping = self.map_classes_to_models(primary_classes, verbose)
        if secondary_classes is None:
            return self.detect_objects(image, primary_mapping, verbose)
        else:
            secondary_mapping = self.map_classes_to_models(secondary_classes, False)
            return self.chain_detection(image, primary_mapping, secondary_mapping, verbose)

    def process_video(self, video_path: str, primary_classes: list[str] | str, secondary_classes: list[str] | str = None,
                      confidence_threshold: float = 0.5, iou_threshold: float = 0.7, video_stride: int = 1, enable_stream_buffer: bool = False,
                      debug: bool = False, verbose: bool = False):
        def debug_show_video(frame: ImageInput) -> bool:
            cv2.imshow("frame", cv2.resize(frame, (1200, 800)))
            return cv2.waitKey(1) & 0xFF == ord('q')

        if issubclass(type(primary_classes), str): primary_classes = [primary_classes]
        if issubclass(type(secondary_classes), str): secondary_classes = [secondary_classes]

        cap = cv2.VideoCapture(video_path)
        frame_counter = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            if frame_counter % video_stride != 0:
                frame_counter += 1
                continue
            detections = self.process_image(frame, primary_classes, secondary_classes, verbose)

            # TODO delete later is for testing
            frame = detections.plot()
            if debug and debug_show_video(frame): break

        cap.release()
        cv2.destroyAllWindows()
