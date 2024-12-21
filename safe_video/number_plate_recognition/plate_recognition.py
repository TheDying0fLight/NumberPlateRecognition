from .utils import *

from PIL import Image
from pathlib import Path
from IPython.display import clear_output

import os
from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Results
import numpy as np
import torch
ImageInput = str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor


class ObjectDetection():
    def __init__(self, file_path: str = "."):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.file_path = file_path
        self.models: list[YOLO] = []
        self.add_model(os.path.join(os.path.abspath("."), "models", "first10ktrain", "weights", "best.pt"))
        self.add_model(os.path.join(os.path.abspath("."), "models", "yolo11n.pt"))

    def apply_blur_to_image(self, image: ImageInput, detection_results: Results) -> np.ndarray:
        image_copy = image.copy()
        for bbox, _ in zip(detection_results.boxes.xyxy, detection_results.boxes.cls):
            cropped_region = self.crop_image(image_copy, bbox)
            blurred_region = cv2.GaussianBlur(cropped_region, (25, 25), 0)
            x1, y1, x2, y2 = bbox.astype("int")
            image_copy[y1:y2, x1:x2] = blurred_region
        return image_copy

    def add_model(self, path: str):
        model = YOLO(path, task="detect")
        model.to(self.device)
        self.models.append(model)

    def get_classes(self) -> list[str]: return np.concatenate([list(model.names.values()) for model in self.models])

    def crop_image(self, image: ImageInput, bbox: np.ndarray) -> np.ndarray:
        assert len(bbox) == 4, "Array must have exactly 4 entries"
        x1, y1, x2, y2 = bbox.astype("int")
        return image[y1:y2, x1:x2]

    def map_classes_to_models(self, classes: list[str]) -> dict[int, list[int]]:
        classes = classes.copy()
        cls_len = len(classes)
        if cls_len == 0: raise ValueError("No classes provided")
        # select the right model for each class
        model_class_dict: dict[int, list] = {}
        for model_idx in range(len(self.models)):
            model_classes = self.models[model_idx].names
            model_classes_vals = model_classes.values()
            model_class_dict[model_idx] = []
            for target_class in classes.copy():
                if len(classes) == 0: break
                if target_class in model_classes_vals:
                    class_idx = find_key_by_value(model_classes, target_class)
                    classes.remove(target_class)
                    model_class_dict[model_idx].append(class_idx)
            else: continue
            break
        if len(classes) > 0: print(f"Could not find models for the following classes: {classes}")
        if len(classes) == cls_len: raise ValueError("No classes found for any model")
        return model_class_dict

    def detect_objects(self, image: ImageInput, model_class_dict: dict[int, list[int]], verbose: bool = False) -> Results:
        # detect and combine results
        result = None
        for model_idx, class_indices in model_class_dict.items():
            if len(class_indices) == 0: continue
            detection_results = self.models[model_idx](image, classes=class_indices)[0].cpu().numpy()
            if result is None: result = detection_results
            else: result = merge_results(result, detection_results)
        if not verbose: clear_output()
        return result

    def chain_detection(self, image: ImageInput, primary_class_dict: dict[int, list[int]], secondary_class_dict: dict[int, list[int]], verbose: bool = False) -> Results:
        primary_results = self.detect_objects(image, primary_class_dict, verbose)
        secondary_results = None

        for bbox in primary_results.boxes.xyxy:
            x1, y1, _, _ = bbox.astype("int")
            cropped_image = self.crop_image(image, bbox)
            results = self.detect_objects(cropped_image, secondary_class_dict, verbose)
            if results.boxes.data.size > 0: results.boxes.data[:, :4] += [x1, y1, x1, y1]

            if secondary_results is None: secondary_results = results
            else: secondary_results = merge_results(secondary_results, results)
        return [primary_results, secondary_results]

    def process_image(self, image: ImageInput, primary_classes: list[str] | str, secondary_classes: list[str] | str = None,
                      remap_classes: bool = True, verbose: bool = False) -> list[Results]:
        if primary_classes is None: raise ValueError("Primary classes must be provided")
        if issubclass(type(primary_classes), str): primary_classes = [primary_classes]
        if issubclass(type(secondary_classes), str): secondary_classes = [secondary_classes]
        if (remap_classes): self._primary_mapping = self.map_classes_to_models(primary_classes)
        if secondary_classes is None: return [self.detect_objects(image, self._primary_mapping, verbose)]
        else:
            if (remap_classes): self._secondary_mapping = self.map_classes_to_models(secondary_classes)
            return self.chain_detection(image, self._primary_mapping, self._secondary_mapping, verbose)

    def process_video(self, video_path: str, primary_classes: list[str] | str, secondary_classes: list[str] | str = None,
                      confidence_threshold: float = 0.5, iou_threshold: float = 0.7, video_stride: int = 1, enable_stream_buffer: bool = False,
                      debug: bool = False, verbose: bool = False):
        def debug_show_video(frame: ImageInput) -> bool:
            cv2.imshow("frame", cv2.resize(frame, (1200, 800)))
            return cv2.waitKey(1) & 0xFF == ord('q')

        cap = cv2.VideoCapture(video_path)
        frame_counter = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            if frame_counter % video_stride != 0:
                frame_counter += 1
                continue
            detections = self.process_image(frame, primary_classes, secondary_classes, frame_counter == 0, verbose)

            # TODO delete later is for testing
            if debug:
                frame = merge_results_list(detections).plot()
                if debug_show_video(frame): break
            frame_counter += 1

        cap.release()
        if debug: cv2.destroyAllWindows()
        # TODO return detections in some way
