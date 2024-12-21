from ultralytics.engine.results import Boxes, Results
from copy import deepcopy
from PIL import Image
from pathlib import Path
from typing import Literal, Callable

import numpy as np
import cv2
import torch
ImageInput = str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor


def merge_results(result1: Results, result2: Results) -> Results:
    """
    Merges the bounding boxes of two YOLO results and also updates the class mapping.

    Args:
        result1 (Results): First YOLO result
        result2 (Results): Second YOLO result

    Returns:
        Results: Merged YOLO result containing all bounding boxes from both results and also updated class mapping
    """
    if result1 is None: return result2
    if result2 is None: return result1
    boxes1: Boxes = result1.boxes
    boxes2: Boxes = deepcopy(result2.boxes)
    merged_result: Results = deepcopy(result1)
    updated_class_mapping: dict[int, str] = {}
    current_max_class_idx: int = max(result1.names.keys(), default=-1)

    # check for existing classes in results1 and append new classes from results2
    for class_id2, class_name2 in result2.names.items():
        existing_class_id = next((k for k, v in result1.names.items() if v == class_name2), None)
        if existing_class_id is not None:
            updated_class_mapping[class_id2] = existing_class_id
        else:
            current_max_class_idx += 1
            updated_class_mapping[class_id2] = current_max_class_idx
            merged_result.names[current_max_class_idx] = class_name2

    # remap classes in second results
    for i, class_id in enumerate(boxes2.data[:, -1]):
        boxes2.data[i, -1] = updated_class_mapping[int(class_id)]

    merged_data = np.vstack([boxes1.data, boxes2.data]) if boxes1.data.size > 0 else boxes2.data
    merged_result.boxes.data = merged_data
    return merged_result


def merge_results_list(results: list[Results]) -> Results:
    """
    Merges the bounding boxes of multiple YOLO results and also updates the class mapping.

    Args:
        results (list[Results]): List of YOLO results

    Returns:
        Results: Merged YOLO result containing all bounding boxes from all results and also updated class mapping
    """
    merged_result: Results = None
    for result in results: merged_result = merge_results(merged_result, result)
    return merged_result


def find_key_by_value(dictionary: dict, value: str) -> int:
    return list(dictionary.keys())[list(dictionary.values()).index(value)]


def filter_results(results: Results, class_filter: list[str] | str) -> Results:
    if type(class_filter) is str: class_filter = [class_filter]

    class_filter = [find_key_by_value(results.names, cls) for cls in class_filter]
    results.boxes.data = np.array([d for d in results.boxes.data if d[-1] in class_filter])
    return results


def apply_censorship(image: ImageInput, detection_results: Results,
                     action: Literal["blur", "beam", "image"] = None, color: tuple = (0, 0, 0), overlayImage: ImageInput = (0, 0, 0)) -> np.ndarray:

    def apply_blur_to_bbox(image: ImageInput, bbox: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(crop_image(image, bbox), (25, 25), 0)

    def apply_beam_to_bbox(image: ImageInput, bbox: np.ndarray) -> np.ndarray:
        return color

    def apply_overlay_to_bbox(image: ImageInput, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype("int")
        return cv2.resize(overlayImage, (x2 - x1, y2 - y1))

    action_dict: dict[str, Callable] = {
        "blur": apply_blur_to_bbox,
        "beam": apply_beam_to_bbox,
        "image": apply_overlay_to_bbox
    }

    if action not in action_dict: raise ValueError(f"Invalid action: {action}")

    image_copy = image.copy()
    for bbox, _ in zip(detection_results.boxes.xyxy, detection_results.boxes.cls):
        x1, y1, x2, y2 = bbox.astype("int")
        modifiedRegion = action_dict[action](image_copy, bbox)
        image_copy[y1:y2, x1:x2] = modifiedRegion
    return image_copy


def crop_image(image: ImageInput, bbox: np.ndarray) -> np.ndarray:
    assert len(bbox) == 4, "Array must have exactly 4 entries"
    x1, y1, x2, y2 = bbox.astype("int")
    return image[y1:y2, x1:x2]
