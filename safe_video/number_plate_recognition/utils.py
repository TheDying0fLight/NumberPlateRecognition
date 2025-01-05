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


def filter_results(results: Results, class_filter: list[str]|str, confidence_threshold: float = 0) -> Results:
    if type(class_filter) is str: class_filter = [class_filter]

    class_filter = [find_key_by_value(results.names, cls) for cls in class_filter]
    filter_results = []
    for data in results.boxes.data:
        cls_idx = int(data[5])
        confidence = data[4]
        if cls_idx in class_filter and confidence >= confidence_threshold:
            filter_results.append(data)
            
    results.boxes.data = np.array(filter_results)
    return results


def apply_censorship(image: ImageInput, detection_results: Results,
                     action: Literal["blur", "beam", "overlay"] = None, color: tuple = (0, 0, 0), overlayImage: ImageInput = None) -> np.ndarray:

    def apply_blur_to_bbox(**kwargs) -> np.ndarray:
            def pixelate_region(region, pixel_size=10):
                height, width = region.shape[:2]
                
                # Ensure width and height do not become zero
                small_width = max(1, width // pixel_size)
                small_height = max(1, height // pixel_size)
                
                # Resize to a smaller size
                small = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
                # Scale back to the original size
                pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
                return pixelated
            
            cropped_region = crop_image(image, bbox)
            height, width = cropped_region.shape[:2]
            kernel_size = max(3, min(height , width))  # Adjust based on size
            if kernel_size % 2 == 0:  # Kernel size must be odd
                kernel_size += 1
            blurred_region = cv2.GaussianBlur(cropped_region, (kernel_size, kernel_size), 0)
            blurred_region = pixelate_region(blurred_region, pixel_size=10) 
            return blurred_region

    def apply_beam_to_bbox(**kwargs) -> np.ndarray:
        return color

    def apply_overlay_to_bbox(**kwargs) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype("int")
        return cv2.resize(overlayImage, (x2 - x1, y2 - y1))

    action_dict: dict[str, Callable] = {
        "blur": apply_blur_to_bbox,
        "beam": apply_beam_to_bbox,
        "overlay": apply_overlay_to_bbox
    }

    if action not in action_dict: raise ValueError(f"Invalid action: {action}")
    if action == "overlay" and overlayImage is None: raise ValueError("Overlay image must be provided for action 'overlay'")

    image_copy = image.copy()
    for bbox in detection_results.boxes.xyxy:
        x1, y1, x2, y2 = bbox.astype("int")
        modifiedRegion = action_dict[action](image=image_copy, bbox=bbox, x1=x1, y1=y1, x2=x2, y2=y2)
        image_copy[y1:y2, x1:x2] = modifiedRegion
    return image_copy


def crop_image(image: ImageInput, bbox: np.ndarray) -> np.ndarray:
    assert len(bbox) == 4, "Array must have exactly 4 entries"
    x1, y1, x2, y2 = bbox.astype("int")
    return image[y1:y2, x1:x2]
