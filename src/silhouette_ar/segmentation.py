from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class MaskInstance:
    instance_id: str
    label: str
    confidence: float
    mask: np.ndarray


def load_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read mask: {path}")
    return (image > threshold).astype(np.uint8)


def connected_instances(mask: np.ndarray, label: str = "object", min_area: int = 500) -> list[MaskInstance]:
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    instances: list[MaskInstance] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
            continue
        instances.append(MaskInstance(f"{label}-{index}", label, 1.0, (labels == index).astype(np.uint8)))
    return instances


class YoloSegmenter:
    def __init__(self, weights: str | Path, confidence: float = .35, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install optional inference support with: pip install -e .[yolo]") from exc
        value = str(weights)
        if not Path(value).exists() and not re.fullmatch(r"yolo\d+[a-z-]*-seg\.pt", value, re.IGNORECASE):
            raise FileNotFoundError(f"Local weights do not exist: {value}")
        self.model = YOLO(value)
        self.confidence = confidence
        self.device = device

    def segment(self, frame: np.ndarray) -> list[MaskInstance]:
        import cv2
        result = self.model.predict(frame, conf=self.confidence, device=self.device, verbose=False)[0]
        if result.masks is None or result.boxes is None:
            return []
        output: list[MaskInstance] = []
        for index, tensor in enumerate(result.masks.data):
            mask = cv2.resize(tensor.cpu().numpy(), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            class_id = int(result.boxes.cls[index])
            track_id = str(index) if result.boxes.id is None else str(int(result.boxes.id[index]))
            output.append(MaskInstance(track_id, str(result.names[class_id]), float(result.boxes.conf[index]), (mask > .5).astype(np.uint8)))
        return output
