from __future__ import annotations

from dataclasses import dataclass
import json
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


def class_instances(class_map: np.ndarray, class_names: list[str], min_area: int = 500,
                    confidence: np.ndarray | None = None) -> list[MaskInstance]:
    """Split a class-index map (0 = background) into per-class connected instances.

    Pixels >= len(class_names) (for example the 255 ignore label) are skipped.
    Each instance keeps its class name, so persistent tracking can match by class.
    """
    instances: list[MaskInstance] = []
    for class_id in sorted(int(v) for v in np.unique(class_map)):
        if class_id <= 0 or class_id >= len(class_names):
            continue
        name = class_names[class_id]
        for item in connected_instances(class_map == class_id, name, min_area):
            score = float(confidence[item.mask > 0].mean()) if confidence is not None else 1.0
            instances.append(MaskInstance(item.instance_id, name, score, item.mask))
    return instances


def polygons_to_mask(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    import cv2
    mask = np.zeros(shape, np.uint8)
    rings = [np.round(np.asarray(p, float)).astype(np.int32).reshape(-1, 1, 2) for p in polygons if len(p) >= 3]
    if rings:
        cv2.fillPoly(mask, rings, 1)
    return mask


class YoloSegmenter:
    """Ultralytics segmentation adapter.

    Masks are rebuilt from ``result.masks.xy`` polygons, which Ultralytics already
    maps to original-image pixels. Resizing ``masks.data`` would ignore the
    letterbox padding (masks are e.g. 384x640 for a 720x1280 frame) and shift
    contours by up to the padding height.
    """

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
        result = self.model.predict(frame, conf=self.confidence, device=self.device, retina_masks=True, verbose=False)[0]
        return yolo_result_instances(result, frame.shape[:2])


def yolo_result_instances(result, shape: tuple[int, int]) -> list[MaskInstance]:
    if result.masks is None or result.boxes is None:
        return []
    output: list[MaskInstance] = []
    for index, polygon in enumerate(result.masks.xy):
        mask = polygons_to_mask([np.asarray(polygon)], shape)
        if not mask.any():
            continue
        class_id = int(result.boxes.cls[index])
        track_id = str(index) if result.boxes.id is None else str(int(result.boxes.id[index]))
        name = str(result.names[class_id])
        output.append(MaskInstance(f"{name}-{track_id}", name, float(result.boxes.conf[index]), mask))
    return output


class UNetSegmenter:
    """Compressed U-Net inference from a PyTorch checkpoint (``.pt``) or ONNX export (``.onnx``).

    Frames are resized to the network input (192x192 by default, as in the paper),
    logits are resized back to the frame, and the class map is split into
    per-class connected instances.
    """

    def __init__(self, weights: str | Path, device: str = "cpu", min_area: int = 500):
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"Local weights do not exist: {path}")
        self.min_area = min_area
        if path.suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("Install ONNX inference support with: pip install -e .[onnx]") from exc
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.classes = list(meta["classes"])
            self.input_size = int(meta.get("input_size", 192))
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.model = None
        else:
            from .unet import load_checkpoint
            self.model, info = load_checkpoint(path, device)
            self.classes = list(info["classes"])
            self.input_size = int(info["config"].get("input_size", 192))
            self.session = None
        self.device = device

    def logits(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Class logits (C, H, W) at the network resolution."""
        import cv2
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        if self.session is not None:
            return self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0][0]
        import torch
        with torch.no_grad():
            return self.model(torch.from_numpy(tensor).to(self.device))[0].cpu().numpy()

    def class_map(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2
        logits = self.logits(frame_bgr)
        height, width = frame_bgr.shape[:2]
        resized = np.stack([cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR) for channel in logits])
        shifted = np.exp(resized - resized.max(axis=0, keepdims=True))
        probability = shifted / shifted.sum(axis=0, keepdims=True)
        return probability.argmax(axis=0).astype(np.uint8), probability.max(axis=0)

    def segment(self, frame_bgr: np.ndarray) -> list[MaskInstance]:
        class_map, confidence = self.class_map(frame_bgr)
        return class_instances(class_map, self.classes, self.min_area, confidence)


def load_segmenter(weights: str | Path, kind: str = "auto", device: str | None = None):
    """``unet`` for this repository's checkpoints/ONNX exports, ``yolo`` for Ultralytics weights."""
    path = Path(weights)
    if kind == "auto":
        kind = "unet" if path.suffix.lower() == ".onnx" else "yolo"
        if path.suffix.lower() == ".pt" and path.is_file():
            try:
                import torch
                payload = torch.load(path, map_location="cpu", weights_only=False)
                kind = "unet" if isinstance(payload, dict) and payload.get("architecture") == "compressed-unet" else "yolo"
            except Exception:
                kind = "yolo"
    if kind == "unet":
        return UNetSegmenter(path, device or "cpu")
    if kind == "yolo":
        return YoloSegmenter(path, device=device)
    raise ValueError(f"Unknown segmenter kind: {kind}")
