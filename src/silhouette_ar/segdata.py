"""Segmentation dataset contract, converters and paper augmentation.

Prepared dataset layout (shared by every route)::

    <root>/images/<stem>.jpg|.png   RGB frames
    <root>/masks/<stem>.png         8-bit class-index masks: 0 background,
                                    1..K object classes, 255 ignored pixels
    <root>/classes.json             {"classes": ["background", "<class 1>", ...]}
    <root>/splits.json              {"train": [stems], "val": [stems]}
    <root>/provenance.json          source, licence and per-item origin records

Routes: COCO instance annotations (``prepare_coco``), self-labelled frames with
binary or indexed PNG masks (``prepare_masks``) and Blender renders
(``prepare_synthetic``, fed by ``scripts/blender_synthetic.py``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import shutil
from urllib.request import urlopen

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IGNORE_INDEX = 255


# ---------------------------------------------------------------- contract

def write_dataset_files(root: Path, classes: list[str], stems: list[str], provenance: dict, val_fraction: float = .15, seed: int = 0) -> dict:
    root = Path(root)
    if classes[0] != "background":
        classes = ["background", *classes]
    stems = sorted(stems)
    shuffled = stems[:]
    random.Random(seed).shuffle(shuffled)
    count = int(round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    if val_fraction > 0 and len(shuffled) > 1:
        count = max(1, count)
    splits = {"train": sorted(shuffled[count:]), "val": sorted(shuffled[:count])}
    (root / "classes.json").write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    (root / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"root": str(root), "classes": classes, "train": len(splits["train"]), "val": len(splits["val"])}


def load_dataset(root: str | Path) -> dict:
    root = Path(root)
    classes = json.loads((root / "classes.json").read_text(encoding="utf-8"))["classes"]
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    images = {p.stem: p for p in (root / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    for split, stems in splits.items():
        missing = [s for s in stems if s not in images or not (root / "masks" / f"{s}.png").is_file()]
        if missing:
            raise ValueError(f"{split} split references missing image/mask pairs: {missing[:5]}")
    return {"root": root, "classes": classes, "splits": splits, "images": images}


def read_sample(dataset: dict, stem: str) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    image = cv2.imread(str(dataset["images"][stem]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(dataset["root"] / "masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise ValueError(f"Could not read sample {stem}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Mask and image sizes differ for {stem}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask.astype(np.uint8)


# ---------------------------------------------------------------- COCO route

def decode_rle(counts, height: int, width: int) -> np.ndarray:
    """Decode COCO RLE (uncompressed list or compressed string) to a binary mask."""
    if isinstance(counts, str):
        values: list[int] = []
        position = 0
        while position < len(counts):
            value, shift, more = 0, 0, True
            while more:
                code = ord(counts[position]) - 48
                value |= (code & 0x1F) << (5 * shift)
                more = bool(code & 0x20)
                position += 1
                shift += 1
                if not more and code & 0x10:
                    value |= -1 << (5 * shift)
            if len(values) > 2:
                value += values[-2]
            values.append(value)
        counts = values
    flat = np.zeros(height * width, np.uint8)
    cursor, fill = 0, 0
    for run in counts:
        flat[cursor:cursor + run] = fill
        cursor += run
        fill = 1 - fill
    return flat.reshape((width, height)).T


def annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    from .segmentation import polygons_to_mask
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return polygons_to_mask([np.asarray(poly, float).reshape(-1, 2) for poly in segmentation], (height, width))
    size = segmentation.get("size", [height, width])
    return decode_rle(segmentation["counts"], int(size[0]), int(size[1]))


def _download(url: str, target: Path, timeout: int = 60) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with urlopen(url, timeout=timeout) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(target)


def prepare_coco(annotations: str | Path, images_dir: str | Path, out: str | Path, categories: list[str], limit: int | None = None,
                 download_missing: bool = False, min_area: int = 0, val_fraction: float = .15, seed: int = 0) -> dict:
    """Convert COCO instance annotations into class-index masks for chosen categories.

    COCO has no animal-doll category; ``teddy bear`` is the closest public toy
    class. Crowd regions of a selected class become 255 (ignored). Images are
    read from ``images_dir`` or, with ``download_missing``, fetched one by one
    from their ``coco_url`` (each keeps its own Flickr licence, recorded in
    provenance.json; nothing is redistributed by this repository).
    """
    import cv2
    data = json.loads(Path(annotations).read_text(encoding="utf-8"))
    by_name = {c["name"]: c["id"] for c in data["categories"]}
    unknown = [name for name in categories if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown COCO categories: {unknown}")
    class_of = {by_name[name]: index + 1 for index, name in enumerate(categories)}
    grouped: dict[int, list[dict]] = {}
    for annotation in data["annotations"]:
        if annotation["category_id"] in class_of and annotation.get("area", 1) >= min_area:
            grouped.setdefault(annotation["image_id"], []).append(annotation)
    infos = {image["id"]: image for image in data["images"]}
    licences = {item["id"]: item for item in data.get("licenses", [])}
    out, images_dir = Path(out), Path(images_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    stems, records = [], []
    for image_id in sorted(grouped)[: limit or None]:
        info = infos[image_id]
        source = images_dir / info["file_name"]
        if not source.is_file():
            if not download_missing:
                continue
            _download(info["coco_url"], source)
        height, width = int(info["height"]), int(info["width"])
        mask = np.zeros((height, width), np.uint8)
        for annotation in sorted(grouped[image_id], key=lambda a: -a.get("area", 0)):
            region = annotation_mask(annotation, height, width) > 0
            mask[region] = IGNORE_INDEX if annotation.get("iscrowd") else class_of[annotation["category_id"]]
        stem = Path(info["file_name"]).stem
        shutil.copyfile(source, out / "images" / source.name)
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), mask)
        stems.append(stem)
        licence = licences.get(info.get("license"), {})
        records.append({"stem": stem, "coco_image_id": image_id, "coco_url": info.get("coco_url"), "flickr_url": info.get("flickr_url"),
                        "license": licence.get("name"), "license_url": licence.get("url"), "instances": len(grouped[image_id])})
    if not stems:
        raise ValueError("No images converted; check --images-dir or pass --download-missing")
    provenance = {"route": "coco", "annotations": str(annotations), "categories": categories,
                  "note": "COCO annotations CC BY 4.0; images keep their Flickr licences. Do not redistribute the prepared folder.",
                  "items": records}
    return write_dataset_files(out, ["background", *categories], stems, provenance, val_fraction, seed)


def fetch_coco_annotations(out_dir: str | Path, split: str = "val2017") -> Path:
    """Download the official 2017 annotation archive and extract one instances file."""
    import zipfile
    out_dir = Path(out_dir)
    target = out_dir / "annotations" / f"instances_{split}.json"
    if target.is_file():
        return target
    archive = out_dir / "annotations_trainval2017.zip"
    if not archive.is_file():
        _download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", archive, timeout=600)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extract(f"annotations/instances_{split}.json", out_dir)
    return target


# ---------------------------------------------------------------- self-labelled route

def read_index_mask(path: Path) -> np.ndarray:
    """Read a mask keeping palette indices (PIL when available) or grey values."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            if image.mode == "P":
                return np.asarray(image, np.uint8)
            return np.asarray(image.convert("L"), np.uint8)
    except ImportError:
        import cv2
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask {path}")
        return mask


def prepare_masks(images_dir: str | Path, masks_dir: str | Path, out: str | Path, classes: list[str], class_index: int = 1,
                  val_fraction: float = .15, seed: int = 0) -> dict:
    """Pair frames with reviewed masks of the same stem.

    Binary masks (values {0,1} or {0,255}) become ``class_index``. Indexed masks
    (palette or grey values 0..K, 255 ignored) keep their indices and must match
    ``classes`` (without background).
    """
    import cv2
    images_dir, masks_dir, out = Path(images_dir), Path(masks_dir), Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    names = ["background", *classes]
    masks = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() == ".png"}
    stems, records = [], []
    for image in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        if image.stem not in masks:
            continue
        mask = read_index_mask(masks[image.stem])
        values = set(np.unique(mask).tolist())
        if values <= {0, 255} or values <= {0, 1}:
            mask = np.where(mask > 0, class_index, 0).astype(np.uint8)
            kind = "binary"
        elif max(values - {IGNORE_INDEX}) < len(names):
            kind = "indexed"
        else:
            raise ValueError(f"{masks[image.stem]} has values {sorted(values)[:8]}; expected binary or indices < {len(names)}")
        frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch for {image.name}")
        shutil.copyfile(image, out / "images" / image.name)
        cv2.imwrite(str(out / "masks" / f"{image.stem}.png"), mask)
        stems.append(image.stem)
        records.append({"stem": image.stem, "mask": kind})
    if not stems:
        raise ValueError("No image/mask pairs with matching stems were found")
    return write_dataset_files(out, names, stems, {"route": "self-labelled", "images": str(images_dir), "masks": str(masks_dir), "items": records}, val_fraction, seed)


# ---------------------------------------------------------------- Blender synthetic route

def prepare_synthetic(renders_dir: str | Path, out: str | Path, textures: str | Path | None = None, seed: int = 0, val_fraction: float = .15) -> dict:
    """Composite Blender RGBA renders over DTD (or fallback) textures with exact masks.

    Expects ``scripts/blender_synthetic.py`` output: ``<id>.png`` (RGBA),
    ``<id>_obj<k>.png`` per-object visibility masks, ``<id>.json`` and
    ``metadata.json`` (classes and asset provenance).
    """
    import cv2
    renders_dir, out = Path(renders_dir), Path(out)
    metadata = json.loads((renders_dir / "metadata.json").read_text(encoding="utf-8"))
    classes = ["background", *metadata["classes"]]
    bank = TextureBank(textures, seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    stems = []
    for record_path in sorted(renders_dir.glob("*.json")):
        if record_path.name == "metadata.json":
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        rgba = cv2.imread(str(renders_dir / record["image"]), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            raise ValueError(f"{record['image']} must be an RGBA render")
        height, width = rgba.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        for item in record["objects"]:
            visible = cv2.imread(str(renders_dir / item["mask"]), cv2.IMREAD_UNCHANGED)
            alpha = visible[:, :, 3] if visible.ndim == 3 and visible.shape[2] == 4 else (visible if visible.ndim == 2 else visible[:, :, 0])
            mask[alpha > 127] = classes.index(item["class"])
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        background = cv2.cvtColor(bank.sample(height, width), cv2.COLOR_RGB2BGR).astype(np.float32)
        image = (rgba[:, :, :3].astype(np.float32) * alpha + background * (1 - alpha)).astype(np.uint8)
        stem = record_path.stem
        cv2.imwrite(str(out / "images" / f"{stem}.png"), image)
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), mask)
        stems.append(stem)
    if not stems:
        raise ValueError(f"No renders found in {renders_dir}")
    provenance = {"route": "blender-synthetic", "renders": str(renders_dir), "textures": str(textures) if textures else "procedural fallback",
                  "assets": metadata.get("assets", []), "generator": metadata.get("generator")}
    return write_dataset_files(out, classes, stems, provenance, val_fraction, seed)


# ---------------------------------------------------------------- augmentation

class TextureBank:
    """Background textures for replacement: DTD images when a root is given, else procedural."""

    def __init__(self, root: str | Path | None = None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if root else []
        if root and not self.paths:
            raise ValueError(f"No texture images under {root}")

    def sample(self, height: int, width: int) -> np.ndarray:
        import cv2
        if self.paths:
            image = cv2.imread(str(self.paths[int(self.rng.integers(len(self.paths)))]), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                scale = max(height / image.shape[0], width / image.shape[1]) * float(self.rng.uniform(1.0, 1.6))
                image = cv2.resize(image, (max(width, int(math.ceil(image.shape[1] * scale))), max(height, int(math.ceil(image.shape[0] * scale)))))
                y = int(self.rng.integers(image.shape[0] - height + 1))
                x = int(self.rng.integers(image.shape[1] - width + 1))
                return image[y:y + height, x:x + width].copy()
        return procedural_texture(height, width, self.rng)


def procedural_texture(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Random stripes, checks, blotches or noise; a fallback when DTD is not downloaded."""
    import cv2
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    colors = rng.integers(0, 256, (2, 3)).astype(np.float32)
    kind = int(rng.integers(4))
    if kind == 0:
        angle, period = rng.uniform(0, math.pi), rng.uniform(6, 40)
        weight = .5 + .5 * np.sin((x * math.cos(angle) + y * math.sin(angle)) * 2 * math.pi / period)
    elif kind == 1:
        size = int(rng.integers(6, 40))
        weight = (((x // size) + (y // size)) % 2).astype(np.float32)
    elif kind == 2:
        small = rng.random((max(2, height // 16), max(2, width // 16))).astype(np.float32)
        weight = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    else:
        weight = rng.random((height, width)).astype(np.float32)
        weight = cv2.GaussianBlur(weight, (0, 0), float(rng.uniform(.5, 3)))
        weight = (weight - weight.min()) / max(1e-6, float(np.ptp(weight)))
    texture = colors[0] * weight[:, :, None] + colors[1] * (1 - weight[:, :, None])
    return np.clip(texture + rng.normal(0, 6, texture.shape), 0, 255).astype(np.uint8)


def replace_background(image: np.ndarray, mask: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Keep labelled (and ignored) pixels; replace background pixels with the texture."""
    return np.where((mask > 0)[:, :, None], image, texture).astype(np.uint8)


def _warp(image: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    height, width = mask.shape
    warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, warped_mask


def random_rotation(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rotate by an angle drawn from the full 0-360 degree range (device rotation)."""
    import cv2
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), float(rng.uniform(0, 360)), 1.0)
    return _warp(image, mask, matrix)


def random_zoom(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, area_range: tuple[float, float] = (.04, .5)) -> tuple[np.ndarray, np.ndarray]:
    """Change the object's share of the frame (zoom in/out) around the object centre."""
    foreground = (mask > 0) & (mask != IGNORE_INDEX)
    ratio = float(foreground.mean())
    if ratio <= 0:
        return image, mask
    scale = float(np.clip(math.sqrt(rng.uniform(*area_range) / ratio), .35, 3.0))
    ys, xs = np.nonzero(foreground)
    cx, cy = float(xs.mean()), float(ys.mean())
    height, width = mask.shape
    tx = rng.uniform(.3, .7) * width
    ty = rng.uniform(.3, .7) * height
    matrix = np.array([[scale, 0, tx - scale * cx], [0, scale, ty - scale * cy]], np.float32)
    return _warp(image, mask, matrix)


def random_flip(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < .5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < .5:
        image, mask = image[::-1], mask[::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, textures: TextureBank | None = None,
            p_background: float = .5, p_rotate: float = .5, p_zoom: float = .5, p_flip: float = .5) -> tuple[np.ndarray, np.ndarray]:
    """Paper augmentation: DTD background replacement, 0-360 degree rotation, zoom
    (object-area ratio) and horizontal/vertical flips. Geometric steps run first so
    replacement also fills the borders they expose."""
    if rng.random() < p_rotate:
        image, mask = random_rotation(image, mask, rng)
    if rng.random() < p_zoom:
        image, mask = random_zoom(image, mask, rng)
    if rng.random() < p_flip:
        image, mask = random_flip(image, mask, rng)
    if textures is not None and rng.random() < p_background:
        image = replace_background(image, mask, textures.sample(*mask.shape))
    return image, mask
