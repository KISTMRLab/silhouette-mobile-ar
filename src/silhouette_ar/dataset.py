"""Inspect a user-downloaded DollDataset tree without assuming segmentation labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_HINTS = ("mask", "label", "annotation", "segmentation")


def inspect_dataset(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    files = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS and not any(h in str(p.relative_to(root)).casefold() for h in MASK_HINTS)]
    candidates = [p for p in files if any(h in str(p.relative_to(root)).casefold() for h in MASK_HINTS) and p.suffix.lower() in IMAGE_EXTENSIONS | {".json", ".txt"}]
    return {"root": str(root.resolve()), "images": len(images), "annotation_candidates": len(candidates),
            "sample_images": [str(p.relative_to(root)) for p in images[:8]],
            "sample_annotation_candidates": [str(p.relative_to(root)) for p in candidates[:8]],
            "message": "Inspect candidate semantics manually before treating them as masks." if candidates else "No annotation-like files found. Supply reviewed masks or run an optional segmenter; images alone do not establish ground-truth masks."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_dataset(args.root), indent=2))


if __name__ == "__main__":
    main()
