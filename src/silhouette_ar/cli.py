from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import CameraIntrinsics, build_meshes
from .segmentation import YoloSegmenter, connected_instances, load_mask


def serialize(mesh):
    return {"instance_id": mesh.instance_id, "label": mesh.label, "contour_px": mesh.contour_px.tolist(), "vertices_world": mesh.vertices_world.tolist(), "triangles": mesh.triangles.tolist(), "floor_contact_world": mesh.floor_contact_world.tolist(), "floor_footprint_world": mesh.floor_footprint_world.tolist(), "interaction_targets": mesh.interaction_targets()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build view-dependent silhouette proxy geometry")
    parser.add_argument("--mask", help="Binary mask image")
    parser.add_argument("--image", help="RGB image used with --weights")
    parser.add_argument("--weights", help="YOLO segmentation weights")
    parser.add_argument("--label", default="object")
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--camera-height", type=float, default=1.4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.mask) == bool(args.image and args.weights):
        parser.error("Provide either --mask, or both --image and --weights")
    if args.mask:
        instances = connected_instances(load_mask(args.mask), args.label, args.min_area)
    else:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        instances = YoloSegmenter(args.weights).segment(frame)
    # Camera pixels use y-down; this pose maps them into a y-up world.
    rotation = np.diag([1.0, -1.0, 1.0])
    meshes = build_meshes(instances, CameraIntrinsics(args.fx, args.fy, args.cx, args.cy), np.array([0.0, args.camera_height, 0.0]), rotation, np.array([0.0, 1.0, 0.0]), 0.0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([serialize(mesh) for mesh in meshes], indent=2), encoding="utf-8")
    print(f"wrote {len(meshes)} silhouette mesh(es) to {args.output}")


if __name__ == "__main__":
    main()
