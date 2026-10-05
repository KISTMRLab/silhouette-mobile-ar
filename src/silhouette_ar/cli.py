from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .geometry import CameraIntrinsics, build_meshes, camera_pose
from .segmentation import connected_instances, load_mask, load_segmenter


def serialize(mesh, clearance_m: float = .06):
    return {"instance_id": mesh.instance_id, "label": mesh.label, "confidence": round(float(mesh.confidence), 4),
            "contour_px": mesh.contour_px.tolist(), "vertices_world": mesh.vertices_world.tolist(), "triangles": mesh.triangles.tolist(),
            "floor_contact_world": mesh.floor_contact_world.tolist(), "floor_footprint_world": mesh.floor_footprint_world.tolist(),
            "interaction_targets": mesh.interaction_targets(clearance_m)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build view-dependent silhouette proxy geometry")
    parser.add_argument("--mask", help="Binary mask image")
    parser.add_argument("--image", help="RGB image used with --weights")
    parser.add_argument("--weights", help="Compressed U-Net checkpoint/ONNX export or YOLO segmentation weights")
    parser.add_argument("--segmenter", choices=["auto", "unet", "yolo"], default="auto")
    parser.add_argument("--label", default="object")
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--camera-height", type=float, default=1.4)
    parser.add_argument("--pitch", type=float, default=0.0, help="Downward camera tilt in degrees")
    parser.add_argument("--yaw", type=float, default=0.0, help="Camera heading about the vertical axis in degrees")
    parser.add_argument("--contact", choices=["bbox", "contour"], default="bbox", help="Floor reference: bounding-box bottom-midpoint (paper) or contour bottom")
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
        instances = load_segmenter(args.weights, args.segmenter).segment(frame)
    origin, rotation = camera_pose(args.camera_height, args.pitch, args.yaw)
    meshes = build_meshes(instances, CameraIntrinsics(args.fx, args.fy, args.cx, args.cy), origin, rotation,
                          [0.0, 1.0, 0.0], 0.0, args.contact)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([serialize(mesh) for mesh in meshes], indent=2), encoding="utf-8")
    print(f"wrote {len(meshes)} silhouette mesh(es) to {args.output}")


if __name__ == "__main__":
    main()
