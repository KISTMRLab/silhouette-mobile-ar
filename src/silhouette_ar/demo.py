"""Local browser adapter for mask-derived silhouette geometry."""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .avatar_http import serve_avatar_asset

import cv2
import numpy as np

from .cli import serialize
from .geometry import CameraIntrinsics, build_meshes
from .interaction import astar_path, occlusion_composite, rasterize_world_footprints
from .segmentation import YoloSegmenter, connected_instances

ROOT = Path(__file__).resolve().parents[2]


def analyze(mask: np.ndarray, calibration: dict) -> dict:
    height, width = mask.shape
    intrinsics = CameraIntrinsics(float(calibration.get("fx", width)), float(calibration.get("fy", width)), width/2, height/2)
    camera_height = float(calibration.get("camera_height", 1.4))
    instances = connected_instances(mask, min_area=max(80, width*height//1000))
    meshes = build_meshes(instances, intrinsics, np.array([0., camera_height, 0.]), np.diag([1., -1., 1.]), np.array([0., 1., 0.]), 0.)
    output = {"width": width, "height": height, "meshes": [serialize(m) for m in meshes], "path": [],
              "calibration": {"fx": intrinsics.fx, "fy": intrinsics.fy, "cx": intrinsics.cx, "cy": intrinsics.cy, "camera_height": camera_height}}
    camera = np.full((height, width, 3), (35, 48, 57), np.uint8)
    camera[mask > 0] = (135, 190, 205)
    virtual = np.zeros((height, width, 4), np.uint8)
    virtual[:, width//3:2*width//3, :3] = (80, 90, 240)
    virtual[:, width//3:2*width//3, 3] = 210
    composite = occlusion_composite(camera, virtual, mask)
    ok, encoded = cv2.imencode(".png", composite)
    if ok:
        output["occlusion_preview"] = "data:image/png;base64," + base64.b64encode(encoded).decode()
    if meshes:
        contacts = np.array([m.floor_contact_world[[0, 2]] for m in meshes])
        min_x, min_z = contacts.min(axis=0)-1.2
        max_x, max_z = contacts.max(axis=0)+1.2
        bounds = (float(min_x), float(min_z), float(max_x), float(max_z))
        cell = .05
        grid = rasterize_world_footprints([m.floor_footprint_world for m in meshes], bounds, cell)
        start = (1, grid.shape[0]//2)
        goal = (grid.shape[1]-2, grid.shape[0]//2)
        output["path"] = [[bounds[0]+x*cell, 0, bounds[1]+y*cell] for x, y in astar_path(grid, start, goal)]
        output["navigation_bounds_xz"] = bounds
    return output


def app(segmenter=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if serve_avatar_asset(self, Path(__file__).resolve().parents[2] / "static"): return
            if self.path in {"/static/avatar.js", "/static/speech.js", "/static/vendor/three.module.js"}:
                body = (ROOT / self.path.lstrip("/")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/":
                self.send_error(404)
                return
            body = (ROOT / "demo" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                raw = base64.b64decode(payload["image"].split(",", 1)[-1])
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Invalid image")
                if self.path == "/api/mask":
                    mask = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8)
                    provenance = "user-supplied binary mask"
                elif self.path == "/api/segment":
                    if segmenter is None:
                        raise ValueError("No segmentation weights configured; restart with --weights")
                    instances = segmenter.segment(image)
                    mask = np.maximum.reduce([m.mask for m in instances]) if instances else np.zeros(image.shape[:2], np.uint8)
                    provenance = "YOLO segmentation inference"
                else:
                    self.send_error(404)
                    return
                answer = analyze(mask, payload.get("calibration", {}))
                answer["provenance"] = provenance
                status = 200
            except (ValueError, KeyError, TypeError) as exc:
                answer, status = {"error": str(exc)}, 400
            body = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Optional YOLO segmentation checkpoint")
    parser.add_argument("--port", type=int, default=8763)
    args = parser.parse_args()
    segmenter = YoloSegmenter(args.weights) if args.weights else None
    print(f"Silhouette demo at http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), app(segmenter)).serve_forever()


if __name__ == "__main__":
    main()
