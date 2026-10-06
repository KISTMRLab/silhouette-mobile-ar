"""Local browser adapter: camera/upload frames -> segmentation -> tracked silhouette meshes,
ray-cast/keyword selection and A* plans for the avatar's interactions."""
from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .avatar_http import serve_avatar_asset

import cv2
import numpy as np

from .cli import serialize
from .geometry import CameraIntrinsics, build_meshes, camera_pose
from .interaction import follow_point, footprint_clearance, plan_on_floor
from .segmentation import class_instances, connected_instances, load_segmenter
from .selection import select_by_keyword, select_by_pixel
from .tracking import SilhouetteTracker

ROOT = Path(__file__).resolve().parents[2]
ORT_DIR = ROOT / "static" / "vendor" / "onnxruntime"
FOLLOW_MARGIN_M = .03  # slack for the object moving between silhouette updates
SESSIONS: dict[str, dict] = {}
LOCK = threading.Lock()


def calibration_from(payload: dict, width: int, height: int) -> dict:
    """Intrinsics and pose from the browser's calibration fields (principal point at the image centre)."""
    if "fx" in payload:
        fx = float(payload["fx"])
    else:
        fx = (width / 2) / math.tan(math.radians(float(payload.get("hfov", 60.0))) / 2)
    fy = float(payload.get("fy", fx))
    camera_height = float(payload.get("camera_height", 1.4))
    pitch = float(payload.get("pitch", 0.0))
    if camera_height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("camera height and focal length must be positive")
    origin, rotation = camera_pose(camera_height, pitch, float(payload.get("yaw", 0.0)))
    return {"intrinsics": CameraIntrinsics(fx, fy, width / 2, height / 2), "origin": origin, "rotation": rotation,
            "public": {"fx": fx, "fy": fy, "cx": width / 2, "cy": height / 2, "camera_height": camera_height, "pitch": pitch,
                       "hfov": math.degrees(2 * math.atan(width / 2 / fx))},
            "contact": payload.get("contact", "bbox"), "clearance": float(payload.get("clearance", .06)),
            "min_area": int(payload.get("min_area", max(60, round(500 * width * height / (640 * 480)))))}


def floor_point(pixel, calibration: dict):
    ray = calibration["rotation"] @ calibration["intrinsics"].ray(np.asarray(pixel, float))
    if ray[1] >= -1e-6:
        return None
    t = -calibration["origin"][1] / ray[1]
    return calibration["origin"] + t * ray


def analyze_instances(instances, width: int, height: int, calibration_payload: dict, tracker: SilhouetteTracker | None = None) -> tuple[dict, list, dict]:
    calibration = calibration_from(calibration_payload, width, height)
    started = time.perf_counter()
    meshes = build_meshes(instances, calibration["intrinsics"], calibration["origin"], calibration["rotation"], [0.0, 1.0, 0.0], 0.0,
                          calibration["contact"])
    if tracker is not None:
        meshes = tracker.update(meshes)
    meshes.sort(key=lambda mesh: mesh.instance_id)
    output = {"width": width, "height": height, "meshes": [serialize(mesh, calibration["clearance"]) for mesh in meshes], "path": [],
              "calibration": calibration["public"], "camera": {"origin": calibration["origin"].tolist(), "rotation": calibration["rotation"].tolist()}}
    spawn = floor_point((width * .16, height * .86), calibration)
    if spawn is None:
        spawn = calibration["origin"] * np.array([1, 0, 1]) + np.array([0, 0, 1.0])
    output["spawn"] = [float(spawn[0]), 0.0, float(spawn[2])]
    if meshes:
        target = output["meshes"][0]["interaction_targets"]["stand"]
        plan = plan_on_floor([mesh.floor_footprint_world for mesh in meshes], spawn[[0, 2]], [target[0], target[2]], clearance_m=calibration["clearance"])
        output["path"] = [[x, 0.0, z] for x, z in plan["path_xz"]]
        output["navigation_bounds_xz"] = plan["bounds"]
    output["timing_ms"] = {"geometry": round((time.perf_counter() - started) * 1000, 1)}
    return output, meshes, calibration


def analyze(mask: np.ndarray, calibration: dict, label: str = "object") -> dict:
    """Binary mask -> connected instances -> meshes, targets and a path to the first object's stand point."""
    height, width = mask.shape
    instances = connected_instances(mask, label, min_area=max(80, width * height // 1000))
    return analyze_instances(instances, width, height, calibration)[0]


def decode_image(data_url: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
    if image is None:
        raise ValueError("Invalid image")
    return image


def session_state(key: str) -> dict:
    with LOCK:
        if key not in SESSIONS:
            if len(SESSIONS) > 32:
                SESSIONS.pop(next(iter(SESSIONS)))
            SESSIONS[key] = {"tracker": SilhouetteTracker(), "calibration": None, "meshes": []}
        return SESSIONS[key]


def handle_frame(payload: dict, segmenter=None) -> dict:
    """One update-loop step: segment (server) or read the supplied mask, then track."""
    state = session_state(str(payload.get("session", "default")))
    if payload.get("reset"):
        state["tracker"].reset()
    timing = {}
    mode = payload.get("segmentation", "mask")
    if mode == "server":
        if segmenter is None:
            raise ValueError("No segmentation weights configured; restart the demo with --weights")
        image = decode_image(payload["image"])
        started = time.perf_counter()
        instances = segmenter.segment(image)
        timing["segmentation"] = round((time.perf_counter() - started) * 1000, 1)
        provenance = f"server {type(segmenter).__name__} inference"
        height, width = image.shape[:2]
        calibration_payload = dict(payload.get("calibration", {}))
    else:
        mask = decode_image(payload["mask"], cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape
        calibration_payload = dict(payload.get("calibration", {}))
        minimum = int(calibration_payload.get("min_area", max(60, round(500 * width * height / (640 * 480)))))
        if payload.get("mask_kind", "binary") == "indexed":
            classes = list(payload.get("classes") or ["background", "object"])
            instances = class_instances(mask, classes, minimum)
        else:
            instances = connected_instances(mask > 127, str(payload.get("label", "object")), minimum)
        provenance = str(payload.get("provenance", "supplied mask"))
    result, meshes, calibration = analyze_instances(instances, width, height, calibration_payload, state["tracker"])
    with LOCK:
        state["calibration"], state["meshes"] = calibration, meshes
    result["timing_ms"].update(timing)
    result["provenance"] = provenance
    result["tracked_ids"] = [mesh.instance_id for mesh in meshes]
    return result


def handle_plan(payload: dict) -> dict:
    state = session_state(str(payload.get("session", "default")))
    tracker, calibration = state["tracker"], state["calibration"]
    if calibration is None:
        raise ValueError("No frame has been analysed in this session yet")
    target = tracker.get(str(payload.get("target_id", "")))
    if target is None:
        return {"ok": False, "reason": "target is not tracked any more", "path": []}
    targets = target.interaction_targets(calibration["clearance"])
    stand = targets["stand"]
    start = payload.get("from") or [stand[0], stand[2]]
    obstacles = [track.mesh.floor_footprint_world for track in tracker.tracks.values()]
    clearance = calibration["clearance"]
    interaction = payload.get("interaction", "approach")
    if interaction == "follow":
        # Following keeps the avatar's whole body outside every dilated footprint:
        # the grid is dilated by the body radius too, and the goal stands beside
        # the object (not in front of it), re-planned on every silhouette update.
        body = max(0.0, float(payload.get("body_radius", 0.0)))
        clearance += body
        others = [track.mesh.floor_footprint_world for track in tracker.tracks.values() if track.mesh is not target]
        goal = follow_point(target.floor_footprint_world, calibration["origin"][[0, 2]], start, clearance + FOLLOW_MARGIN_M,
                            blocked=lambda p: any(footprint_clearance(f, p) < clearance for f in others))
        stand = [float(goal[0]), 0.0, float(goal[1])]
        targets["follow"] = stand
        targets["follow_standoff"] = clearance + FOLLOW_MARGIN_M
    plan = plan_on_floor(obstacles, start, [stand[0], stand[2]], clearance_m=clearance)
    return {"ok": plan["ok"], "reason": plan.get("reason"), "path": [[x, 0.0, z] for x, z in plan["path_xz"]], "target_id": target.instance_id,
            "label": target.label, "targets": targets, "interaction": interaction, "clearance": clearance}


def handle_select(payload: dict) -> dict:
    state = session_state(str(payload.get("session", "default")))
    calibration, meshes = state["calibration"], list(state["meshes"])
    if calibration is None:
        raise ValueError("No frame has been analysed in this session yet")
    if payload.get("keyword"):
        mesh = select_by_keyword(meshes, str(payload["keyword"]), calibration["origin"])
        method, distance = "keyword", None
    else:
        pixel = payload.get("pixel") or [calibration["intrinsics"].cx, calibration["intrinsics"].cy]
        mesh, distance = select_by_pixel(meshes, pixel, calibration["intrinsics"], calibration["origin"], calibration["rotation"])
        method = "ray-cast"
        distance = None if mesh is None else round(float(distance), 3)
    if mesh is None:
        return {"ok": False, "method": method, "labels": sorted({m.label for m in meshes})}
    return {"ok": True, "method": method, "target_id": mesh.instance_id, "label": mesh.label, "distance": distance}


def app(segmenter=None, onnx_model: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, body: bytes, kind: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path.startswith("/static/vendor/onnxruntime/"):
                target = (ORT_DIR / path.rsplit("/", 1)[-1]).resolve()
                if target.parent == ORT_DIR.resolve() and target.is_file() and target.suffix in {".mjs", ".js", ".wasm"}:
                    kind = "application/wasm" if target.suffix == ".wasm" else "text/javascript"
                    return self.send_body(target.read_bytes(), kind)
                return self.send_error(404, "Run python scripts/prepare_onnx_web.py first")
            if serve_avatar_asset(self, Path(__file__).resolve().parents[2] / "static"): return
            if path in {"/static/avatar.js", "/static/speech.js", "/static/vendor/three.module.js"}:
                return self.send_body((ROOT / path.lstrip("/")).read_bytes(), "text/javascript")
            if path == "/demo/app.js":
                return self.send_body((ROOT / "demo" / "app.js").read_bytes(), "text/javascript")
            if path == "/api/status":
                status = {"segmenter": type(segmenter).__name__ if segmenter else None,
                          "classes": getattr(segmenter, "classes", None),
                          "onnx": bool(onnx_model), "onnx_runtime_web": (ORT_DIR / "ort.wasm.min.mjs").is_file()}
                return self.send_body(json.dumps(status).encode(), "application/json")
            if path in {"/api/model.onnx", "/api/model.json"} and onnx_model:
                source = onnx_model if path.endswith(".onnx") else onnx_model.with_suffix(".json")
                return self.send_body(source.read_bytes(), mimetypes.guess_type(source.name)[0] or "application/octet-stream")
            if path != "/":
                return self.send_error(404)
            self.send_body((ROOT / "demo" / "index.html").read_bytes(), "text/html; charset=utf-8")

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/frame":
                    answer = handle_frame(payload, segmenter)
                elif self.path == "/api/plan":
                    answer = handle_plan(payload)
                elif self.path == "/api/select":
                    answer = handle_select(payload)
                elif self.path == "/api/mask":
                    image = decode_image(payload["image"])
                    answer = analyze((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8), payload.get("calibration", {}))
                    answer["provenance"] = "user-supplied binary mask"
                elif self.path == "/api/segment":
                    if segmenter is None:
                        raise ValueError("No segmentation weights configured; restart with --weights")
                    image = decode_image(payload["image"])
                    answer = analyze_instances(segmenter.segment(image), image.shape[1], image.shape[0], payload.get("calibration", {}))[0]
                    answer["provenance"] = f"{type(segmenter).__name__} inference (per-instance labels)"
                else:
                    return self.send_error(404)
                status = 200
            except (ValueError, KeyError, TypeError) as exc:
                answer, status = {"error": str(exc)}, 400
            self.send_body(json.dumps(answer).encode(), "application/json", status)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Compressed U-Net checkpoint (.pt) / ONNX export, or YOLO segmentation weights, for server-side segmentation")
    parser.add_argument("--segmenter", choices=["auto", "unet", "yolo"], default="auto")
    parser.add_argument("--onnx", help="ONNX export (with its .json sidecar) served to the browser for in-browser segmentation")
    parser.add_argument("--port", type=int, default=8763)
    args = parser.parse_args()
    segmenter = load_segmenter(args.weights, args.segmenter) if args.weights else None
    onnx_model = Path(args.onnx).resolve() if args.onnx else None
    if onnx_model and not (onnx_model.is_file() and onnx_model.with_suffix(".json").is_file()):
        raise SystemExit(f"{onnx_model} or its .json sidecar is missing; run silhouette-seg export-onnx")
    print(f"Silhouette demo at http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), app(segmenter, onnx_model)).serve_forever()


if __name__ == "__main__":
    main()
