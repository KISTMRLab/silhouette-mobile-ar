import base64
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from silhouette_ar.dataset import inspect_dataset
from silhouette_ar.demo import analyze, handle_frame, handle_plan, handle_select
from silhouette_ar.interaction import follow_point, footprint_clearance, ride_seat

CALIBRATION = {"hfov": 60, "camera_height": .6, "pitch": 25}


def png(mask: np.ndarray) -> str:
    ok, data = cv2.imencode(".png", mask)
    assert ok
    return "data:image/png;base64," + base64.b64encode(data).decode()


def scene(offset: int = 0) -> np.ndarray:
    mask = np.zeros((480, 640), np.uint8)
    mask[260:360, 120 + offset:200 + offset] = 1  # bear
    mask[250:340, 420:480] = 2  # rabbit
    return mask


def test_mask_reaches_mesh_targets_and_collision_aware_route():
    mask = np.zeros((120, 160), np.uint8)
    mask[30:104, 50:105] = 1
    result = analyze(mask, {"fx": 160, "fy": 160, "camera_height": 1.4, "pitch": 15})
    assert len(result["meshes"]) == 1
    mesh = result["meshes"][0]
    assert len(mesh["triangles"]) >= 2
    assert {"point", "approach", "pet", "push", "ride"} <= set(mesh["interaction_targets"])
    assert len(result["path"]) >= 2
    assert np.allclose(result["path"][-1][::2], np.array(mesh["interaction_targets"]["stand"])[::2], atol=.03)


def test_update_loop_keeps_ids_follows_motion_and_keeps_class_labels():
    session = "pytest-tracking"
    first = handle_frame({"session": session, "reset": True, "mask": png(scene()), "mask_kind": "indexed",
                          "classes": ["background", "bear", "rabbit"], "calibration": CALIBRATION})
    labels = {m["instance_id"]: m["label"] for m in first["meshes"]}
    assert labels == {"bear-1": "bear", "rabbit-1": "rabbit"}
    before = next(m for m in first["meshes"] if m["label"] == "bear")["floor_contact_world"]
    second = handle_frame({"session": session, "mask": png(scene(offset=25)), "mask_kind": "indexed",
                           "classes": ["background", "bear", "rabbit"], "calibration": CALIBRATION})
    after = next(m for m in second["meshes"] if m["label"] == "bear")
    assert after["instance_id"] == "bear-1" and after["floor_contact_world"][0] > before[0]
    plan = handle_plan({"session": session, "target_id": "bear-1", "from": first["spawn"][::2], "interaction": "follow"})
    assert plan["ok"] and plan["path"] and plan["label"] == "bear"
    assert np.allclose(plan["path"][-1][::2], np.array(plan["targets"]["follow"])[::2], atol=.03)


def segment_distance(p, a, b) -> float:
    ab, ap = b - a, p - a
    t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(ap - t * ab))


def path_clearance(path, footprint) -> float:
    """Smallest signed floor distance from a polyline (sampled every 5 mm) to a footprint."""
    points = []
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, float)[::2], np.asarray(b, float)[::2]
        steps = max(1, int(np.linalg.norm(b - a) / .005))
        points += [a + (b - a) * k / steps for k in range(steps + 1)]
    return min(footprint_clearance(footprint, p) for p in points)


def test_follow_keeps_body_radius_stand_off_beside_moving_object():
    session, body = "pytest-follow", .08
    classes = ["background", "bear", "rabbit"]
    frame = handle_frame({"session": session, "reset": True, "mask": png(scene()), "mask_kind": "indexed", "classes": classes,
                          "calibration": CALIBRATION})
    start = frame["spawn"][::2]
    for offset in (0, 25, 50, 75):  # the bear walks to the right; each update re-plans
        frame = handle_frame({"session": session, "mask": png(scene(offset)), "mask_kind": "indexed", "classes": classes,
                              "calibration": CALIBRATION})
        bear = next(m for m in frame["meshes"] if m["label"] == "bear")
        footprint = np.asarray(bear["floor_footprint_world"])
        plan = handle_plan({"session": session, "target_id": bear["instance_id"], "from": start, "interaction": "follow",
                            "body_radius": body})
        assert plan["ok"] and plan["clearance"] == pytest.approx(.06 + body)
        goal = np.asarray(plan["targets"]["follow"])[::2]
        # The goal stays the full stand-off outside the footprint (root + body radius + clearance)...
        assert footprint_clearance(footprint, goal) >= .06 + body
        # ...and beside the object as seen from the camera, not in front of it.
        center, camera = footprint.mean(axis=0), np.zeros(2)
        to_goal, to_camera = goal - center, camera - center
        assert np.dot(to_goal, to_camera) / (np.linalg.norm(to_goal) * np.linalg.norm(to_camera)) < math.cos(math.radians(40))
        # The walk itself never brings the avatar's body into any mesh (grid-cell tolerance).
        for mesh in frame["meshes"]:
            assert path_clearance(plan["path"], np.asarray(mesh["floor_footprint_world"])) > body
        start = list(goal)


def test_follow_point_side_and_blocking():
    footprint = np.array([[-.1, .95], [.1, .95], [.1, 1.05], [-.1, 1.05]])
    left = follow_point(footprint, [0, 0], [-.5, .6], .15)
    right = follow_point(footprint, [0, 0], [.5, .6], .15)
    assert left[0] < -.2 and right[0] > .2  # stays on the avatar's side
    assert left[1] < 1.0 and right[1] < 1.0  # turned toward the camera
    for point in (left, right):
        assert .15 <= footprint_clearance(footprint, point) < .17
    blocked = follow_point(footprint, [0, 0], [-.5, .6], .15, blocked=lambda p: p[0] < 0)
    assert blocked[0] > 0 and footprint_clearance(footprint, blocked) >= .15


def test_ride_seat_is_on_the_body_top_inside_the_footprint():
    session = "pytest-ride"
    frame = handle_frame({"session": session, "reset": True, "mask": png(scene()), "mask_kind": "indexed",
                          "classes": ["background", "bear", "rabbit"], "calibration": CALIBRATION})
    for mesh in frame["meshes"]:
        seat = np.asarray(mesh["interaction_targets"]["ride"])
        heights = np.asarray(mesh["vertices_world"])[:, 1]
        assert footprint_clearance(np.asarray(mesh["floor_footprint_world"]), seat[::2]) <= -.03 + 1e-3  # over the object
        assert heights.max() - .02 <= seat[1] <= heights.max() + 1e-6  # at the (body) top of the silhouette
        assert np.allclose(seat[1], mesh["interaction_targets"]["pet"][1])


def test_ride_seat_slides_inside_footprint_and_keeps_height():
    footprint = np.array([[-.1, .95], [.1, .95], [.1, 1.05], [-.1, 1.05]])
    edge = ride_seat(footprint, [.02, .25, 1.049])  # body top on the far edge
    assert edge[1] == .25 and 0 <= edge[0] <= .02 and footprint_clearance(footprint, edge[::2]) <= -.03 + 1e-3
    inside = ride_seat(footprint, [0, .2, 1.0])
    assert np.allclose(inside, [0, .2, 1.0])  # already well inside: unchanged
    thin = np.array([[-.1, .99], [.1, .99], [.1, 1.01], [-.1, 1.01]])
    assert np.allclose(ride_seat(thin, [0, .2, 1.01])[::2], thin.mean(axis=0))  # too thin: centre


def test_ray_cast_and_keyword_selection():
    session = "pytest-select"
    handle_frame({"session": session, "reset": True, "mask": png(scene()), "mask_kind": "indexed",
                  "classes": ["background", "bear", "rabbit"], "calibration": CALIBRATION})
    hit = handle_select({"session": session, "pixel": [450, 300]})
    assert hit["ok"] and hit["target_id"] == "rabbit-1" and hit["method"] == "ray-cast"
    assert not handle_select({"session": session, "pixel": [320, 40]})["ok"]
    spoken = handle_select({"session": session, "keyword": "please go to the bears"})
    assert spoken["ok"] and spoken["target_id"] == "bear-1" and spoken["method"] == "keyword"


def test_dataset_inventory_does_not_infer_masks_from_images(tmp_path: Path):
    folder = tmp_path / "deer_doll_img"
    folder.mkdir()
    (folder / "frame.png").write_bytes(b"example")
    result = inspect_dataset(tmp_path)
    assert result["images"] == 1 and result["annotation_candidates"] == 0
    assert "No annotation-like files" in result["message"]
