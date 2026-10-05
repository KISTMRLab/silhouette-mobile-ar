import base64
from pathlib import Path

import cv2
import numpy as np

from silhouette_ar.dataset import inspect_dataset
from silhouette_ar.demo import analyze, handle_frame, handle_plan, handle_select

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
    assert np.allclose(plan["path"][-1][::2], np.array(plan["targets"]["stand"])[::2], atol=.03)


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
