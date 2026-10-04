from pathlib import Path

import numpy as np

from silhouette_ar.dataset import inspect_dataset
from silhouette_ar.demo import analyze


def test_mask_reaches_mesh_targets_and_collision_aware_route():
    mask = np.zeros((120, 160), np.uint8)
    mask[30:104, 50:105] = 1
    result = analyze(mask, {"fx": 160, "fy": 160, "camera_height": 1.4})
    assert len(result["meshes"]) == 1
    mesh = result["meshes"][0]
    assert len(mesh["triangles"]) >= 2
    assert set(mesh["interaction_targets"]) == {"point", "approach", "touch", "ride"}
    assert len(result["path"]) > 2
    assert mesh["floor_footprint_world"][0][1] != mesh["floor_footprint_world"][2][1]


def test_dataset_inventory_does_not_infer_masks_from_images(tmp_path: Path):
    folder = tmp_path / "deer_doll_img"
    folder.mkdir()
    (folder / "frame.png").write_bytes(b"example")
    result = inspect_dataset(tmp_path)
    assert result["images"] == 1 and result["annotation_candidates"] == 0
    assert "No annotation-like files" in result["message"]
