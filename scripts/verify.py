"""Turn a procedural segmentation mask into geometry and interaction outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np

from silhouette_ar.geometry import CameraIntrinsics, build_meshes
from silhouette_ar.interaction import astar_path, obstacle_grid, occlusion_composite
from silhouette_ar.segmentation import connected_instances


def main():
    output = Path("outputs/verify").resolve(); output.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((100, 120), np.uint8); mask[42:89, 40:81] = 1
    instances = connected_instances(mask, "demo-object", min_area=100)
    meshes = build_meshes(instances, CameraIntrinsics(100, 100, 60, 50),
                          np.array([0., 1.4, 0.]), np.diag([1., -1., 1.]),
                          np.array([0., 1., 0.]), 0.)
    camera = np.zeros((100, 120, 3), np.uint8)
    virtual = np.zeros((100, 120, 4), np.uint8); virtual[:, :, 1] = 255; virtual[:, :, 3] = 255
    composited = occlusion_composite(camera, virtual, mask)
    grid = obstacle_grid(mask.shape, [mask], clearance_px=2)
    path = astar_path(grid, (5, 60), (110, 60))
    assert len(meshes) == 1 and path and composited[60, 60].sum() == 0
    mesh = meshes[0]
    payload = {"instance_id": mesh.instance_id, "vertices_world": mesh.vertices_world.tolist(),
               "triangles": mesh.triangles.tolist(), "interaction_targets": mesh.interaction_targets(),
               "path": path}
    (output / "silhouette.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mask_path = output / "mask.png"
    cv2.imwrite(str(mask_path), mask * 255)
    cv2.imwrite(str(output / "occlusion-composite.png"), composited)
    cli_output = output / "cli" / "silhouettes.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-m", "silhouette_ar.cli", "--mask", str(mask_path),
                    "--label", "demo-object", "--min-area", "100", "--fx", "100", "--fy", "100",
                    "--cx", "60", "--cy", "50", "--camera-height", "1.4", "--output", str(cli_output)],
                   check=True, env=environment)
    assert json.loads(cli_output.read_text(encoding="utf-8"))[0]["label"] == "demo-object"
    print(f"verification passed: vertices={len(mesh.vertices_world)}; path-points={len(path)}; artifacts -> {output}")


if __name__ == "__main__":
    main()
