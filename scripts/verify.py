"""Run procedural masks through segmentation-to-interaction code and write inspectable outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np

from silhouette_ar.geometry import CameraIntrinsics, build_meshes, camera_pose
from silhouette_ar.interaction import astar_path, obstacle_grid, occlusion_composite, plan_on_floor
from silhouette_ar.segmentation import class_instances, connected_instances
from silhouette_ar.selection import select_by_keyword, select_by_pixel
from silhouette_ar.tracking import SilhouetteTracker


def scene_masks(offset):
    class_map = np.zeros((480, 640), np.uint8)
    class_map[270:370, 120 + offset:200 + offset] = 1
    class_map[240:330, 420:480] = 2
    return class_map


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

    # Update loop, selection and an approach plan on a tilted camera (class-index masks).
    intrinsics, (origin, rotation) = CameraIntrinsics(500, 500, 320, 240), camera_pose(.6, 25)
    tracker, frames = SilhouetteTracker(), []
    for offset in (0, 20, 40):
        found = build_meshes(class_instances(scene_masks(offset), ["background", "zebra", "panda"], 500), intrinsics, origin, rotation, [0, 1, 0], 0)
        frames.append({m.instance_id: m.floor_contact_world.round(3).tolist() for m in tracker.update(found)})
    assert all(sorted(f) == ["panda-1", "zebra-1"] for f in frames), frames
    visible = tracker.visible()
    hit, _ = select_by_pixel(visible, (450, 300), intrinsics, origin, rotation)
    spoken = select_by_keyword(visible, "follow the zebra", origin)
    zebra = tracker.get("zebra-1")
    stand = zebra.interaction_targets()["stand"]
    plan = plan_on_floor([m.floor_footprint_world for m in visible], [0.6, 2.2], [stand[0], stand[2]], clearance_m=.06)
    assert hit.instance_id == "panda-1" and spoken.instance_id == "zebra-1" and plan["ok"]
    (output / "update-loop.json").write_text(json.dumps({"frames": frames, "ray_cast_pixel_450_300": hit.instance_id, "keyword": spoken.instance_id,
                                                        "zebra_targets": zebra.interaction_targets(), "approach_path_xz": plan["path_xz"]}, indent=2), encoding="utf-8")

    cli_output = output / "cli" / "silhouettes.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-m", "silhouette_ar.cli", "--mask", str(mask_path),
                    "--label", "demo-object", "--min-area", "100", "--fx", "100", "--fy", "100",
                    "--cx", "60", "--cy", "50", "--camera-height", "1.4", "--output", str(cli_output)],
                   check=True, env=environment)
    assert json.loads(cli_output.read_text(encoding="utf-8"))[0]["label"] == "demo-object"
    summary = f"verification passed: vertices={len(mesh.vertices_world)}; path-points={len(path)}; tracked ids stable over 3 frames; approach waypoints={len(plan['path_xz'])}"
    try:
        from silhouette_ar.unet import PAPER_CONFIG, build_unet, count_parameters
        summary += f"; compressed U-Net parameters={count_parameters(build_unet(2, **PAPER_CONFIG)):,}"
    except RuntimeError:
        summary += "; PyTorch not installed (U-Net check skipped)"
    print(f"{summary}; artifacts -> {output}")


if __name__ == "__main__":
    main()
