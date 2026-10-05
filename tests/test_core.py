import math
import warnings

import cv2
import numpy as np
import pytest

from silhouette_ar.geometry import CameraIntrinsics, build_meshes, camera_pose, pitch_from_floor_tap
from silhouette_ar.interaction import astar_path, obstacle_grid, occlusion_composite, octile_distance, plan_on_floor
from silhouette_ar.segmentation import MaskInstance, connected_instances

TARGET_KEYS = {"point", "approach", "stand", "touch", "pet", "push", "ride", "top_height", "footprint_center"}


def rectangle_mask():
    mask = np.zeros((100, 120), np.uint8)
    cv2.rectangle(mask, (42, 45), (78, 88), 1, -1)
    return mask


def test_mask_becomes_triangulated_view_dependent_mesh():
    instances = connected_instances(rectangle_mask(), min_area=20)
    mesh = build_meshes(instances, CameraIntrinsics(100, 100, 60, 50), np.array([0., 1.4, 0.]), np.diag([1., -1., 1.]), np.array([0., 1., 0.]), 0.)[0]
    assert mesh.vertices_world.shape[1] == 3
    assert len(mesh.triangles) == len(mesh.contour_px) - 2
    assert np.isclose(mesh.floor_contact_world[1], 0)
    assert set(mesh.interaction_targets()) == TARGET_KEYS
    # Equal-distance (camera-centred) vertices, not a plane.
    distances = np.linalg.norm(mesh.vertices_world - mesh.camera_origin, axis=1)
    assert np.allclose(distances, distances[0])


def test_mask_occludes_virtual_layer_and_blocks_path():
    mask = rectangle_mask()
    camera = np.zeros((100, 120, 3), np.uint8)
    virtual = np.zeros((100, 120, 4), np.uint8); virtual[:, :, 2] = 255; virtual[:, :, 3] = 255
    composited = occlusion_composite(camera, virtual, mask, virtual_in_front=False)
    assert composited[60, 60].sum() == 0 and composited[10, 10, 2] == 255
    grid = obstacle_grid(mask.shape, [mask], clearance_px=1)
    path = astar_path(grid, (5, 60), (110, 60))
    assert path and all(not grid[y, x] for x, y in path)


def test_pose_helper_matches_legacy_convention_and_tilts_down():
    origin, rotation = camera_pose(1.4, 0, 0)
    assert np.allclose(rotation, np.diag([1., -1., 1.])) and np.allclose(origin, [0, 1.4, 0])
    _, tilted = camera_pose(1.4, 30, 0)
    forward = tilted @ np.array([0, 0, 1.0])
    assert forward[1] < 0 and np.isclose(math.degrees(math.atan2(-forward[1], forward[2])), 30)


def test_one_tap_floor_calibration_recovers_pitch():
    intrinsics = CameraIntrinsics(500, 500, 320, 240)
    origin, rotation = camera_pose(0.6, 25, 0)
    floor = np.array([0.0, 0.0, 1.2])
    camera = rotation.T @ (floor - origin)
    pixel_y = intrinsics.cy + intrinsics.fy * camera[1] / camera[2]
    assert math.isclose(pitch_from_floor_tap(pixel_y, intrinsics, 0.6, 1.2), 25, abs_tol=1e-6)


def yawed_mesh(yaw_deg: float, pitch_deg: float = 0.0):
    mask = np.zeros((480, 640), np.uint8)
    cv2.rectangle(mask, (220, 200), (420, 400), 1, -1)
    origin, rotation = camera_pose(1.4, pitch_deg, yaw_deg)
    return build_meshes(connected_instances(mask, min_area=20), CameraIntrinsics(500, 500, 320, 240), origin, rotation, [0, 1, 0], 0)[0]


def test_footprint_is_world_space_for_any_yaw():
    """Audit regression: a 90 degree yaw used to shrink a ~1.4 m wide object to a 0.16 m strip."""
    straight, turned = yawed_mesh(0), yawed_mesh(90)
    width = np.ptp(straight.vertices_world[:, 0])
    assert width > 1.0
    assert np.isclose(np.ptp(straight.floor_footprint_world[:, 0]), width, rtol=.02)
    assert np.isclose(np.ptp(turned.floor_footprint_world[:, 1]), width, rtol=.02)  # now spans world z
    assert np.ptp(turned.floor_footprint_world[:, 0]) < .5 * width


def test_tilted_view_footprint_comes_from_the_mesh_projection():
    mesh = yawed_mesh(0, pitch_deg=35)
    depth = np.ptp(mesh.floor_footprint_world[:, 1])
    top = mesh.vertices_world[np.argmax(mesh.vertices_world[:, 1])]
    assert depth > np.ptp(mesh.vertices_world[:, 2]) * .9
    assert top[2] > mesh.floor_contact_world[2] + .2  # the incline leans away from the camera


def test_stand_point_is_outside_the_hole_toward_the_camera_and_reachable():
    """Audit regression: the old approach target was the floor contact inside the hole (unreachable)."""
    mesh = yawed_mesh(0, pitch_deg=20)
    targets = mesh.interaction_targets(clearance_m=.06)
    stand = np.array(targets["stand"])
    polygon = mesh.floor_footprint_world.astype(np.float32).reshape(-1, 1, 2)
    assert -cv2.pointPolygonTest(polygon, (float(stand[0]), float(stand[2])), True) >= .06
    centre = np.array(targets["footprint_center"])
    assert np.linalg.norm(stand[[0, 2]] - mesh.camera_origin[[0, 2]]) < np.linalg.norm(centre[[0, 2]] - mesh.camera_origin[[0, 2]])
    plan = plan_on_floor([mesh.floor_footprint_world], [-2.0, mesh.floor_contact_world[2] + 1.5], stand[[0, 2]], clearance_m=.06)
    assert plan["ok"] and not plan["goal_snapped"]
    assert np.allclose(plan["path_xz"][-1], stand[[0, 2]], atol=.03)


def test_ride_and_pet_use_the_body_top_not_a_thin_ear():
    mask = np.zeros((480, 640), np.uint8)
    cv2.ellipse(mask, (320, 330), (60, 70), 0, 0, 360, 1, -1)
    cv2.rectangle(mask, (300, 180), (308, 270), 1, -1)  # thin ear
    origin, rotation = camera_pose(.6, 25)
    mesh = build_meshes(connected_instances(mask, min_area=20), CameraIntrinsics(500, 500, 320, 240), origin, rotation, [0, 1, 0], 0)[0]
    targets = mesh.interaction_targets()
    assert targets["ride"][1] < targets["touch"][1] - .05
    assert targets["ride"][1] > .5 * targets["top_height"] * .5
    assert abs(targets["ride"][0] - mesh.floor_contact_world[0]) < .03 and targets["pet"] == targets["ride"]


def test_bounding_box_bottom_midpoint_is_default_and_contour_mode_remains():
    mask = np.zeros((200, 200), np.uint8)
    cv2.fillPoly(mask, [np.array([[40, 60], [160, 60], [160, 150], [60, 170]], np.int32)], 1)
    instances = connected_instances(mask, min_area=20)
    origin, rotation = camera_pose(1.0, 10)
    intrinsics = CameraIntrinsics(200, 200, 100, 100)
    bbox = build_meshes(instances, intrinsics, origin, rotation, [0, 1, 0], 0)[0]
    legacy = build_meshes(instances, intrinsics, origin, rotation, [0, 1, 0], 0, contact_mode="contour")[0]
    assert bbox.floor_contact_world[0] == pytest.approx(0.0, abs=.01)  # bbox x-mid = 100 = cx
    assert legacy.floor_contact_world[0] < -.05


def test_bad_instances_are_skipped_with_a_warning():
    good = np.zeros((100, 120), np.uint8); good[60:90, 40:80] = 1
    sky = np.zeros((100, 120), np.uint8); sky[5:25, 10:30] = 1  # above the horizon: floor ray behind the camera
    instances = [MaskInstance("sky", "object", 1., sky), MaskInstance("good", "object", 1., good)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        meshes = build_meshes(instances, CameraIntrinsics(100, 100, 60, 50), *camera_pose(1.4), [0, 1, 0], 0)
    assert [mesh.instance_id for mesh in meshes] == ["good"]
    assert any("skipping instance sky" in str(w.message) for w in caught)


def test_octile_heuristic_is_exact_on_open_grid():
    assert octile_distance((0, 0), (3, 5)) == pytest.approx(3 * math.sqrt(2) + 2)
    grid = np.zeros((30, 30), bool)
    path = astar_path(grid, (0, 0), (20, 7))
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
    assert length == pytest.approx(octile_distance((0, 0), (20, 7)))
