import cv2
import numpy as np

from silhouette_ar.geometry import CameraIntrinsics, build_meshes
from silhouette_ar.interaction import astar_path, obstacle_grid, occlusion_composite
from silhouette_ar.segmentation import connected_instances


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
    assert set(mesh.interaction_targets()) == {"point", "approach", "touch", "ride"}


def test_mask_occludes_virtual_layer_and_blocks_path():
    mask = rectangle_mask()
    camera = np.zeros((100, 120, 3), np.uint8)
    virtual = np.zeros((100, 120, 4), np.uint8); virtual[:, :, 2] = 255; virtual[:, :, 3] = 255
    composited = occlusion_composite(camera, virtual, mask, virtual_in_front=False)
    assert composited[60, 60].sum() == 0 and composited[10, 10, 2] == 255
    grid = obstacle_grid(mask.shape, [mask], clearance_px=1)
    path = astar_path(grid, (5, 60), (110, 60))
    assert path and all(not grid[y, x] for x, y in path)

