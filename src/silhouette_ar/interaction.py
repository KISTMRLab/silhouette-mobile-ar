from __future__ import annotations

import heapq

import numpy as np


def occlusion_composite(
    camera_bgr: np.ndarray,
    virtual_bgra: np.ndarray,
    object_mask: np.ndarray,
    virtual_in_front: bool = False,
    virtual_depth: np.ndarray | None = None,
    object_depth: np.ndarray | None = None,
) -> np.ndarray:
    """Composite a virtual layer; real-object mask hides it when the object is nearer."""
    if camera_bgr.shape[:2] != virtual_bgra.shape[:2] or camera_bgr.shape[:2] != object_mask.shape[:2]:
        raise ValueError("Camera, virtual layer, and mask dimensions must match")
    alpha = virtual_bgra[:, :, 3].astype(float) / 255.0
    if virtual_depth is not None or object_depth is not None:
        if virtual_depth is None or object_depth is None or virtual_depth.shape != object_mask.shape or object_depth.shape != object_mask.shape:
            raise ValueError("Both depth maps must match the object mask")
        visible = (object_mask == 0) | (virtual_depth <= object_depth)
        alpha = alpha * visible
    elif not virtual_in_front:
        alpha = alpha * (object_mask == 0)
    alpha = alpha[:, :, None]
    return np.clip(virtual_bgra[:, :, :3] * alpha + camera_bgr * (1 - alpha), 0, 255).astype(np.uint8)


def obstacle_grid(shape: tuple[int, int], masks: list[np.ndarray], clearance_px: int = 8) -> np.ndarray:
    import cv2
    obstacles = np.zeros(shape, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clearance_px * 2 + 1, clearance_px * 2 + 1))
    for mask in masks:
        obstacles |= cv2.dilate((mask > 0).astype(np.uint8), kernel)
    return obstacles.astype(bool)


def rasterize_world_footprints(
    footprints_xz: list[np.ndarray],
    bounds_xz: tuple[float, float, float, float],
    cell_size_m: float,
    clearance_m: float = .08,
) -> np.ndarray:
    """Rasterize mesh floor projections into a navigation obstacle grid."""
    import cv2
    min_x, min_z, max_x, max_z = bounds_xz
    width = int(np.ceil((max_x - min_x) / cell_size_m))
    height = int(np.ceil((max_z - min_z) / cell_size_m))
    if width <= 0 or height <= 0:
        raise ValueError("bounds must have positive area")
    grid = np.zeros((height, width), np.uint8)
    for polygon in footprints_xz:
        pixels = np.column_stack(((polygon[:, 0] - min_x) / cell_size_m, (polygon[:, 1] - min_z) / cell_size_m)).round().astype(np.int32)
        cv2.fillPoly(grid, [pixels], 1)
    clearance = max(0, int(np.ceil(clearance_m / cell_size_m)))
    if clearance:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clearance * 2 + 1, clearance * 2 + 1))
        grid = cv2.dilate(grid, kernel)
    return grid.astype(bool)


def astar_path(obstacles: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = obstacles.shape
    for point in (start, goal):
        if not (0 <= point[0] < width and 0 <= point[1] < height) or obstacles[point[1], point[0]]:
            return []
    queue = [(0.0, start)]
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            path = [current]
            while current in previous:
                current = previous[current]
                path.append(current)
            return path[::-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            nxt = current[0] + dx, current[1] + dy
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height) or obstacles[nxt[1], nxt[0]]:
                continue
            new_cost = cost[current] + (1.414 if dx and dy else 1.0)
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                previous[nxt] = current
                heuristic = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                heapq.heappush(queue, (new_cost + heuristic, nxt))
    return []
