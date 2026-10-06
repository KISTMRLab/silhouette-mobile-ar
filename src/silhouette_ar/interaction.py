from __future__ import annotations

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)


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
    """Rasterize mesh floor projections into a navigation obstacle grid (row = z, column = x)."""
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


def octile_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Admissible, consistent heuristic for 8-connected grids with diagonal cost sqrt(2)."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def astar_path(obstacles: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = obstacles.shape
    for point in (start, goal):
        if not (0 <= point[0] < width and 0 <= point[1] < height) or obstacles[point[1], point[0]]:
            return []
    queue = [(octile_distance(start, goal), 0.0, start)]
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in previous:
                current = previous[current]
                path.append(current)
            return path[::-1]
        closed.add(current)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            nxt = current[0] + dx, current[1] + dy
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height) or obstacles[nxt[1], nxt[0]]:
                continue
            if dx and dy and (obstacles[current[1], nxt[0]] or obstacles[nxt[1], current[0]]):
                continue  # no corner cutting through an obstacle
            new_cost = current_cost + (SQRT2 if dx and dy else 1.0)
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                previous[nxt] = current
                heapq.heappush(queue, (new_cost + octile_distance(nxt, goal), new_cost, nxt))
    return []


def stand_point(footprint_xz: np.ndarray, toward_xz: np.ndarray, clearance_m: float, step_m: float = .01,
                max_distance_m: float = 5.0, blocked=None) -> np.ndarray:
    """Standing position next to an object (paper Section 5.3).

    Starts at the centre of the un-walkable area and moves toward ``toward_xz``
    (the camera's floor position) until it is ``clearance_m`` outside the
    footprint, i.e. outside the dilated hole. ``blocked(xz)`` may reject points
    occupied by other objects; nearby directions are then tried.
    """
    import cv2
    polygon = np.asarray(footprint_xz, np.float32).reshape(-1, 1, 2)
    center = np.asarray(footprint_xz, float).mean(axis=0)
    direction = np.asarray(toward_xz, float) - center
    if np.linalg.norm(direction) < 1e-9:
        direction = np.array([0.0, -1.0])
    base = math.atan2(direction[1], direction[0])
    for offset in (0, 20, -20, 40, -40, 60, -60, 90, -90, 135, -135, 180):
        angle = base + math.radians(offset)
        unit = np.array([math.cos(angle), math.sin(angle)])
        distance = 0.0
        while distance <= max_distance_m:
            point = center + distance * unit
            outside = -cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True)
            if outside >= clearance_m:
                if blocked is None or not blocked(point):
                    return point
                break
            distance += step_m
    return center + clearance_m * direction / max(np.linalg.norm(direction), 1e-9)


def footprint_clearance(footprint_xz: np.ndarray, point_xz) -> float:
    """Signed floor distance from ``point_xz`` to a footprint polygon (positive outside)."""
    import cv2
    polygon = np.asarray(footprint_xz, np.float32).reshape(-1, 1, 2)
    return float(-cv2.pointPolygonTest(polygon, (float(point_xz[0]), float(point_xz[1])), True))


def ride_seat(footprint_xz: np.ndarray, body_top, inset_m: float = .03) -> np.ndarray:
    """Seat for riding: the silhouette's body top, kept ``inset_m`` inside the footprint.

    The equal-distance mesh leans away from a downward-pitched camera, so its body
    top projects near the far edge of the floor footprint. The seat keeps the top
    height but slides along the floor toward the footprint centre until it is
    ``inset_m`` inside (or reaches the centre), so the rider sits over the object
    even while the next silhouette update lags a moving object by a few cm.
    """
    footprint = np.asarray(footprint_xz, float)
    seat = np.asarray(body_top, float).copy()
    center = footprint.mean(axis=0)
    start = seat[[0, 2]].copy()
    if -footprint_clearance(footprint, start) >= inset_m:
        return seat
    best = center
    for k in np.linspace(0.0, 1.0, 41)[1:]:
        point = start + k * (center - start)
        if -footprint_clearance(footprint, point) >= inset_m:
            best = point
            break
    seat[0], seat[2] = best
    return seat


def follow_point(footprint_xz: np.ndarray, camera_xz, from_xz, standoff_m: float, toward_camera_deg: float = 30.0,
                 blocked=None) -> np.ndarray:
    """Stand-off position for following an object.

    The avatar stays beside the object as seen from the camera, on the side it
    already occupies, turned ``toward_camera_deg`` toward the camera so it is a
    little nearer than the object. The point is ``standoff_m`` outside the
    footprint; ``standoff_m`` should cover the navigation clearance plus the
    avatar's body radius, so neither the root nor the body enters the object's
    dilated hole. Standing straight toward the camera (``stand_point``) would put
    the avatar in front of the object in the image and hide it.
    """
    footprint = np.asarray(footprint_xz, float)
    center = footprint.mean(axis=0)
    view = center - np.asarray(camera_xz, float)
    if np.linalg.norm(view) < 1e-9:
        view = np.array([0.0, 1.0])
    view /= np.linalg.norm(view)
    lateral = np.array([view[1], -view[0]])
    offset = np.asarray(from_xz, float) - center if from_xz is not None else lateral
    if float(np.dot(offset, lateral)) < 0:
        lateral = -lateral
    angle = math.radians(toward_camera_deg)
    direction = math.cos(angle) * lateral - math.sin(angle) * view
    return stand_point(footprint, center + direction, standoff_m, blocked=blocked)


def nearest_free(grid: np.ndarray, cell: tuple[int, int], max_radius: int = 40) -> tuple[int, int] | None:
    height, width = grid.shape
    x, y = min(max(cell[0], 0), width - 1), min(max(cell[1], 0), height - 1)
    if not grid[y, x]:
        return x, y
    for radius in range(1, max_radius + 1):
        best = None
        for dy in range(-radius, radius + 1):
            for dx in (-radius, radius) if abs(dy) != radius else range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not grid[ny, nx]:
                    d = dx * dx + dy * dy
                    if best is None or d < best[0]:
                        best = (d, (nx, ny))
        if best:
            return best[1]
    return None


def line_of_sight(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    for t in np.linspace(0.0, 1.0, steps + 1):
        x = int(round(a[0] + (b[0] - a[0]) * t))
        y = int(round(a[1] + (b[1] - a[1]) * t))
        if grid[y, x]:
            return False
    return True


def smooth_path(grid: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop waypoints that are visible from an earlier waypoint (string pulling)."""
    if len(path) <= 2:
        return list(path)
    result = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        nxt = len(path) - 1
        while nxt > anchor + 1 and not line_of_sight(grid, path[anchor], path[nxt]):
            nxt -= 1
        result.append(path[nxt])
        anchor = nxt
    return result


def plan_on_floor(footprints_xz: list[np.ndarray], start_xz, goal_xz, cell_size_m: float = .025, clearance_m: float = .06,
                  margin_m: float = .6, smooth: bool = True) -> dict:
    """A* over the walkable floor: footprints are holes dilated by ``clearance_m``.

    Start/goal cells inside a hole snap to the nearest free cell. Returns world
    x/z waypoints plus the grid diagnostics used by the demo and tests.
    """
    start_xz, goal_xz = np.asarray(start_xz, float), np.asarray(goal_xz, float)
    points = [start_xz[None], goal_xz[None]] + [np.asarray(p, float) for p in footprints_xz]
    stacked = np.vstack(points)
    min_x, min_z = stacked.min(axis=0) - margin_m
    max_x, max_z = stacked.max(axis=0) + margin_m
    bounds = (float(min_x), float(min_z), float(max_x), float(max_z))
    grid = rasterize_world_footprints(list(footprints_xz), bounds, cell_size_m, clearance_m)

    def to_cell(p):
        return int(round((p[0] - min_x) / cell_size_m)), int(round((p[1] - min_z) / cell_size_m))

    start_cell = nearest_free(grid, to_cell(start_xz))
    goal_cell = nearest_free(grid, to_cell(goal_xz))
    if start_cell is None or goal_cell is None:
        return {"path_xz": [], "ok": False, "reason": "no free cell near start or goal", "bounds": bounds}
    cells = astar_path(grid, start_cell, goal_cell)
    if not cells:
        return {"path_xz": [], "ok": False, "reason": "goal unreachable", "bounds": bounds}
    if smooth:
        cells = smooth_path(grid, cells)
    path = [[min_x + x * cell_size_m, min_z + y * cell_size_m] for x, y in cells]
    return {"path_xz": path, "ok": True, "bounds": bounds, "cells": len(cells), "grid_shape": list(grid.shape),
            "start_snapped": start_cell != to_cell(start_xz), "goal_snapped": goal_cell != to_cell(goal_xz)}
