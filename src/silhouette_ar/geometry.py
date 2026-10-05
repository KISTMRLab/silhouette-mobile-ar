from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings

import numpy as np

from .segmentation import MaskInstance


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def ray(self, pixel: np.ndarray) -> np.ndarray:
        direction = np.array([(pixel[0] - self.cx) / self.fx, (pixel[1] - self.cy) / self.fy, 1.0], dtype=float)
        return direction / np.linalg.norm(direction)


def camera_pose(height: float, pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Camera origin and camera-to-world matrix for the documented convention.

    Camera pixels use x right, y down, z forward. The world is y up with the floor
    at y=0; at zero yaw the camera looks along world +z. ``pitch_deg`` tilts the
    view down toward the floor; ``yaw_deg`` turns it about the vertical axis.
    """
    pitch, yaw = math.radians(pitch_deg), math.radians(yaw_deg)
    c, s = math.cos(pitch), math.sin(pitch)
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, -c, -s], [0.0, -s, c]])
    cy, sy = math.cos(yaw), math.sin(yaw)
    turn = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return np.array([0.0, float(height), 0.0]), turn @ tilt


def pitch_from_floor_tap(pixel_y: float, intrinsics: CameraIntrinsics, camera_height: float, floor_distance: float) -> float:
    """One-tap floor calibration: pitch (degrees) that puts the tapped pixel on the
    floor at ``floor_distance`` metres (horizontal) in front of the camera."""
    if camera_height <= 0 or floor_distance <= 0:
        raise ValueError("camera height and tap distance must be positive")
    depression = math.atan2(camera_height, floor_distance)
    below_axis = math.atan2(pixel_y - intrinsics.cy, intrinsics.fy)
    return math.degrees(depression - below_axis)


def _unit_plane(normal: np.ndarray, offset: float) -> tuple[np.ndarray, float]:
    normal = np.asarray(normal, float)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError("floor normal must be non-zero")
    return normal / length, float(offset) / length


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull (counter-clockwise)."""
    pts = sorted(set(map(tuple, np.round(points, 9))))
    if len(pts) < 3:
        return np.asarray(pts, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], float)


@dataclass
class SilhouetteMesh:
    instance_id: str
    label: str
    contour_px: np.ndarray
    vertices_world: np.ndarray
    triangles: np.ndarray
    floor_contact_world: np.ndarray
    mask: np.ndarray
    camera_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    floor_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    floor_offset: float = 0.0
    confidence: float = 1.0
    min_half_depth_m: float = .04
    depth_ratio: float = .15

    def heights(self) -> np.ndarray:
        normal, offset = _unit_plane(self.floor_normal, self.floor_offset)
        return self.vertices_world @ normal + offset

    @property
    def floor_footprint_3d(self) -> np.ndarray:
        """Orthographic projection of the mesh onto the floor plane (convex hull, 3D points).

        The equal-distance mesh tilts with the device, so its projection covers the
        floor area the object occupies (paper Section 3.2, Step 4; Section 5.2). A
        camera looking level yields an almost flat strip, so the hull is padded
        along the horizontal viewing direction to at least
        ``max(min_half_depth_m, depth_ratio * width)`` on each side.
        """
        normal, offset = _unit_plane(self.floor_normal, self.floor_offset)
        projected = self.vertices_world - np.outer(self.vertices_world @ normal + offset, normal)
        helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < .9 else np.array([0.0, 0.0, 1.0])
        axis_u = np.cross(normal, helper)
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        view = self.floor_contact_world - self.camera_origin
        view = view - np.dot(view, normal) * normal
        if np.linalg.norm(view) < 1e-9:
            view = axis_v.copy()
        view /= np.linalg.norm(view)
        across = np.cross(normal, view)
        along = projected @ view
        width = float(np.ptp(projected @ across)) if len(projected) else 0.0
        half_depth = max(self.min_half_depth_m, self.depth_ratio * width)
        pad = max(0.0, half_depth - float(np.ptp(along)) / 2)
        points = np.vstack([projected + pad * view, projected - pad * view]) if pad > 0 else projected
        origin = points.mean(axis=0)
        uv = np.column_stack(((points - origin) @ axis_u, (points - origin) @ axis_v))
        hull = _convex_hull_2d(uv)
        return origin + np.outer(hull[:, 0], axis_u) + np.outer(hull[:, 1], axis_v)

    @property
    def floor_footprint_world(self) -> np.ndarray:
        """Floor footprint polygon in world x/z (the navigation grid's plane)."""
        return self.floor_footprint_3d[:, [0, 2]]

    def body_top(self, width_ratio: float = .5) -> np.ndarray:
        """Centre of the highest mask row that is at least ``width_ratio`` of the widest row.

        Thin protrusions (ears, antennae) are skipped, so riding/petting lands on
        the body rather than on a 2 cm ear tip. Falls back to the highest vertex.
        """
        top = self.vertices_world[int(np.argmax(self.heights()))]
        rows = (np.asarray(self.mask) > 0).sum(axis=1)
        if not rows.any():
            return top
        row = float(np.argmax(rows >= width_ratio * rows.max())) + .5
        crossings = []
        count = len(self.contour_px)
        for i in range(count):
            j = (i + 1) % count
            a, b = self.contour_px[i], self.contour_px[j]
            if a[1] == b[1] or (a[1] - row) * (b[1] - row) > 0:
                continue
            t = (row - a[1]) / (b[1] - a[1])
            crossings.append((a[0] + t * (b[0] - a[0]), self.vertices_world[i] + t * (self.vertices_world[j] - self.vertices_world[i])))
        if len(crossings) < 2:
            return top
        left, right = min(crossings, key=lambda c: c[0]), max(crossings, key=lambda c: c[0])
        return (left[1] + right[1]) / 2

    def interaction_targets(self, clearance_m: float = .06, margin_m: float = .04) -> dict[str, list[float]]:
        """Targets for point, approach/stand, touch, pet, push and ride (paper Section 5).

        ``touch`` is the highest silhouette vertex; ``pet`` and ``ride`` use the
        body top (see ``body_top``); ``stand`` is the standing position outside the
        dilated hole, toward the camera (Section 5.3).
        """
        from .interaction import stand_point

        normal, offset = _unit_plane(self.floor_normal, self.floor_offset)
        heights = self.heights()
        top = self.vertices_world[int(np.argmax(heights))]
        body = self.body_top()
        height = float(heights.max())
        center = self.vertices_world.mean(axis=0)
        band = self.vertices_world[(heights >= .3 * height) & (heights <= .6 * height)]
        push = band.mean(axis=0) if len(band) else center
        footprint = self.floor_footprint_3d
        stand_xz = stand_point(footprint[:, [0, 2]], self.camera_origin[[0, 2]], clearance_m + margin_m)
        stand = np.array([stand_xz[0], 0.0, stand_xz[1]])
        stand[1] = -(normal[0] * stand[0] + normal[2] * stand[2] + offset) / normal[1] if abs(normal[1]) > 1e-6 else 0.0
        return {"point": center.tolist(), "approach": stand.tolist(), "stand": stand.tolist(), "touch": top.tolist(),
                "pet": body.tolist(), "push": push.tolist(), "ride": body.tolist(), "top_height": height,
                "footprint_center": footprint.mean(axis=0).tolist()}


def contour_from_mask(mask: np.ndarray, epsilon_ratio: float = .008) -> np.ndarray:
    import cv2
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Mask contains no contour")
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :].astype(float)
    if len(simplified) < 3:
        raise ValueError("Contour has fewer than three vertices")
    return simplified


def _signed_area(points: np.ndarray) -> float:
    return .5 * sum(points[i, 0] * points[(i + 1) % len(points), 1] - points[(i + 1) % len(points), 0] * points[i, 1] for i in range(len(points)))


def triangulate_polygon(points: np.ndarray) -> np.ndarray:
    """Ear-clipping triangulation for a simple contour polygon."""
    order = list(range(len(points)))
    if _signed_area(points) < 0:
        order.reverse()
    triangles: list[list[int]] = []

    def cross(a, b, c):
        ab, ac = b - a, c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= -1e-8 and cross(b, c, p) >= -1e-8 and cross(c, a, p) >= -1e-8

    guard = 0
    while len(order) > 3 and guard < len(points) ** 2:
        clipped = False
        for cursor in range(len(order)):
            previous, current, following = order[cursor - 1], order[cursor], order[(cursor + 1) % len(order)]
            a, b, c = points[previous], points[current], points[following]
            if cross(a, b, c) <= 1e-8:
                continue
            if any(inside(points[index], a, b, c) for index in order if index not in {previous, current, following}):
                continue
            triangles.append([previous, current, following])
            order.pop(cursor)
            clipped = True
            break
        if not clipped:
            raise ValueError("Contour is self-intersecting or numerically degenerate")
        guard += 1
    triangles.append(order)
    return np.asarray(triangles, dtype=np.int32)


def bottom_reference_pixel(contour_px: np.ndarray, mode: str = "bbox") -> np.ndarray:
    """Pixel whose floor ray fixes the mesh distance.

    ``bbox`` (default, paper Section 3.2 Step 2) uses the bounding box's
    bottom-midpoint. ``contour`` keeps the earlier behaviour: the mean x of the
    contour points within 2 px of the lowest row.
    """
    bottom_y = contour_px[:, 1].max()
    if mode == "bbox":
        return np.array([(contour_px[:, 0].min() + contour_px[:, 0].max()) / 2, bottom_y])
    if mode == "contour":
        candidates = contour_px[np.abs(contour_px[:, 1] - bottom_y) < 2.0]
        return np.array([candidates[:, 0].mean(), bottom_y])
    raise ValueError(f"Unknown contact mode: {mode}")


def project_contour(
    contour_px: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_origin: np.ndarray,
    camera_to_world: np.ndarray,
    floor_normal: np.ndarray,
    floor_offset: float,
    contact_mode: str = "bbox",
) -> tuple[np.ndarray, np.ndarray]:
    bottom = bottom_reference_pixel(contour_px, contact_mode)
    bottom_ray = camera_to_world @ intrinsics.ray(bottom)
    denominator = float(np.dot(floor_normal, bottom_ray))
    if abs(denominator) < 1e-8:
        raise ValueError("Bottom ray is parallel to the floor plane")
    distance = -(float(np.dot(floor_normal, camera_origin)) + floor_offset) / denominator
    if distance <= 0:
        raise ValueError("Floor intersection is behind the camera; check pose and plane convention")
    contact = camera_origin + distance * bottom_ray
    vertices = np.stack([camera_origin + distance * (camera_to_world @ intrinsics.ray(pixel)) for pixel in contour_px])
    return vertices, contact


def build_meshes(instances: list[MaskInstance], intrinsics: CameraIntrinsics, camera_origin: np.ndarray, camera_to_world: np.ndarray,
                 floor_normal: np.ndarray, floor_offset: float, contact_mode: str = "bbox") -> list[SilhouetteMesh]:
    """Build one equal-distance silhouette mesh per instance.

    Instances whose bottom ray misses the floor (behind the camera, parallel to
    it) or whose contour cannot be triangulated are skipped with a warning so one
    bad region does not abort the frame.
    """
    meshes: list[SilhouetteMesh] = []
    for instance in instances:
        try:
            contour = contour_from_mask(instance.mask)
            vertices, contact = project_contour(contour, intrinsics, camera_origin, camera_to_world, floor_normal, floor_offset, contact_mode)
            triangles = triangulate_polygon(contour)
        except ValueError as exc:
            warnings.warn(f"skipping instance {instance.instance_id} ({instance.label}): {exc}", RuntimeWarning, stacklevel=2)
            continue
        meshes.append(SilhouetteMesh(instance.instance_id, instance.label, contour, vertices, triangles, contact, instance.mask,
                                     np.asarray(camera_origin, float).copy(), np.asarray(floor_normal, float).copy(), float(floor_offset),
                                     float(instance.confidence)))
    return meshes


def match_existing(new_mesh: SilhouetteMesh, existing: list[SilhouetteMesh], max_distance_m: float = .35) -> SilhouetteMesh | None:
    candidates = [item for item in existing if item.label == new_mesh.label]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: np.linalg.norm(item.floor_contact_world - new_mesh.floor_contact_world))
    return nearest if np.linalg.norm(nearest.floor_contact_world - new_mesh.floor_contact_world) <= max_distance_m else None
