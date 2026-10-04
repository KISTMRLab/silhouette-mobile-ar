from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class SilhouetteMesh:
    instance_id: str
    label: str
    contour_px: np.ndarray
    vertices_world: np.ndarray
    triangles: np.ndarray
    floor_contact_world: np.ndarray
    mask: np.ndarray

    @property
    def floor_footprint_world(self) -> np.ndarray:
        """Vertical projection of the ordered silhouette boundary onto world XZ."""
        return self.vertices_world[:, (0, 2)]

    def interaction_targets(self) -> dict[str, list[float]]:
        top = self.vertices_world[int(np.argmin(self.contour_px[:, 1]))]
        center = self.vertices_world.mean(axis=0)
        approach = self.floor_contact_world.copy()
        return {"point": center.tolist(), "approach": approach.tolist(), "touch": top.tolist(), "ride": top.tolist()}


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


def project_contour(
    contour_px: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_origin: np.ndarray,
    camera_to_world: np.ndarray,
    floor_normal: np.ndarray,
    floor_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    bottom_y = contour_px[:, 1].max()
    candidates = contour_px[np.abs(contour_px[:, 1] - bottom_y) < 2.0]
    bottom = np.array([candidates[:, 0].mean(), bottom_y])
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


def build_meshes(instances: list[MaskInstance], intrinsics: CameraIntrinsics, camera_origin: np.ndarray, camera_to_world: np.ndarray, floor_normal: np.ndarray, floor_offset: float) -> list[SilhouetteMesh]:
    meshes: list[SilhouetteMesh] = []
    for instance in instances:
        contour = contour_from_mask(instance.mask)
        vertices, contact = project_contour(contour, intrinsics, camera_origin, camera_to_world, floor_normal, floor_offset)
        meshes.append(SilhouetteMesh(instance.instance_id, instance.label, contour, vertices, triangulate_polygon(contour), contact, instance.mask))
    return meshes


def match_existing(new_mesh: SilhouetteMesh, existing: list[SilhouetteMesh], max_distance_m: float = .35) -> SilhouetteMesh | None:
    candidates = [item for item in existing if item.label == new_mesh.label]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: np.linalg.norm(item.floor_contact_world - new_mesh.floor_contact_world))
    return nearest if np.linalg.norm(nearest.floor_contact_world - new_mesh.floor_contact_world) <= max_distance_m else None
