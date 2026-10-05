"""Target selection (paper Section 5.1): a camera ray through the gaze/tap pixel
hits a silhouette mesh, or the spoken/typed object class is used as a keyword."""
from __future__ import annotations

import re

import numpy as np

from .geometry import CameraIntrinsics, SilhouetteMesh

SYNONYMS = {"teddy": "teddy bear", "teddybear": "teddy bear", "doll": None, "toy": None}


def ray_triangle_distances(origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Moller-Trumbore distances along the ray (inf where a triangle is missed)."""
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    edge1, edge2 = b - a, c - a
    p = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, p)
    valid = np.abs(det) > 1e-12
    inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
    t_vec = origin - a
    u = np.einsum("ij,ij->i", t_vec, p) * inv
    q = np.cross(t_vec, edge1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", edge2, q) * inv
    hit = valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    return np.where(hit, t, np.inf)


def select_by_ray(meshes: list[SilhouetteMesh], origin: np.ndarray, direction: np.ndarray) -> tuple[SilhouetteMesh | None, float]:
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    best, best_t = None, np.inf
    for mesh in meshes:
        distances = ray_triangle_distances(np.asarray(origin, float), direction, mesh.vertices_world, mesh.triangles)
        t = float(distances.min()) if len(distances) else np.inf
        if t < best_t:
            best, best_t = mesh, t
    return best, best_t


def select_by_pixel(meshes: list[SilhouetteMesh], pixel, intrinsics: CameraIntrinsics, camera_origin: np.ndarray,
                    camera_to_world: np.ndarray) -> tuple[SilhouetteMesh | None, float]:
    """Ray-cast from the camera through an image pixel (screen centre = gaze)."""
    direction = camera_to_world @ intrinsics.ray(np.asarray(pixel, float))
    return select_by_ray(meshes, camera_origin, direction)


def _words(text: str) -> list[str]:
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in re.findall(r"[a-z0-9]+", text.casefold())]


def select_by_keyword(meshes: list[SilhouetteMesh], text: str, camera_origin: np.ndarray | None = None) -> SilhouetteMesh | None:
    """Match an object class (or an instance id such as ``bear-2``) in the utterance.

    Several objects of the class: an ordinal number in the utterance picks that
    id, otherwise the object nearest the camera is chosen.
    """
    words = _words(text)
    joined = " ".join(words)
    for mesh in meshes:
        if mesh.instance_id.casefold() in text.casefold():
            return mesh
    expanded = joined
    for word, target in SYNONYMS.items():
        if target and re.search(rf"\b{word}\b", joined):
            expanded += " " + " ".join(_words(target))
    matches = [mesh for mesh in meshes if " ".join(_words(mesh.label)) and re.search(rf"\b{re.escape(' '.join(_words(mesh.label)))}\b", expanded)]
    if not matches:
        return None
    numbers = [w for w in words if w.isdigit()]
    for number in numbers:
        for mesh in matches:
            if mesh.instance_id.endswith(f"-{number}"):
                return mesh
    origin = np.zeros(3) if camera_origin is None else np.asarray(camera_origin, float)
    return min(matches, key=lambda mesh: float(np.linalg.norm(mesh.floor_contact_world - origin)))
