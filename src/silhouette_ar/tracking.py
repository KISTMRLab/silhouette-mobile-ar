"""Silhouette-mesh update loop (paper Section 3.3, Figure 5).

Each frame's meshes are matched to existing ones by class and a small floor
distance (``match_existing``). A match keeps the existing identifier and takes
the new geometry, so a moving object keeps its id and can be followed; an
unmatched mesh is placed as a new object. Objects that are not seen for
``max_missed`` updates are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import SilhouetteMesh, match_existing


@dataclass
class Track:
    mesh: SilhouetteMesh
    missed: int = 0
    updates: int = 1


@dataclass
class SilhouetteTracker:
    max_distance_m: float = .35
    max_missed: int = 2
    tracks: dict[str, Track] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def _new_id(self, label: str) -> str:
        self.counters[label] = self.counters.get(label, 0) + 1
        return f"{label.replace(' ', '-')}-{self.counters[label]}"

    def update(self, meshes: list[SilhouetteMesh]) -> list[SilhouetteMesh]:
        """Assign persistent ids to ``meshes`` (in place) and return the visible set."""
        unmatched = {id(track.mesh): track for track in self.tracks.values()}
        seen: set[str] = set()
        ordered = sorted(meshes, key=lambda mesh: min((float(((t.mesh.floor_contact_world - mesh.floor_contact_world) ** 2).sum())
                                                       for t in unmatched.values() if t.mesh.label == mesh.label), default=float("inf")))
        for mesh in ordered:
            previous = match_existing(mesh, [track.mesh for track in unmatched.values()], self.max_distance_m)
            if previous is not None:
                track = unmatched.pop(id(previous))
                mesh.instance_id = previous.instance_id
                track.mesh, track.missed = mesh, 0
                track.updates += 1
            else:
                mesh.instance_id = self._new_id(mesh.label)
                self.tracks[mesh.instance_id] = Track(mesh)
            seen.add(mesh.instance_id)
        for key in list(self.tracks):
            if key not in seen:
                self.tracks[key].missed += 1
                if self.tracks[key].missed > self.max_missed:
                    del self.tracks[key]
        return meshes

    def get(self, instance_id: str) -> SilhouetteMesh | None:
        track = self.tracks.get(instance_id)
        return track.mesh if track else None

    def visible(self) -> list[SilhouetteMesh]:
        return [track.mesh for track in self.tracks.values() if track.missed == 0]

    def reset(self) -> None:
        self.tracks.clear()
        self.counters.clear()
