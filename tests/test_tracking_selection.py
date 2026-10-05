import numpy as np

from silhouette_ar.geometry import CameraIntrinsics, build_meshes, camera_pose
from silhouette_ar.segmentation import MaskInstance
from silhouette_ar.selection import select_by_keyword, select_by_pixel
from silhouette_ar.tracking import SilhouetteTracker

INTRINSICS = CameraIntrinsics(500, 500, 320, 240)
POSE = camera_pose(.6, 25)


def frame(*boxes):
    instances = []
    for index, (label, x0, y0, x1, y1) in enumerate(boxes):
        mask = np.zeros((480, 640), np.uint8)
        mask[y0:y1, x0:x1] = 1
        instances.append(MaskInstance(f"raw-{index}", label, 1., mask))
    return build_meshes(instances, INTRINSICS, *POSE, [0, 1, 0], 0)


def test_tracker_matches_by_class_and_distance():
    tracker = SilhouetteTracker(max_distance_m=.2, max_missed=1)
    first = tracker.update(frame(("zebra", 100, 300, 180, 380), ("panda", 400, 300, 470, 370)))
    assert sorted(m.instance_id for m in first) == ["panda-1", "zebra-1"]
    moved = tracker.update(frame(("zebra", 115, 300, 195, 380), ("panda", 400, 300, 470, 370)))
    assert sorted(m.instance_id for m in moved) == ["panda-1", "zebra-1"]
    # Same place, different class -> a new object; a far jump -> a new object.
    relabelled = tracker.update(frame(("giraffe", 115, 300, 195, 380), ("zebra", 560, 140, 630, 200)))
    assert sorted(m.instance_id for m in relabelled) == ["giraffe-1", "zebra-2"]
    tracker.update([])
    tracker.update([])
    assert tracker.get("panda-1") is None


def test_pixel_ray_hits_the_nearest_mesh():
    meshes = frame(("zebra", 100, 300, 180, 380), ("panda", 140, 250, 260, 330))
    near, _ = select_by_pixel(meshes, (160, 320), INTRINSICS, *POSE)  # inside both silhouettes
    assert near.label == "zebra"  # lower in the image = closer on the floor = hit first
    only, _ = select_by_pixel(meshes, (220, 300), INTRINSICS, *POSE)
    assert only.label == "panda"
    nothing, _ = select_by_pixel(meshes, (600, 30), INTRINSICS, *POSE)
    assert nothing is None


def test_keyword_selection_uses_class_names_ids_and_synonyms():
    meshes = frame(("teddy bear", 100, 300, 180, 380), ("teddy bear", 400, 200, 450, 260), ("panda", 250, 300, 300, 360))
    tracker = SilhouetteTracker()
    tracker.update(meshes)
    origin = POSE[0]
    assert select_by_keyword(meshes, "Point at the panda", origin).label == "panda"
    nearest = select_by_keyword(meshes, "go to the teddy", origin)
    assert nearest.instance_id == "teddy-bear-1"
    assert select_by_keyword(meshes, "teddy bear 2", origin).instance_id == "teddy-bear-2"
    assert select_by_keyword(meshes, "the giraffe", origin) is None
