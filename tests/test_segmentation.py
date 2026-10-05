import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from silhouette_ar import segdata
from silhouette_ar.segmentation import class_instances, yolo_result_instances

torch = pytest.importorskip("torch")

from silhouette_ar.segtrain import main as seg_main  # noqa: E402
from silhouette_ar.unet import PAPER_CONFIG, build_unet, count_parameters  # noqa: E402

PAPER_PARAMETERS = 2_160_194  # 16 base filters, 4 poolings, 3x3 up-conv, BN, background + 1 class


def test_compressed_unet_parameter_count_and_shape():
    model = build_unet(2, **PAPER_CONFIG)
    assert count_parameters(model) == PAPER_PARAMETERS
    assert 2.0e6 < PAPER_PARAMETERS < 2.5e6  # paper: about 2.4M; the 32-filter reference is 7.76M
    reference = build_unet(2, base_filters=32, depth=4, up_kernel=2, batch_norm=False)
    assert count_parameters(reference) == 7_760_130
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 192, 192)).shape == (1, 2, 192, 192)
    assert build_unet(4, **PAPER_CONFIG)(torch.zeros(2, 3, 192, 192)).shape == (2, 4, 192, 192)


def test_class_instances_keep_class_names():
    class_map = np.zeros((60, 80), np.uint8)
    class_map[5:25, 5:25] = 1
    class_map[30:55, 40:70] = 2
    class_map[5:25, 50:70] = 1
    class_map[40:42, 5:7] = 255
    instances = class_instances(class_map, ["background", "zebra", "panda"], min_area=50)
    assert sorted(i.label for i in instances) == ["panda", "zebra", "zebra"]


def test_yolo_polygons_are_used_in_original_pixels():
    """Audit regression: masks.data is letterboxed (384x640 for 720x1280); masks.xy is not."""
    polygon = np.array([[100, 300], [300, 300], [300, 700], [100, 700]], np.float32)
    result = SimpleNamespace(masks=SimpleNamespace(xy=[polygon], data=np.zeros((1, 384, 640))),
                             boxes=SimpleNamespace(cls=[15], conf=[.9], id=None), names={15: "cat"})
    instance = yolo_result_instances(result, (720, 1280))[0]
    ys, xs = np.nonzero(instance.mask)
    assert instance.mask.shape == (720, 1280) and instance.label == "cat"
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (300, 700, 100, 300)


def tiny_coco(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "val2017"
    images.mkdir()
    records, annotations = [], []
    rng = np.random.default_rng(0)
    for index in range(6):
        image = rng.integers(0, 255, (64, 80, 3), np.uint8)
        cv2.circle(image, (40, 32), 15, (20, 200, 40), -1)
        name = f"{index:012d}.jpg"
        cv2.imwrite(str(images / name), image)
        records.append({"id": index, "file_name": name, "height": 64, "width": 80, "license": 1, "coco_url": "", "flickr_url": ""})
        annotations.append({"id": 10 + index, "image_id": index, "category_id": 88, "iscrowd": 0, "area": 700,
                            "segmentation": [[25, 17, 55, 17, 55, 47, 25, 47]]})
    # Crowd RLE (uncompressed) on image 0 and an unrelated category on image 1.
    crowd = np.zeros((64, 80), np.uint8); crowd[2:6, 2:10] = 1
    flat = crowd.T.flatten()
    runs, value, count = [], 0, 0
    for pixel in flat:
        if pixel != value:
            runs.append(count); value, count = pixel, 0
        count += 1
    runs.append(count)
    annotations.append({"id": 99, "image_id": 0, "category_id": 88, "iscrowd": 1, "area": 32, "segmentation": {"counts": runs, "size": [64, 80]}})
    annotations.append({"id": 98, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": 50, "segmentation": [[0, 0, 5, 0, 5, 5]]})
    file = tmp_path / "instances.json"
    file.write_text(json.dumps({"images": records, "annotations": annotations, "licenses": [{"id": 1, "name": "CC BY", "url": "x"}],
                                "categories": [{"id": 1, "name": "person"}, {"id": 88, "name": "teddy bear"}]}), encoding="utf-8")
    return file, images


def test_compressed_rle_decoder_matches_known_encoding():
    # pycocotools' string form of runs [4, 1, 2, 1, 1] (later runs delta-coded, negative delta -> "O").
    mask = segdata.decode_rle("4120O", 3, 3)
    assert mask.sum() == 2 and mask[1, 1] == 1 and mask[1, 2] == 1
    assert segdata.decode_rle([4, 1, 4], 3, 3)[1, 1] == 1


def test_coco_route_prepares_trains_exports_and_segments(tmp_path: Path):
    annotations, images = tiny_coco(tmp_path)
    prepared = tmp_path / "prepared"
    summary = segdata.prepare_coco(annotations, images, prepared, ["teddy bear"], val_fraction=.34)
    assert summary["classes"] == ["background", "teddy bear"] and summary["train"] == 4 and summary["val"] == 2
    mask = cv2.imread(str(prepared / "masks" / "000000000000.png"), cv2.IMREAD_UNCHANGED)
    assert mask[32, 40] == 1 and mask[3, 5] == 255 and mask[60, 2] == 0
    provenance = json.loads((prepared / "provenance.json").read_text())
    assert provenance["items"][0]["license"] == "CC BY"

    out = tmp_path / "model"
    seg_main(["train", "--data", str(prepared), "--out", str(out), "--epochs", "2", "--batch-size", "2", "--input-size", "64"])
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["parameters"] == PAPER_PARAMETERS
    assert [round(r["lr"], 7) for r in metrics["history"]] == [1e-4, 1e-4]
    assert 0 <= metrics["history"][-1]["val_mean_iou"] <= 1

    from silhouette_ar.segmentation import UNetSegmenter, load_segmenter
    segmenter = load_segmenter(out / "model.pt")
    assert isinstance(segmenter, UNetSegmenter) and segmenter.classes == ["background", "teddy bear"]
    class_map, confidence = segmenter.class_map(cv2.imread(str(images / "000000000002.jpg")))
    assert class_map.shape == (64, 80) and confidence.shape == (64, 80)

    pytest.importorskip("onnx")
    from silhouette_ar.segtrain import export_onnx
    meta = export_onnx(out / "model.pt", out / "model.onnx")
    assert meta["classes"] == ["background", "teddy bear"] and (out / "model.json").is_file()
    if "onnxruntime not installed" not in meta["check"]:
        assert float(meta["check"].rsplit("=", 1)[1]) < 1e-3
        onnx_map, _ = UNetSegmenter(out / "model.onnx").class_map(cv2.imread(str(images / "000000000002.jpg")))
        assert (onnx_map == class_map).mean() > .99


def test_learning_rate_is_divided_by_ten_every_two_epochs(tmp_path: Path):
    annotations, images = tiny_coco(tmp_path)
    prepared = tmp_path / "prepared"
    segdata.prepare_coco(annotations, images, prepared, ["teddy bear"])
    out = tmp_path / "model"
    seg_main(["train", "--data", str(prepared), "--out", str(out), "--epochs", "5", "--batch-size", "4", "--input-size", "32",
              "--max-steps", "1", "--base-filters", "4", "--depth", "2"])
    lrs = [r["lr"] for r in json.loads((out / "metrics.json").read_text())["history"]]
    assert np.allclose(lrs, [1e-4, 1e-4, 1e-5, 1e-5, 1e-6])


def test_self_labelled_binary_and_indexed_masks(tmp_path: Path):
    images, masks = tmp_path / "frames", tmp_path / "labels"
    images.mkdir(); masks.mkdir()
    for name, values in (("a", (0, 255)), ("b", (0, 2))):
        cv2.imwrite(str(images / f"{name}.jpg"), np.full((20, 30, 3), 90, np.uint8))
        mask = np.full((20, 30), values[0], np.uint8); mask[5:15, 5:20] = values[1]
        cv2.imwrite(str(masks / f"{name}.png"), mask)
    cv2.imwrite(str(images / "unlabelled.jpg"), np.zeros((20, 30, 3), np.uint8))
    summary = segdata.prepare_masks(images, masks, tmp_path / "out", ["zebra", "panda"], class_index=1)
    assert summary["train"] + summary["val"] == 2
    assert cv2.imread(str(tmp_path / "out/masks/a.png"), cv2.IMREAD_UNCHANGED)[10, 10] == 1
    assert cv2.imread(str(tmp_path / "out/masks/b.png"), cv2.IMREAD_UNCHANGED)[10, 10] == 2


def test_synthetic_renders_composite_over_textures_with_exact_masks(tmp_path: Path):
    renders = tmp_path / "renders"
    renders.mkdir()
    rgba = np.zeros((40, 50, 4), np.uint8); rgba[10:30, 10:25] = (30, 60, 200, 255)
    cv2.imwrite(str(renders / "r0.png"), rgba)
    visible = np.zeros((40, 50, 4), np.uint8); visible[10:30, 10:25, 3] = 255
    cv2.imwrite(str(renders / "r0_obj0.png"), visible)
    (renders / "r0.json").write_text(json.dumps({"image": "r0.png", "objects": [{"class": "toy-bear", "mask": "r0_obj0.png"}]}))
    (renders / "metadata.json").write_text(json.dumps({"classes": ["toy-bear"], "assets": [{"name": "procedural toy-bear", "license": "CC0-1.0"}]}))
    summary = segdata.prepare_synthetic(renders, tmp_path / "prepared", val_fraction=0)
    mask = cv2.imread(str(tmp_path / "prepared/masks/r0.png"), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(tmp_path / "prepared/images/r0.png"))
    assert summary["classes"] == ["background", "toy-bear"] and mask[20, 15] == 1 and mask[2, 2] == 0
    assert tuple(image[20, 15]) == (30, 60, 200)


def test_paper_augmentations_keep_image_and_mask_aligned():
    rng = np.random.default_rng(3)
    image = np.zeros((96, 96, 3), np.uint8)
    mask = np.zeros((96, 96), np.uint8)
    cv2.rectangle(mask, (30, 40), (60, 70), 1, -1)
    image[mask > 0] = (250, 10, 10)
    bank = segdata.TextureBank(None, seed=1)
    for _ in range(12):
        out_image, out_mask = segdata.augment(image, mask, rng, bank, 1, 1, 1, 1)
        assert out_image.shape == image.shape and out_mask.shape == mask.shape
        assert set(np.unique(out_mask)) <= {0, 1}
        inner = cv2.erode(out_mask, np.ones((3, 3), np.uint8)) > 0
        if inner.any():
            assert (out_image[inner][:, 0] > 200).mean() > .95  # object pixels stay under the mask
    replaced = segdata.replace_background(image, mask, np.full_like(image, 77))
    assert (replaced[mask == 0] == 77).all() and (replaced[mask > 0] == image[mask > 0]).all()
    zoom_image, zoom_mask = segdata.random_zoom(image, mask, np.random.default_rng(0), (.25, .25))
    assert abs(zoom_mask.mean() - .25) < .03
