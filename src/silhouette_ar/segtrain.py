"""Prepare data for, train, evaluate and export the compressed U-Net segmenter.

    silhouette-seg params
    silhouette-seg fetch-coco --out data/coco --categories "teddy bear" --limit 200
    silhouette-seg prepare-coco --annotations data/coco/annotations/instances_val2017.json \
        --images-dir data/coco/val2017 --categories "teddy bear" --out data/prepared/coco-teddy
    silhouette-seg prepare-masks --images frames/ --masks masks/ --classes doll --out data/prepared/own
    silhouette-seg prepare-synthetic --renders outputs/blender --textures data/dtd/images --out data/prepared/synthetic
    silhouette-seg train --data data/prepared/coco-teddy --textures data/dtd/images --out outputs/unet
    silhouette-seg evaluate --data data/prepared/coco-teddy --weights outputs/unet/model.pt
    silhouette-seg export-onnx --weights outputs/unet/model.pt --out outputs/unet/model.onnx
    silhouette-seg predict --weights outputs/unet/model.pt --image frame.jpg --out outputs/predict
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np

from . import segdata
from .unet import PAPER_CONFIG, build_unet, count_parameters, load_checkpoint, save_checkpoint


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Training needs PyTorch: pip install -e .[train]") from exc
    return torch


def make_batches(dataset: dict, split: str, size: int, batch_size: int, rng: np.random.Generator, textures=None, train: bool = True,
                 augment: bool = True):
    import cv2
    stems = list(dataset["splits"][split])
    if train:
        rng.shuffle(stems)
    for start in range(0, len(stems), batch_size):
        images, masks = [], []
        for stem in stems[start:start + batch_size]:
            image, mask = segdata.read_sample(dataset, stem)
            if train and augment:
                image, mask = segdata.augment(image, mask, rng, textures)
            images.append(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA))
            masks.append(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST))
        yield (np.stack(images).astype(np.float32) / 255.0).transpose(0, 3, 1, 2), np.stack(masks).astype(np.int64)


def confusion_update(confusion: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    valid = target != segdata.IGNORE_INDEX
    classes = confusion.shape[0]
    np.add.at(confusion, (target[valid].clip(0, classes - 1), prediction[valid]), 1)


def summarize(confusion: np.ndarray) -> dict:
    intersection = np.diag(confusion).astype(float)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    present = union > 0
    iou = np.where(present, intersection / np.maximum(union, 1), np.nan)
    return {"pixel_accuracy": float(intersection.sum() / max(1, confusion.sum())),
            "mean_iou": float(np.nanmean(iou)) if present.any() else 0.0,
            "per_class_iou": [None if np.isnan(v) else float(v) for v in iou]}


def evaluate_model(model, dataset: dict, split: str, size: int, device: str, batch_size: int = 8) -> dict:
    torch = _torch()
    confusion = np.zeros((len(dataset["classes"]),) * 2, np.int64)
    model.eval()
    with torch.no_grad():
        for images, masks in make_batches(dataset, split, size, batch_size, np.random.default_rng(0), train=False):
            prediction = model(torch.from_numpy(images).to(device)).argmax(1).cpu().numpy()
            confusion_update(confusion, prediction, masks)
    return summarize(confusion)


def train(args) -> dict:
    torch = _torch()
    torch.manual_seed(args.seed)
    dataset = segdata.load_dataset(args.data)
    if not dataset["splits"].get("train"):
        raise SystemExit("The training split is empty")
    config = {**PAPER_CONFIG, "base_filters": args.base_filters, "depth": args.depth, "up_kernel": args.up_kernel, "input_size": args.input_size}
    model = build_unet(len(dataset["classes"]), **config).to(args.device)
    parameters = count_parameters(model)
    print(f"compressed U-Net: {parameters:,} parameters; classes={dataset['classes']}; input={args.input_size}x{args.input_size}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=.1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=segdata.IGNORE_INDEX)
    textures = None if args.no_augment else segdata.TextureBank(args.textures, args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, losses = time.perf_counter(), []
        lr = optimizer.param_groups[0]["lr"]
        batches = make_batches(dataset, "train", args.input_size, args.batch_size, rng, textures, train=True, augment=not args.no_augment)
        for step, (images, masks) in enumerate(batches):
            optimizer.zero_grad()
            loss = loss_fn(model(torch.from_numpy(images).to(args.device)), torch.from_numpy(masks).to(args.device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if args.max_steps and step + 1 >= args.max_steps:
                break
        scheduler.step()
        record = {"epoch": epoch, "lr": lr, "loss": float(np.mean(losses)) if losses else None, "seconds": round(time.perf_counter() - started, 2)}
        if dataset["splits"].get("val"):
            record.update({f"val_{k}": v for k, v in evaluate_model(model, dataset, "val", args.input_size, args.device, args.batch_size).items()})
        history.append(record)
        print(json.dumps(record), flush=True)
    metrics = {"parameters": parameters, "history": history}
    save_checkpoint(out / "model.pt", model, dataset["classes"], config, metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved {out / 'model.pt'}")
    return metrics


def export_onnx(weights: str | Path, out: str | Path, opset: int = 17, check: bool = True) -> dict:
    torch = _torch()
    model, info = load_checkpoint(weights, "cpu")
    size = int(info["config"].get("input_size", 192))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    example = torch.rand(1, 3, size, size)
    options = {"input_names": ["image"], "output_names": ["logits"], "opset_version": opset,
               "dynamic_axes": {"image": {0: "batch"}, "logits": {0: "batch"}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # the TorchScript exporter needs only the onnx package
        try:
            torch.onnx.export(model, example, str(out), dynamo=False, **options)
        except TypeError:  # older PyTorch without the dynamo flag
            torch.onnx.export(model, example, str(out), **options)
    meta = {"classes": info["classes"], "input_size": size, "input": "image: float32 NCHW RGB in [0,1]", "output": "logits: NCHW class scores",
            "parameters": info.get("parameters"), "source_checkpoint": Path(weights).name}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if check:
        try:
            import onnxruntime as ort
        except ImportError:
            meta["check"] = "onnxruntime not installed; skipped"
        else:
            session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            onnx_out = session.run(None, {"image": example.numpy()})[0]
            with torch.no_grad():
                difference = float(np.abs(onnx_out - model(example).numpy()).max())
            meta["check"] = f"max |onnx - torch| = {difference:.2e}"
    print(json.dumps(meta, indent=2))
    return meta


def predict(weights: str, image_path: str, out: str) -> dict:
    import cv2
    from .segmentation import UNetSegmenter
    segmenter = UNetSegmenter(weights)
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")
    started = time.perf_counter()
    class_map, _ = segmenter.class_map(frame)
    elapsed = (time.perf_counter() - started) * 1000
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{Path(image_path).stem}-classes.png"), class_map)
    palette = np.array([[0, 0, 0]] + [[(37 * i) % 255, (97 * i) % 255, (173 * i) % 255] for i in range(1, 256)], np.uint8)
    overlay = cv2.addWeighted(frame, .6, palette[class_map], .4, 0)
    cv2.imwrite(str(out_dir / f"{Path(image_path).stem}-overlay.png"), overlay)
    instances = segmenter.segment(frame)
    result = {"inference_ms": round(elapsed, 1), "instances": [{"label": i.label, "pixels": int(i.mask.sum()), "confidence": round(i.confidence, 3)} for i in instances]}
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("params", help="Print the compressed U-Net parameter count")
    p.add_argument("--classes", type=int, default=2, help="Output classes including background")
    p.add_argument("--up-kernel", type=int, default=PAPER_CONFIG["up_kernel"])

    p = sub.add_parser("fetch-coco", help="Download COCO val2017 instance annotations and selected images")
    p.add_argument("--out", default="data/coco")
    p.add_argument("--categories", nargs="+", default=["teddy bear"])
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--annotations", help="Existing instances_val2017.json (skips the archive download)")

    p = sub.add_parser("prepare-coco", help="COCO instance annotations -> class-index masks")
    p.add_argument("--annotations", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--categories", nargs="+", default=["teddy bear"])
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--download-missing", action="store_true", help="Fetch missing images individually from coco_url")
    p.add_argument("--min-area", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=.15)

    p = sub.add_parser("prepare-masks", help="Self-labelled frames + binary/indexed PNG masks")
    p.add_argument("--images", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--classes", nargs="+", required=True, help="Object class names (background is implicit)")
    p.add_argument("--class-index", type=int, default=1, help="Index used for binary masks")
    p.add_argument("--out", required=True)
    p.add_argument("--val-fraction", type=float, default=.15)

    p = sub.add_parser("prepare-synthetic", help="Blender renders -> composited images + exact masks")
    p.add_argument("--renders", required=True)
    p.add_argument("--textures", help="DTD images folder (procedural textures when omitted)")
    p.add_argument("--out", required=True)

    p = sub.add_parser("train", help="Train with the paper schedule (Adam 1e-4, /10 every 2 epochs, 5 epochs)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="outputs/unet")
    p.add_argument("--textures", help="DTD images folder for background replacement (procedural fallback)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-step", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--input-size", type=int, default=PAPER_CONFIG["input_size"])
    p.add_argument("--base-filters", type=int, default=PAPER_CONFIG["base_filters"])
    p.add_argument("--depth", type=int, default=PAPER_CONFIG["depth"])
    p.add_argument("--up-kernel", type=int, default=PAPER_CONFIG["up_kernel"])
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="Limit steps per epoch (smoke runs)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("evaluate", help="Pixel accuracy and mIoU on a split")
    p.add_argument("--data", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("export-onnx", help="Export a checkpoint to ONNX (+ sidecar JSON) for onnxruntime / onnxruntime-web")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)

    p = sub.add_parser("predict", help="Segment one image with a .pt or .onnx model")
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="outputs/predict")

    args = parser.parse_args(argv)
    if args.command == "params":
        model = build_unet(args.classes, **{**PAPER_CONFIG, "up_kernel": args.up_kernel})
        print(f"compressed U-Net (16 base filters, 4 poolings, {args.up_kernel}x{args.up_kernel} up-conv, 192x192 input, {args.classes} classes): {count_parameters(model):,} parameters")
    elif args.command == "fetch-coco":
        out = Path(args.out)
        annotations = Path(args.annotations) if args.annotations else segdata.fetch_coco_annotations(out)
        print(json.dumps(segdata.prepare_coco(annotations, out / "val2017", out / "prepared", args.categories, args.limit, download_missing=True), indent=2))
    elif args.command == "prepare-coco":
        print(json.dumps(segdata.prepare_coco(args.annotations, args.images_dir, args.out, args.categories, args.limit, args.download_missing, args.min_area, args.val_fraction), indent=2))
    elif args.command == "prepare-masks":
        print(json.dumps(segdata.prepare_masks(args.images, args.masks, args.out, args.classes, args.class_index, args.val_fraction), indent=2))
    elif args.command == "prepare-synthetic":
        print(json.dumps(segdata.prepare_synthetic(args.renders, args.out, args.textures), indent=2))
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        model, info = load_checkpoint(args.weights, args.device)
        dataset = segdata.load_dataset(args.data)
        if dataset["classes"] != info["classes"]:
            raise SystemExit(f"Dataset classes {dataset['classes']} differ from checkpoint classes {info['classes']}")
        print(json.dumps(evaluate_model(model, dataset, args.split, int(info["config"].get("input_size", 192)), args.device), indent=2))
    elif args.command == "export-onnx":
        export_onnx(args.weights, args.out, args.opset)
    elif args.command == "predict":
        predict(args.weights, args.image, args.out)


if __name__ == "__main__":
    main()
