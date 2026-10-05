# Reimplementation requirements

This project implements the paper's segmentation-to-silhouette pipeline and its virtual-human interactions, rather than presenting a box overlay as an AR reconstruction.

## Required behavior

1. **Segmentation network.** A compressed U-Net with 16 first-level filters, 4 poolings, a 192×192 RGB input and multi-class softmax output (background + K classes). Its parameter count is printed and tested: 2,160,194 for one class; the paper reports about 2.4M.
2. **Training.** Adam starting at 1e-4, divided by 10 every two epochs, for 5 epochs by default. Augmentation:
   - background replacement with DTD textures (procedural textures when DTD is absent);
   - 0–360° rotation;
   - zoom by changing the object's area ratio;
   - horizontal and vertical flips.
3. **Data preparation.** One images + class-index-mask contract with three routes:
   - COCO instance annotations (polygon and RLE) for selected categories; `teddy bear` is the closest public doll class;
   - self-labelled frames with binary or indexed PNG masks;
   - optional Blender renders with exact per-object masks and recorded asset provenance.
4. **Inference and export.** A U-Net segmenter from a PyTorch checkpoint or an ONNX export, and an ONNX export command for onnxruntime / onnxruntime-web. YOLO stays an alternative: its masks are rebuilt from original-pixel polygons (`masks.xy`), and every instance keeps its class label.
5. **Instances.** Split masks into connected instances per class and drop regions under 500 px (paper, Section 3.2). Trace and simplify contours, then triangulate them.
6. **Mesh.** Cast the bounding box's bottom-midpoint ray to the floor plane, then place every contour ray at that distance. The result is the paper's equal-distance, camera-centred silhouette mesh, which tilts with the device. The earlier contour-bottom rule remains as an option. Instances whose floor ray misses, or whose contour cannot be triangulated, are skipped with a warning; the rest of the frame is still built.
7. **Floor footprint.** The orthographic projection of the mesh vertices onto the floor plane, for any yaw, pitch or floor normal. It is padded along the view direction when the view is level.
8. **Update loop.** Match meshes each frame by class and a small floor distance (`match_existing`), keeping persistent ids so that moving objects can be followed.
9. **Selection.** Ray-cast from the camera through a tap or the screen centre (gaze), or match a keyword (class name or id) from speech recognition or typed text.
10. **Interactions:**
    - walkable floor with dilated holes, 8-connected A* with an octile heuristic and path smoothing;
    - a standing point slightly toward the camera from the centre of the un-walkable area, outside the dilated hole, used by approach, pet and push;
    - point, approach, follow (re-planned every update), pet, push and ride (seated on the silhouette's body top).
11. **Occlusion.** The silhouette meshes are rendered depth-only over the camera image so that they hide the avatar. A 2D mask or depth compositor remains for offline outputs.
12. **Web demo.** Desktop webcam or uploaded image/video with one-tap floor calibration, server-side (U-Net/YOLO) or in-browser ONNX segmentation, and a sample scene with an authored mask that needs no model.

## Deliberate boundaries

- The silhouette mesh is a camera-facing 2.5D proxy, not hidden-surface recovery or full 3D reconstruction.
- Registration accuracy depends on camera pose, intrinsics, the floor plane and mask quality. The demo assumes the principal point at the image centre and calibrates pitch only.
- The original institute code, DollDataset annotations, trained weights, Unity character, user-study material and results are not bundled. Prepared datasets and trained models stay in ignored `data/` and `outputs/`.
- **Not reproduced in the web demo:**
  - Live mobile AR with on-device segmentation. WebXR camera access is Android-only and experimental, so the demo uses a desktop webcam or a file with manual floor calibration instead of ARCore tracking.
  - The paper's mobile timings, and its rendering decoupled from the segmentation pipeline.

## Acceptance checks

- Tests use tiny synthetic data and need no downloads. They cover:
  - the U-Net parameter count;
  - data converters and augmentation alignment;
  - the learning-rate schedule;
  - training, ONNX export and its parity with PyTorch;
  - geometry, tracking, selection and planning;
  - the audit regressions: yaw footprint, reachable standing point, skipped bad instances, YOLO letterbox and the octile heuristic.
- `scripts/verify.py` writes mesh, occlusion, update-loop and path outputs.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; mouth shapes follow a rule-based text-to-phoneme-to-viseme track timed to speech playback, an approximation rather than forced phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
