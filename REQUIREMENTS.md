# Reimplementation requirements

This project implements the paper's silhouette-proxy geometry rather than presenting a box overlay as an AR reconstruction.

## Required behavior

1. Accept binary/label masks from public segmentation systems or user frames; optionally infer instance masks through current YOLO segmentation weights.
2. Extract connected object instances, reject tiny regions, trace contours, and simplify boundaries.
3. Triangulate each contour into usable 2D silhouette geometry.
4. Cast the bottom-midpoint ray to a declared floor plane, then place all contour rays at that reference distance to form the paper's view-dependent planar 3D silhouette.
5. Preserve class/instance metadata and update proxies by class plus spatial proximity.
6. Use masks for per-pixel mutual occlusion, silhouette-derived footprint obstacles for collision-aware paths, and mesh extrema for point/approach/touch/riding targets.
7. Export geometry and debug overlays without requiring an AR device.

## Deliberate boundaries

- The generated proxy is a camera-facing 2.5D silhouette, not hidden-surface recovery or full 3D reconstruction.
- Camera pose, intrinsics, floor plane, and mask quality determine registration accuracy.
- The CLI's camera convention is documented and only demonstrates projection math; mobile AR adapters must supply calibrated pose/plane values.
- No paper model, DollDataset copy, ARCore session, Unity character, or evaluation results are bundled.

## Acceptance checks

- Tests use procedural masks to verify contour/mesh generation, occlusion, contact targets, and collision-aware path failure/success.
- All checks run without downloading a model or dataset.

