# Feature Visualization for Depth V3 Batch Pipeline

**Date:** 2026-07-15
**Status:** Approved

## Summary

Generate a PCA→RGB visualization PNG inline alongside each `*_features.pth` during the Depth V3 SLURM batch run. No separate post-processing step required.

## Motivation

The batch pipeline currently saves raw 3D-aware features as `.pth` files (256-dim per patch) but produces no visual output for them. Metric depth gets a `*_metric_depth_vis.png` color visualization; features get nothing. A quick PCA→RGB patch grid makes it easy to visually verify that features are reasonable (spatially coherent, not degenerate) without loading `.pth` files and running a separate script.

## Design

### Output

Per image (when `--save-features` is enabled):

| File | Description |
|---|---|
| `*_features.pth` | Raw feature tensor + metadata (existing, unchanged) |
| `*_features_vis.png` | PCA→RGB patch grid visualization (new) |

Naming follows the existing `*_metric_depth_vis.png` convention.

### PCA→RGB Pipeline

1. Decode `features_b64` from server response → float32 array `[1, N, D]`
2. PCA to 3 components (numpy SVD on the small `N×D` matrix; no sklearn dependency)
3. Per-channel min-max normalization to [0, 255]
4. Reshape to `[feature_h, feature_w, 3]`
5. Nearest-neighbor upsample to input image size via PIL/cv2
6. Save as PNG

The 37×37×256 feature map is tiny; numpy SVD takes <1ms. No GPU needed.

### Files Changed

#### `spagent/external_experts/Depth_AnythingV2/depth_v3_client.py`

- Add `_save_features_vis()` static method: takes decoded features array, spatial dims, input image size, output path. Runs PCA→RGB, upsamples, saves PNG. Returns the output path.
- In `_process_server_result()`: after saving `features.pth`, also call `_save_features_vis()` and set `result["features_vis_path"]`.
- In `extract_features()`: same — save vis alongside `.pth`, return `features_vis_path`.

#### `spagent/external_experts/Depth_AnythingV2/mock_depth_v3_service.py`

- Add `_create_mock_features_vis()`: generates a synthetic colored grid (random RGB at patch resolution, upsampled). Returns `features_vis_path`.
- In `infer()`: when `output_mode == "features"`, also call `_create_mock_features_vis()` and set `result["features_vis_path"]`.

#### `scripts/run_depth_v3_batch.py`

- In the features block (~line 340): after copying `features_path`, also copy `features_vis_path` to `{image_id}_features_vis.png`.
- In `check_existing_outputs()`: add `{image_id}_features_vis.png` to the expected files list when `save_flags["features"]` is true.

#### `scripts/run_depth_v3_batch.slurm`

- No changes. `--save-features` is already enabled.

### Dependency Choice: numpy-only PCA

Use `numpy.linalg.svd` instead of `sklearn.decomposition.PCA` to avoid adding sklearn as a dependency in the SLURM batch environment. The feature matrix is at most ~1400×256 (37×37 patches × 256 dims), so full SVD is negligible. The existing `visualize_depth_v3_outputs.py` post-hoc script can keep sklearn for its richer options.

### Resume Compatibility

The `check_existing_outputs()` function already checks for `*_features.pth`. Adding `*_features_vis.png` to the check means a resume will reprocess any image missing the vis file. For images already processed before this change (have `.pth` but no `_vis.png`), re-running will regenerate just the visualization.

## Out of Scope

- Cosine-similarity heatmaps (can be added later)
- DA3 `PCARGBVisualizer` integration (GPU-accelerated, overkill for single-frame batch)
- Changes to the server (`depth_v3_server.py`) — all visualization is client-side
- Changes to the standalone `visualize_depth_v3_outputs.py` script
