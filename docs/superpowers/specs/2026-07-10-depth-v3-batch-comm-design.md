# Depth V3 Batch Inference on CoMM Dataset — Design Spec

**Date:** 2026-07-10
**Branch:** `add_depth_tool`
**Status:** Draft

## Overview

Extend the V3 server and tool to support hidden feature extraction, then build a batch inference script that loads the CoMM `test_data.pth`, extracts all image paths, and runs Depth V3 on each image with configurable output selection.

## Dataset

| Property | Value |
|---|---|
| Location | `comm/test_data.pth` |
| Format | List of dicts with `dataset_type`, `data_id`, `step_info` |
| Test samples | 500 |
| Images | ~12,379 in `comm/val_and_test_images/` |
| Image reference | `step_info[i][j]["image_path"]` (e.g., `images/WikiHow/565261.jpg`) |
| Image root | `comm/val_and_test_images/` |

## Components

### 1. Server Extension (`depth_v3_server.py`)

- New endpoint: `POST /features` — accepts base64 image, returns last-layer hidden feature tensor
- Extend `POST /infer` to support `output_mode="features"` returning depth + features in one call
- Feature extraction hooks into the model forward pass to capture the penultimate layer output before the depth head
- Feature tensor returned as base64-encoded float32 bytes with shape metadata

### 2. Client Extension (`depth_v3_client.py`)

- Extend `infer()` to handle `output_mode="features"` return shape
- New method: `infer_with_features(image_path)` — calls `/features` endpoint, saves `.pth` feature file

### 3. Mock Extension (`mock_depth_v3_service.py`)

- `output_mode="features"` returns a synthetic feature tensor (shape `[1, 256, H//14, W//14]`) saved as `.pth`

### 4. Tool Extension (`depth_tool.py`)

- Add `"features"` to `V3_OUTPUT_MODES` enum
- V3 parameters schema updated with the new mode

### 5. Batch Script (`scripts/run_depth_v3_batch.py`)

- Loads CoMM `test_data.pth`, extracts all unique `image_path` entries
- Resolves image paths relative to `comm/val_and_test_images/`
- CLI flags: `--save-depth`, `--save-metric-depth`, `--save-gaussians`, `--save-features`
- Calls the Depth V3 tool (mock or real) for each image
- Per-image output naming: `depth_v3_{source}_{id}_depth.png`, etc.
- Progress bar, resume support (skip existing outputs), summary stats

## Output File Structure

```
outputs/depth_v3_batch/
├── WikiHow/
│   ├── 565261_depth.png          # --save-depth
│   ├── 565261_metric.npy         # --save-metric-depth
│   ├── 565261_gs.ply             # --save-gaussians
│   └── 565261_features.pth       # --save-features
├── Instructables/
│   └── ...
└── batch_summary.json            # run metadata, stats, errors
```

## Feature Tensor Shape

DA3NESTED-GIANT-LARGE-1.1 uses a DINO encoder. Last-layer hidden features before the depth head have shape `[1, num_patches, embed_dim]` where `embed_dim ≈ 1536` for the nested giant model and `num_patches = (H/14) × (W/14)`.

Saved as a `.pth` dict:

```python
{
    "image_id": "WikiHow_565261",
    "features": tensor,           # shape [1, num_patches, embed_dim]
    "patch_h": H // 14,
    "patch_w": W // 14,
    "embed_dim": 1536,
}
```

## Batch Script CLI

```bash
python scripts/run_depth_v3_batch.py \
    --data_path comm/test_data.pth \
    --image_root comm/val_and_test_images \
    --output_dir outputs/depth_v3_batch \
    --save-depth --save-metric-depth --save-gaussians --save-features \
    --use_mock        # for testing without GPU

# Real server
python scripts/run_depth_v3_batch.py \
    --data_path comm/test_data.pth \
    --image_root comm/val_and_test_images \
    --output_dir outputs/depth_v3_batch \
    --save-depth --save-features \
    --server_url http://127.0.0.1:20039
```

## Resume Support

The script checks for existing output files before processing each image. If all requested outputs already exist for an image, it is skipped. This makes the batch run safe to interrupt and restart.

## Error Handling

- Missing image file → log warning, skip, continue
- Server unreachable → log error, record in summary, continue
- Individual image failure → log error, record in summary, continue
- At end, `batch_summary.json` contains: total images, processed, skipped, failed, per-image status

## Non-Goals

- SPAgent LLM evaluation (batch inference only)
- Video input support
- Multi-view batch (single image per inference call)
- Aggregated feature file (per-image `.pth` only)
