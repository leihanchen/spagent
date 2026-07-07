# Depth Anything V3 Tool Integration — Design Spec

**Date:** 2026-07-07
**Branch:** `new_tool`
**Status:** Draft

## Overview

Add Depth Anything V3 (`DA3NESTED-GIANT-LARGE-1.1`) as a new backend option for the existing `DepthEstimationTool`. V3 brings metric depth estimation, 3D gaussian splatting, camera pose estimation, and spatial metrics — all absent from the current V2 tool. The V2 backend remains the default and is untouched.

## Model

| Property | Value |
|---|---|
| Model ID | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` |
| Parameters | 1.40B (nested: 1.15B any-view Giant + 0.35B metric Large) |
| License | CC BY-NC 4.0 (non-commercial) |
| Format | Safetensors, auto-download from HuggingFace |
| Architecture | Plain DINO encoder + unified depth-ray representation |
| Capabilities | Relative depth, metric depth, camera pose estimation, 3D gaussians, sky segmentation, pose-conditioned inference |

## Architecture

Three layers, following the existing server/client/tool pattern:

### 1. External Expert (`external_experts/Depth_AnythingV2/`)

New files added alongside existing V2 code:

```
external_experts/Depth_AnythingV2/
├── depth_server.py              # unchanged (V2)
├── depth_client.py              # unchanged (V2)
├── depth_anything_v2/           # unchanged (V2 model code)
├── depth_anything_v3/           # NEW: V3 model code
│   ├── __init__.py
│   ├── da3_model.py             # DA3NESTED-GIANT-LARGE-1.1 wrapper
│   └── dino_encoder.py          # DINO backbone (if not imported from upstream)
├── depth_v3_server.py           # NEW: Flask server (port 20039)
├── depth_v3_client.py           # NEW: HTTP client
├── mock_depth_v3_service.py     # NEW: Mock service
├── mock_depth_service.py        # NEW: Mock for V2 (referenced but missing)
└── __init__.py                  # MODIFIED: export V3 classes
```

**Server** (`depth_v3_server.py`):
- Flask server on port `20039`
- Loads model via `DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")`
- Endpoints: `GET /health`, `POST /infer`
- Input: base64-encoded image(s), output mode, render flags
- Output: depth map (colored), metric depth values, PLY bytes, GS PLY bytes, confidence map, camera poses, rendered view images

**Client** (`depth_v3_client.py`):
- HTTP client matching the server interface
- `health_check()`, `infer(image_path, output_mode, return_metrics, render_views)`
- Saves outputs under `outputs/` with deterministic filenames

**Mock service** (`mock_depth_v3_service.py`):
- Returns synthetic depth map + gaussian placeholder for testing
- No GPU or checkpoint needed
- Matches the same return shape as the real client

### 2. Tool (`tools/depth_tool.py`)

`DepthEstimationTool.__init__` gains a `backend` parameter:

```python
def __init__(self, use_mock: bool = True,
             server_url: str = "http://10.8.131.51:20019",
             backend: str = "v2"):
```

- `backend="v2"`: loads V2 client/mock (existing behavior, unchanged)
- `backend="v3"`: loads V3 client/mock from `depth_v3_client` / `mock_depth_v3_service`

`parameters` property returns different schemas per backend:
- V2: `{"image_path": string}` (unchanged)
- V3: `image_path` (string or list[string]), `output_mode` (enum), `return_metrics` (boolean), `render_views` (boolean)

`call()` dispatches to the appropriate client and normalizes the response.

### 3. Catalog (`tools/catalog.py`)

The `"depth"` catalog entry gains `backend="v2"` in `default_kwargs`:

```python
ToolCatalogEntry(
    "depth",
    DepthEstimationTool,
    "2d_perception",
    "depth_estimation_tool",
    {"server_url": DEFAULT_SERVER_URLS["depth"], "backend": "v2"},
),
```

V3 server URL added to `DEFAULT_SERVER_URLS`:

```python
"depth_v3": "http://127.0.0.1:20039",
```

Users opt into V3 via:

```python
build_tools(overrides={"depth": {"backend": "v3", "server_url": "http://..."}})
```

## V3 Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string or list[string] | ✅ | — | Single image (monocular) or list (multi-view). Multi-view improves pose and geometry accuracy. |
| `output_mode` | enum | ❌ | `"depth"` | `"depth"` — relative depth map (colored visualization), `"metric_depth"` — absolute depth in meters, `"point_cloud"` — PLY point cloud, `"gaussians"` — 3D gaussian splatting (GS PLY + rendered views) |
| `return_metrics` | boolean | ❌ | `false` | When true, include computed spatial metrics: depth min/max/mean (meters), point count, coverage percent |
| `render_views` | boolean | ❌ | `false` | When true, render the 3D output from multiple canonical viewpoints as base64 images (works with point_cloud and gaussians modes) |

## V3 Return Shape

```python
{
    "success": True,
    "backend": "v3",
    "output_mode": "metric_depth",
    "output_path": "outputs/depth_v3_scene_001.png",     # primary visualization
    "ply_path": "outputs/depth_v3_scene_001.ply",         # for point_cloud / gaussians
    "gs_ply_path": "outputs/depth_v3_scene_001_gs.ply",   # for gaussians mode
    "rendered_views": [                                    # if render_views=True
        {"view": "front", "image": "<base64>"},
        {"view": "top", "image": "<base64>"},
        {"view": "side", "image": "<base64>"},
    ],
    "camera_pose": {                                       # always included for V3
        "extrinsics": [...],  # world-to-camera, shape [3, 4]
        "intrinsics": [...],  # shape [3, 3]
    },
    "shape": [480, 640],
    "metrics": {                    # if return_metrics=True
        "depth_min_m": 0.5,
        "depth_max_m": 12.3,
        "depth_mean_m": 3.1,
        "point_count": 307200,
        "coverage_percent": 98.5
    }
}
```

## Error Handling

- Image file not found → `{"success": False, "error": "Image file not found: ..."}`
- Invalid output_mode → `{"success": False, "error": "Invalid output_mode: ..."}`
- Server unreachable → `{"success": False, "error": "V3 server unreachable: ..."}`
- Model load failure → logged at server startup, `/health` returns error status
- Mock mode → always returns synthetic data, never fails (unless image missing)

## Testing

- **Unit tests** (`test/test_depth_v3_tool.py`): mock-mode tests for all output modes, parameter validation, error paths
- **Integration tests** (`test/test_depth_v3_tool.py`): real server tests (gated behind `DEPTH_V3_REAL_TEST=1` env var)
- **CLI smoke tests**: `python test/test_tool.py --tool depth --backend v3 --use_mock --image assets/example.png --output_mode metric_depth`

## Backward Compatibility

| Concern | Resolution |
|---|---|
| Existing V2 users | `backend` defaults to `"v2"`. No change to existing code, configs, or behavior. |
| Tool name / catalog key | Unchanged: `depth_estimation_tool` / `"depth"` |
| `use_mock` semantics | Same for both backends |
| V2 server/client/model | Completely untouched |
| V2 `__init__.py` exports | Existing exports preserved; V3 exports added |

## Dependencies

- `torch >= 2.0`
- `transformers >= 4.45`
- `huggingface_hub` (for model auto-download)
- `flask` (for server)
- `opencv-python` (for image processing)
- `numpy`
- `pillow`
- `matplotlib` (for depth map colormap)
- `open3d` or `trimesh` (for PLY export, optional)

## Non-Goals

- Replacing V2 as the default backend
- Adding V3 to the `3d` tool group (stays in `2d_perception`)
- Supporting all DA3 variants (only `DA3NESTED-GIANT-LARGE-1.1` is targeted)
- Video input support (single image or image list only)
- Real-time/interactive 3D viewer (static renders only)
