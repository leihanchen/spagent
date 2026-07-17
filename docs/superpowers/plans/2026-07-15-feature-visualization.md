# Feature Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inline PCA→RGB feature visualization PNG generation alongside each `*_features.pth` during Depth V3 batch inference.

**Architecture:** A new `_save_features_vis()` static method on `DepthV3Client` decodes the already-available `features_b64`, runs numpy-only SVD PCA to 3 components, reshapes to the patch grid, upsamples via nearest-neighbor, and saves a PNG. The mock service gets a parallel `_create_mock_features_vis()`. The batch script copies the vis file and checks it for resume.

**Tech Stack:** numpy (SVD), PIL/cv2 (resize + save), torch (load .pth in tests only)

## Global Constraints

- No sklearn dependency — use `numpy.linalg.svd` for PCA
- No changes to `depth_v3_server.py` — all visualization is client-side
- No changes to `run_depth_v3_batch.slurm` — `--save-features` already enabled
- Output naming: `{image_id}_features_vis.png` in same source subdirectory as other outputs
- Feature tensor shape: `[1, num_patches, 256]` where `num_patches = feature_h × feature_w`
- Default mock image shape: `[480, 640]` → `feature_h=34, feature_w=45`

---

### Task 1: Add `_save_features_vis()` to `DepthV3Client`

**Files:**
- Modify: `spagent/external_experts/Depth_AnythingV2/depth_v3_client.py:378-420` (add method after `_save_features_pth`)
- Test: `test/test_3d_tools.py`

**Interfaces:**
- Consumes: `features_b64` (base64-encoded float32 bytes from server), `feature_h`, `feature_w`, `feature_dim` (from server response)
- Produces: `_save_features_vis(features_b64, feature_h, feature_w, feature_dim, output_path)` — static method that saves a PCA→RGB PNG and returns the output path string. Also produces `result["features_vis_path"]` key in `_process_server_result()` and `extract_features()` return dicts.

- [ ] **Step 1: Write the failing test for `features_vis_path` in features mode**

Add to `test/test_3d_tools.py` in `TestDepthV3Tool`, after `test_v3_features_spatial_structure` (~line 311):

```python
def test_v3_features_vis_png(self):
    """V3 features mode should also produce a PCA→RGB visualization PNG."""
    import numpy as np
    from PIL import Image

    from spagent.tools import DepthEstimationTool

    tool = DepthEstimationTool(use_mock=True, backend="v3")
    result = tool.call(
        image_path="assets/example.png",
        output_mode="features",
        feature_source="depth_decoder",
    )

    assert result["success"] is True
    assert "features_vis_path" in result
    assert result["features_vis_path"] is not None
    assert os.path.exists(result["features_vis_path"])
    assert result["features_vis_path"].endswith("_features_vis.png")

    # Verify it's a valid RGB image
    img = Image.open(result["features_vis_path"])
    assert img.mode == "RGB"
    arr = np.array(img)
    assert arr.ndim == 3
    assert arr.shape[2] == 3
    # Should not be all-black (PCA of random features produces color)
    assert arr.max() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /scratch/leihan/spagent && python -m pytest test/test_3d_tools.py::TestDepthV3Tool::test_v3_features_vis_png -v`
Expected: FAIL with `AssertionError` on `"features_vis_path" in result` (key doesn't exist yet)

- [ ] **Step 3: Add `_save_features_vis()` static method to `DepthV3Client`**

Add after `_save_features_pth` (after line 420) in `spagent/external_experts/Depth_AnythingV2/depth_v3_client.py`:

```python
@staticmethod
def _save_features_vis(
    features_b64: str,
    feature_h: int,
    feature_w: int,
    feature_dim: int,
    output_path: str,
) -> str:
    """PCA→RGB visualization of 3D-aware features.

    Decodes base64 features, projects to 3 components via numpy SVD,
    normalizes each channel to [0, 255], reshapes to the patch grid,
    upsamples to a reasonable display size, and saves as PNG.

    Args:
        features_b64: Base64-encoded float32 feature bytes.
        feature_h: Patch grid height.
        feature_w: Patch grid width.
        feature_dim: Feature vector dimension (typically 256).
        output_path: Path to save the PNG.

    Returns:
        The output_path on success.
    """
    from PIL import Image as PILImage

    # Decode features
    features_bytes = base64.b64decode(features_b64)
    patches = np.frombuffer(features_bytes, np.float32).copy()

    num_patches = feature_h * feature_w
    if num_patches > 0 and feature_dim > 0:
        patches = patches.reshape(num_patches, feature_dim)
    else:
        logger.warning(
            "Invalid feature dims for vis: h=%d w=%d dim=%d; skipping vis",
            feature_h, feature_w, feature_dim,
        )
        return ""

    # PCA to 3 components via numpy SVD (no sklearn needed)
    n_components = min(3, feature_dim, num_patches)
    centered = patches - patches.mean(axis=0, keepdims=True)
    # Economy SVD: U(N×k), S(k), Vt(k×D) — only need Vt for projection
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:n_components].T  # (N, n_components)

    # Pad to 3 channels if fewer components
    if n_components < 3:
        pad = np.zeros((num_patches, 3 - n_components), dtype=np.float32)
        projected = np.concatenate([projected, pad], axis=1)

    # Per-channel min-max normalization to [0, 255]
    rgb = np.zeros((num_patches, 3), dtype=np.uint8)
    for c in range(3):
        col = projected[:, c]
        col_min, col_max = float(col.min()), float(col.max())
        if col_max - col_min > 1e-8:
            rgb[:, c] = np.round((col - col_min) / (col_max - col_min) * 255).astype(
                np.uint8
            )

    # Reshape to spatial grid
    rgb = rgb.reshape(feature_h, feature_w, 3)

    # Nearest-neighbor upsample to a reasonable display size
    display_h = max(feature_h * 8, 224)
    display_w = max(feature_w * 8, 224)
    rgb_pil = PILImage.fromarray(rgb, mode="RGB").resize(
        (display_w, display_h), PILImage.NEAREST
    )
    rgb_pil.save(output_path)
    logger.info("V3 features vis saved: %s (%dx%d grid → %dx%d)",
                output_path, feature_w, feature_h, display_w, display_h)
    return output_path
```

- [ ] **Step 4: Wire `_save_features_vis` into `_process_server_result()`**

In `depth_v3_client.py`, inside `_process_server_result()`, after the block that sets `result["features_path"]` (after line 333), add:

```python
            # -- feature visualization (PCA→RGB) -------------------------
            vis_path = os.path.join(self.output_dir, f"{base_name}_features_vis.png")
            self._save_features_vis(
                server_result["features_b64"],
                feature_h,
                feature_w,
                feature_dim,
                vis_path,
            )
            if vis_path:
                result["features_vis_path"] = vis_path
```

- [ ] **Step 5: Wire `_save_features_vis` into `extract_features()`**

In `depth_v3_client.py`, inside `extract_features()`, after the `self._save_features_pth(...)` call (after line 223), add:

```python
            # Feature visualization
            vis_path = output_path.replace("_features.pth", "_features_vis.png")
            self._save_features_vis(
                server_result["features_b64"],
                feature_h,
                feature_w,
                feature_dim,
                vis_path,
            )
```

And in the return dict (after `"features_path": output_path,`), add:

```python
                "features_vis_path": vis_path if vis_path else None,
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /scratch/leihan/spagent && python -m pytest test/test_3d_tools.py::TestDepthV3Tool::test_v3_features_vis_png -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add spagent/external_experts/Depth_AnythingV2/depth_v3_client.py test/test_3d_tools.py
git commit -m "feat(depth): add PCA→RGB feature visualization to V3 client

_save_features_vis() decodes features_b64, runs numpy SVD PCA to
3 components, normalizes to [0,255], reshapes to patch grid, and
saves as nearest-neighbor upsampled PNG. Wired into both
_process_server_result() and extract_features().

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 2: Add `_create_mock_features_vis()` to `MockDepthV3Service`

**Files:**
- Modify: `spagent/external_experts/Depth_AnythingV2/mock_depth_v3_service.py:306-363` (add method after `_create_mock_features`, wire into `infer()`)
- Test: `test/test_3d_tools.py` (already covered by Task 1's test since mock is used)

**Interfaces:**
- Consumes: `shape` (image [H, W]), `feature_source` — same as `_create_mock_features`
- Produces: `result["features_vis_path"]` key in mock `infer()` return dict

- [ ] **Step 1: Add `_create_mock_features_vis()` method**

Add after `_create_mock_features` (after line 362) in `mock_depth_v3_service.py`:

```python
def _create_mock_features_vis(
    self,
    output_path: str,
    shape: List[int],
    feature_source: str = "depth_decoder",
) -> str:
    """Generate a synthetic PCA-like feature visualization PNG.

    Creates a colored grid at patch resolution and upsamples via
    nearest-neighbor, matching the real client's _save_features_vis output.

    Args:
        output_path: Path to save the PNG.
        shape: Image shape [H, W].
        feature_source: Feature source label (for logging only).

    Returns:
        The output_path.
    """
    h, w = shape
    feature_h = h // 14
    feature_w = w // 14

    # Synthetic colored grid: spatial gradient in R/G/B channels
    row_grad = np.linspace(0, 255, feature_h, dtype=np.uint8).reshape(-1, 1)
    col_grad = np.linspace(0, 255, feature_w, dtype=np.uint8).reshape(1, -1)
    r = np.broadcast_to(row_grad, (feature_h, feature_w)).copy()
    g = np.broadcast_to(col_grad, (feature_h, feature_w)).copy()
    b = np.full((feature_h, feature_w), 128, dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    # Nearest-neighbor upsample to display size
    display_h = max(feature_h * 8, 224)
    display_w = max(feature_w * 8, 224)
    rgb_pil = Image.fromarray(rgb, mode="RGB").resize(
        (display_w, display_h), Image.NEAREST
    )
    rgb_pil.save(output_path)
    logger.info(
        "Mock V3 features vis saved: %s (source=%s, %dx%d grid → %dx%d)",
        output_path, feature_source, feature_w, feature_h, display_w, display_h,
    )
    return output_path
```

- [ ] **Step 2: Wire into `infer()` features block**

In `mock_depth_v3_service.py`, inside `infer()`, after the features block that sets `result["features_path"]` (after line 137), add:

```python
                # Feature visualization
                vis_path = os.path.join(self.output_dir, f"{base_name}_features_vis.png")
                self._create_mock_features_vis(vis_path, shape, feature_source=feature_source)
                result["features_vis_path"] = vis_path
```

- [ ] **Step 3: Run the test to verify it still passes**

Run: `cd /scratch/leihan/spagent && python -m pytest test/test_3d_tools.py::TestDepthV3Tool::test_v3_features_vis_png -v`
Expected: PASS (mock now produces `features_vis_path`)

- [ ] **Step 4: Run the full V3 test suite to check for regressions**

Run: `cd /scratch/leihan/spagent && python -m pytest test/test_3d_tools.py::TestDepthV3Tool -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add spagent/external_experts/Depth_AnythingV2/mock_depth_v3_service.py
git commit -m "feat(depth): add mock feature visualization PNG

_create_mock_features_vis() generates a synthetic colored grid
at patch resolution, upsampled via nearest-neighbor. Wired into
infer() so mock mode produces features_vis_path like the real client.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 3: Wire feature visualization into batch script

**Files:**
- Modify: `scripts/run_depth_v3_batch.py:148-189` (resume check), `scripts/run_depth_v3_batch.py:339-356` (features block)
- Test: manual verification (batch script uses the tool which is already tested)

**Interfaces:**
- Consumes: `result["features_vis_path"]` from `DepthEstimationTool.call(output_mode="features")`
- Produces: `{image_id}_features_vis.png` copied to source output dir; resume check includes this file

- [ ] **Step 1: Add `features_vis.png` to resume check**

In `run_depth_v3_batch.py`, inside `check_existing_outputs()`, after the line that appends `features_path` (line 170), add:

```python
    features_vis_path = os.path.join(source_dir, f"{image_id}_features_vis.png")
    if save_flags.get("features"):
        expected.append(features_vis_path)
```

Note: declare `features_vis_path` before the `if` block so it's available for the stale-check below. The full block becomes:

```python
    features_path = os.path.join(source_dir, f"{image_id}_features.pth")
    features_vis_path = os.path.join(source_dir, f"{image_id}_features_vis.png")
    if save_flags.get("features"):
        expected.append(features_path)
        expected.append(features_vis_path)
```

- [ ] **Step 2: Copy `features_vis_path` in the features block**

In `run_depth_v3_batch.py`, inside the features block (after line 352 where `features_path` is copied), add:

```python
                    _copy_output(
                        result.get("features_vis_path"),
                        source_dir,
                        f"{image_id}_features_vis.png",
                    )
```

- [ ] **Step 3: Run the full test suite to check for regressions**

Run: `cd /scratch/leihan/spagent && python -m pytest test/test_3d_tools.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/run_depth_v3_batch.py
git commit -m "feat(depth): wire feature visualization into batch pipeline

Copy features_vis_path to {image_id}_features_vis.png alongside
features.pth. Add _features_vis.png to resume check so incomplete
runs are reprocessed.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

### Task 4: Update SLURM script output listing

**Files:**
- Modify: `scripts/run_depth_v3_batch.slurm:299-303` (metric artifacts echo block)

**Interfaces:**
- No new interfaces — just documentation in the SLURM script's completion message

- [ ] **Step 1: Add `*_features_vis.png` to the output listing**

In `run_depth_v3_batch.slurm`, update the "Metric artifacts per image" block (lines 299-303) to include the new file:

```bash
echo " Metric artifacts per image:"
echo "   *_metric_depth.npy        (float32 meters, HxW)"
echo "   *_metric_depth_vis.png    (color visualization only)"
echo "   *_features.pth            (3D-aware features, 256-dim per patch)"
echo "   *_features_vis.png        (PCA→RGB feature visualization)"
echo "   *_metrics.json            (optional min/max summary)"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_depth_v3_batch.slurm
git commit -m "docs(depth): list *_features_vis.png in SLURM output summary

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```
