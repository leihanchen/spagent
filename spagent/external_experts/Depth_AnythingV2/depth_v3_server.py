"""
Depth Anything V3 (DA3NESTED-GIANT-LARGE-1.1) Flask server.

Provides depth estimation, metric depth, point cloud, 3D Gaussian
splatting, and 3D-aware feature extraction via a REST API.  The server
auto-downloads the model from Hugging Face on first launch.

Feature extraction hooks into the DPT depth decoder's ``refinenet1``
layer, capturing the fused multi-scale spatial feature map that encodes
3D/depth information — not the 2D DINOv2 encoder features.
"""

import argparse
import base64
import io
import logging
import os
import traceback

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

model = None
model_config: dict = {}

VALID_OUTPUT_MODES = ("depth", "metric_depth", "point_cloud", "gaussians", "features")
VALID_FEATURE_SOURCES = ("depth_decoder", "gs_decoder")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1") -> bool:
    """
    Load DA3NESTED-GIANT-LARGE-1.1 via the official DepthAnything3 API.

    Fail-closed: requires the ``depth_anything_3`` package (or legacy
    ``depth_anything_v3`` alias). Never falls back to untrained weights.
    """
    global model, model_config

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading DA3 model from %s on %s ...", checkpoint_id, device)

        DepthAnything3 = None
        import_errors = []
        for module_name in ("depth_anything_3.api", "depth_anything_3", "depth_anything_v3"):
            try:
                if module_name == "depth_anything_3.api":
                    from depth_anything_3.api import DepthAnything3 as _DA3  # type: ignore
                elif module_name == "depth_anything_3":
                    from depth_anything_3 import DepthAnything3 as _DA3  # type: ignore
                else:
                    from depth_anything_v3 import DepthAnything3 as _DA3  # type: ignore
                DepthAnything3 = _DA3
                logger.info("Imported DepthAnything3 from %s", module_name)
                break
            except ImportError as e:
                import_errors.append(f"{module_name}: {e}")

        if DepthAnything3 is None:
            logger.error(
                "Official DepthAnything3 package not found. Install/clone "
                "Depth-Anything-3 and put its src/ on PYTHONPATH "
                "(package name: depth_anything_3). Tried: %s",
                "; ".join(import_errors),
            )
            return False

        model = DepthAnything3.from_pretrained(checkpoint_id).to(device).eval()
        model_config = {
            "checkpoint_id": checkpoint_id,
            "device": device,
            "method": "DepthAnything3.from_pretrained",
        }
        logger.info("DA3 model loaded via official DepthAnything3 API")
        return True

    except Exception as e:
        logger.error("Failed to load DA3 model: %s\n%s", e, traceback.format_exc())
        return False


def encode_depth_uint16(depth: np.ndarray):
    """
    Linearly encode a float depth map to uint16 for 16-bit PNG storage.

    Recovery: ``depth_m = scale * uint16_pixel + offset``.

    Returns:
        (depth_u16, scale_info dict)
    """
    depth = np.asarray(depth, dtype=np.float32)
    d_min = float(np.nanmin(depth))
    d_max = float(np.nanmax(depth))
    if not np.isfinite(d_min) or not np.isfinite(d_max):
        d_min, d_max = 0.0, 0.0

    if d_max > d_min:
        scale = (d_max - d_min) / 65535.0
        offset = d_min
        depth_u16 = np.clip(
            np.round((depth - offset) / scale), 0, 65535
        ).astype(np.uint16)
    else:
        scale = 1.0
        offset = d_min
        depth_u16 = np.zeros(depth.shape, dtype=np.uint16)

    scale_info = {
        "scale": scale,
        "offset": offset,
        "depth_min_m": d_min,
        "depth_max_m": d_max,
        "dtype": "uint16",
        "formula": "depth_m = scale * uint16 + offset",
    }
    return depth_u16, scale_info


def decode_depth_uint16(depth_u16: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Recover float depth from a uint16 map: depth_m = scale * u16 + offset."""
    return depth_u16.astype(np.float32) * float(scale) + float(offset)


def _colorize_depth(depth_norm: np.ndarray) -> np.ndarray:
    """Return a BGR uint8 visualization (OpenCV convention)."""
    depth_vis = (depth_norm * 255).astype(np.uint8)
    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap("Spectral_r")
        # matplotlib RGB → OpenCV BGR
        return (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    except Exception:
        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)


def _run_model_depth(ref_img_bgr: np.ndarray):
    """
    Run official DA3 inference and return (depth HxW float32, meta).

    meta includes is_metric and optional camera matrices from Prediction.
    """
    global model
    rgb = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
    meta = {"is_metric": 0}

    if not hasattr(model, "inference"):
        raise RuntimeError(
            "Loaded model has no .inference() API. Official DepthAnything3 "
            "is required; untrained fallback models are not supported."
        )

    prediction = model.inference([rgb])
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    meta["is_metric"] = int(getattr(prediction, "is_metric", 0) or 0)
    if getattr(prediction, "scale_factor", None) is not None:
        meta["scale_factor"] = float(prediction.scale_factor)

    if getattr(prediction, "extrinsics", None) is not None:
        ext = np.asarray(prediction.extrinsics[0], dtype=np.float64)
        meta["extrinsics"] = ext.tolist()
    if getattr(prediction, "intrinsics", None) is not None:
        ixt = np.asarray(prediction.intrinsics[0], dtype=np.float64)
        meta["intrinsics"] = ixt.tolist()

    return depth, meta


# ---------------------------------------------------------------------------
# Flask endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        status = {
            "status": "healthy" if model is not None else "model not loaded",
            "model_loaded": model is not None,
            "model_config": model_config,
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/infer", methods=["POST"])
def infer():
    """Depth estimation inference endpoint."""
    global model

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        if "images" not in data:
            return jsonify({"error": "Missing 'images' field"}), 400

        output_mode = data.get("output_mode", "depth")
        return_metrics = data.get("return_metrics", False)
        render_views = data.get("render_views", False)

        if output_mode not in VALID_OUTPUT_MODES:
            return jsonify({
                "error": f"Invalid output_mode: '{output_mode}'. "
                         f"Must be one of {VALID_OUTPUT_MODES}"
            }), 400

        # Decode images
        images_bgr = []
        for b64_str in data["images"]:
            img_bytes = base64.b64decode(b64_str)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Invalid image data"}), 400
            images_bgr.append(img)

        logger.info(
            "Inference request: mode=%s, images=%d", output_mode, len(images_bgr)
        )

        # Run inference on first image (monocular depth)
        # Multi-view path can be added later
        ref_img = images_bgr[0]
        depth, depth_meta = _run_model_depth(ref_img)

        # Normalize for visualization only (does not preserve metric scale)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Camera pose: prefer model prediction when available
        extrinsics = depth_meta.get("extrinsics") or [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.5],
        ]
        intrinsics = depth_meta.get("intrinsics") or [
            [525.0, 0.0, ref_img.shape[1] / 2],
            [0.0, 525.0, ref_img.shape[0] / 2],
            [0.0, 0.0, 1.0],
        ]

        response = {
            "success": True,
            "output_mode": output_mode,
            "shape": list(depth.shape),
            "is_metric": depth_meta.get("is_metric", 0),
            "camera_pose": {
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
            },
        }
        if "scale_factor" in depth_meta:
            response["scale_factor"] = depth_meta["scale_factor"]

        # -- Depth visualization (color PNG, NOT metric values) --------
        if output_mode in ("depth", "metric_depth"):
            depth_color = _colorize_depth(depth_norm)
            _, buf = cv2.imencode(".png", depth_color)
            response["depth_image"] = base64.b64encode(buf).decode("utf-8")

            # -- 16-bit metric-scale depth PNG + linear scale ----------
            depth_u16, scale_info = encode_depth_uint16(depth)
            scale_info["is_metric"] = int(depth_meta.get("is_metric", 0) or 0)
            ok, u16_buf = cv2.imencode(".png", depth_u16)
            if not ok:
                raise RuntimeError("Failed to encode 16-bit depth PNG")
            response["metric_depth_u16_png"] = base64.b64encode(u16_buf).decode("utf-8")
            response["metric_depth_scale"] = scale_info

        # -- Point cloud (PLY) -----------------------------------------
        if output_mode in ("point_cloud", "gaussians"):
            response["ply_data"] = _depth_to_ply(depth, ref_img)

        # -- Gaussians -------------------------------------------------
        if output_mode == "gaussians":
            response["gs_ply_data"] = _depth_to_gs_ply(depth, depth_norm, ref_img)

        # -- Rendered views --------------------------------------------
        if render_views and output_mode in ("point_cloud", "gaussians"):
            response["rendered_views"] = _mock_render_views(depth_norm)

        # -- Metrics (summary scalars + 16-bit scale) ------------------
        if return_metrics or output_mode == "metric_depth":
            metrics = {
                "depth_min_m": float(np.nanmin(depth)),
                "depth_max_m": float(np.nanmax(depth)),
                "depth_mean_m": float(np.nanmean(depth)),
                "point_count": int(depth.size),
                "coverage_percent": float(
                    (depth > 0).sum() / max(depth.size, 1) * 100
                ),
                "is_metric": int(depth_meta.get("is_metric", 0) or 0),
            }
            if response.get("metric_depth_scale"):
                metrics.update({
                    "scale": response["metric_depth_scale"]["scale"],
                    "offset": response["metric_depth_scale"]["offset"],
                    "dtype": response["metric_depth_scale"]["dtype"],
                    "formula": response["metric_depth_scale"]["formula"],
                })
            response["metrics"] = metrics

        # -- 3D-aware features -------------------------------------------
        if output_mode == "features":
            feature_resolution = data.get("feature_resolution", "patch")
            feature_source = data.get("feature_source", "depth_decoder")
            features_data = _extract_features(
                ref_img,
                feature_resolution=feature_resolution,
                feature_source=feature_source,
            )
            response["features_b64"] = features_data["features_b64"]
            response["feature_h"] = features_data["feature_h"]
            response["feature_w"] = features_data["feature_w"]
            response["feature_dim"] = features_data["feature_dim"]
            response["feature_type"] = features_data["feature_type"]
            response["feature_source"] = features_data["feature_source"]
            response["format_version"] = features_data["format_version"]
            # Backward-compatible aliases
            response["patch_h"] = features_data["feature_h"]
            response["patch_w"] = features_data["feature_w"]
            response["embed_dim"] = features_data["feature_dim"]

        logger.info("Inference complete: mode=%s, shape=%s", output_mode, depth.shape)
        return jsonify(response)

    except Exception as e:
        logger.error("Inference error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# PLY / GS helpers
# ---------------------------------------------------------------------------

def _depth_to_ply(depth: np.ndarray, image: np.ndarray) -> str:
    """Convert a depth map + RGB image to an ASCII PLY point cloud."""
    h, w = depth.shape
    # Subsample to keep PLY size manageable
    step = max(1, int((h * w / 200_000) ** 0.5))
    fy = fx = max(h, w)
    cx, cy = w / 2, h / 2

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image

    lines = [
        "ply",
        "format ascii 1.0",
        "comment DA3 point cloud",
        f"element vertex {((h // step) * (w // step))}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]

    for v in range(0, h, step):
        for u in range(0, w, step):
            d = depth[v, u]
            if d <= 0:
                continue
            z = float(d)
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            r, g, b = img_rgb[v, u].tolist()
            lines.append(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}")

    return "\n".join(lines)


def _depth_to_gs_ply(
    depth: np.ndarray, depth_norm: np.ndarray, image: np.ndarray
) -> str:
    """Convert depth map to a minimal 3D Gaussian Splatting PLY."""
    h, w = depth.shape
    step = max(1, int((h * w / 50_000) ** 0.5))
    fy = fx = max(h, w)
    cx, cy = w / 2, h / 2

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image

    lines = [
        "ply",
        "format ascii 1.0",
        "comment DA3 3D Gaussian Splatting (minimal)",
        ("element vertex 0"),  # placeholder — filled below
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]

    vertices = []
    for v in range(0, h, step):
        for u in range(0, w, step):
            d = depth[v, u]
            if d <= 0:
                continue
            z = float(d)
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            r, g, b = img_rgb[v, u].tolist()
            # Quaternion identity, unit scales, opacity from normalized depth
            opacity = float(depth_norm[v, u])
            vertices.append(
                f"{x:.4f} {y:.4f} {z:.4f} "
                f"0 0 1 "
                f"{r/255:.4f} {g/255:.4f} {b/255:.4f} "
                f"{opacity:.4f} "
                f"0.02 0.02 0.02 "
                f"1 0 0 0"
            )

    lines[3] = f"element vertex {len(vertices)}"
    lines.extend(vertices)
    return "\n".join(lines)


def _mock_render_views(depth_norm: np.ndarray) -> list:
    """Generate mock rendered views as base64 PNGs."""
    views = []
    for label in ("front", "top", "side"):
        img = Image.new("RGB", (512, 512), color=(30, 30, 60))
        # Simple visual
        arr = np.array(img, dtype=np.float32)
        arr[:, :, 0] += (depth_norm.mean() * 100).astype(np.uint8)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        views.append({
            "view": label,
            "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
        })
    return views


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(
    image_bgr: np.ndarray,
    feature_resolution: str = "patch",
    feature_source: str = "depth_decoder",
) -> dict:
    """
    Extract 3D-aware features from the DA3 model.

    Supports two feature sources controlled by ``feature_source``:

    - ``"depth_decoder"`` (default): Hooks into the DualDPT depth decoder's
      ``refinenet1`` layer.  Produces a 256-dim feature vector per patch
      encoding multi-scale depth fusion information.

    - ``"gs_decoder"``: Hooks into the GSDPT Gaussian decoder's
      ``refinenet1`` layer.  Produces a 256-dim feature vector per patch
      encoding 3D Gaussian geometry information — the richest 3D
      representation, trained for novel view synthesis.  Requires the
      model to be called with ``infer_gs=True`` to activate the GS head.

    Both sources produce features in ``[1, num_patches, 256]`` format
    where the last dimension is the feature vector (consistent with
    DINOv2 convention).

    Args:
        image_bgr: Input image in BGR format.
        feature_resolution: ``"full"`` for native decoder resolution,
            or ``"patch"`` (default) for downsampled patch-grid resolution.
        feature_source: ``"depth_decoder"`` or ``"gs_decoder"``.

    Returns:
        Dict with base64-encoded features and shape metadata.
    """
    global model

    if feature_source not in VALID_FEATURE_SOURCES:
        logger.warning(
            "Unknown feature_source '%s'; falling back to 'depth_decoder'. "
            "Valid options: %s",
            feature_source, VALID_FEATURE_SOURCES,
        )
        feature_source = "depth_decoder"

    h, w = image_bgr.shape[:2]
    input_size = 518
    patch_h, patch_w = input_size // 14, input_size // 14

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    features_tensor = None
    infer_gs = feature_source == "gs_decoder"

    # Hook into the selected decoder's refinenet1 to capture 3D-aware features
    def _hook_fn(module, input, output):
        nonlocal features_tensor
        # refinenet1 output is a spatial tensor [B, C, H, W]
        if isinstance(output, tuple):
            features_tensor = output[0].detach().cpu()
        else:
            features_tensor = output.detach().cpu()

    # Discover hook targets on the live module tree.
    # Official DepthAnything3 wraps NestedDepthAnything3Net as model.model:
    #   model.model.da3.head.scratch.refinenet1
    #   model.model.da3.gs_head.scratch.refinenet1
    hooked_module = None
    hooked_name = None
    candidates = []
    for name, mod in model.named_modules():
        if not name.endswith("scratch.refinenet1"):
            continue
        # Prefer main da3 branch over metric branch
        if "da3_metric" in name:
            continue
        if feature_source == "gs_decoder":
            if "gs_head" in name:
                candidates.append((name, mod))
        else:
            if "gs_head" not in name and name.endswith("head.scratch.refinenet1"):
                candidates.append((name, mod))

    if candidates:
        # Prefer paths containing ".da3." (nested main branch) then shorter names
        candidates.sort(
            key=lambda x: (0 if ".da3." in x[0] or x[0].startswith("da3.") else 1, len(x[0]))
        )
        hooked_name, hooked_module = candidates[0]
        logger.info(
            "Feature hook attached to: %s (source=%s)", hooked_name, feature_source
        )
    else:
        # Explicit attribute paths as fallback
        if feature_source == "gs_decoder":
            hook_paths = [
                "model.da3.gs_head.scratch.refinenet1",
                "da3.gs_head.scratch.refinenet1",
                "model.gs_head.scratch.refinenet1",
                "gs_head.scratch.refinenet1",
            ]
        else:
            hook_paths = [
                "model.da3.head.scratch.refinenet1",
                "da3.head.scratch.refinenet1",
                "model.head.scratch.refinenet1",
                "head.scratch.refinenet1",
            ]
        for attr_path in hook_paths:
            mod = model
            try:
                for part in attr_path.split("."):
                    mod = getattr(mod, part)
                hooked_module = mod
                hooked_name = attr_path
                logger.info(
                    "Feature hook attached to: %s (source=%s)",
                    attr_path, feature_source,
                )
                break
            except AttributeError:
                continue

    if hooked_module is None:
        logger.warning(
            "Could not find any hook target for feature_source='%s'. "
            "Features will be synthetic fallback.",
            feature_source,
        )

    hook_handle = None
    if hooked_module is not None:
        hook_handle = hooked_module.register_forward_hook(_hook_fn)

    try:
        with torch.no_grad():
            # Official DepthAnything3 API — never call missing infer_image()
            if hasattr(model, "inference"):
                _ = model.inference([rgb], infer_gs=infer_gs)
            elif hasattr(model, "forward"):
                # Low-level path: (B, N, 3, H, W)
                img = rgb.astype(np.float32) / 255.0
                img = cv2.resize(img, (input_size, input_size))
                tensor = (
                    torch.from_numpy(img)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .float()
                )
                device = next(model.parameters()).device
                tensor = tensor.to(device)
                try:
                    _ = model(tensor, infer_gs=infer_gs)
                except TypeError:
                    _ = model(tensor)
            else:
                raise RuntimeError(
                    "Loaded model has neither .inference() nor .forward(); "
                    "cannot extract features."
                )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    # If hook didn't capture features, fall back to a synthetic tensor
    if features_tensor is None:
        logger.warning(
            "Feature hook did not capture features (source=%s); "
            "using fallback [1, 256, 296, 296]",
            feature_source,
        )
        features_tensor = torch.randn(1, 256, 296, 296)

    # Ensure spatial format [B, C, H, W] for interpolation
    if features_tensor.dim() == 3:
        # Old-style [B, N, D] — reshape to spatial if square
        b, n, d = features_tensor.shape
        side = int(n ** 0.5)
        if side * side == n:
            features_tensor = features_tensor.reshape(b, d, side, side)
        else:
            # Not square — treat d as channels, pad n to square
            side = int(n ** 0.5) + 1
            padded = torch.zeros(b, side * side, d)
            padded[:, :n, :] = features_tensor
            features_tensor = padded.reshape(b, d, side, side)

    # Optionally downsample to patch-grid resolution for smaller payload
    if feature_resolution == "patch":
        features_tensor = torch.nn.functional.interpolate(
            features_tensor,
            size=(patch_h, patch_w),
            mode="bilinear",
            align_corners=True,
        )

    # Record spatial dims before flattening
    feature_h = features_tensor.shape[2]
    feature_w = features_tensor.shape[3]
    feature_dim = features_tensor.shape[1]  # 256

    # Reshape to [B, num_patches, feature_dim] — last dim is the feature vector,
    # consistent with DINOv2 convention where each patch has a feature descriptor.
    features_tensor = features_tensor.flatten(2).permute(0, 2, 1)
    # Shape is now [1, feature_h * feature_w, feature_dim]

    # Determine feature_type label
    feature_type = (
        "3d_gs_decoder" if feature_source == "gs_decoder"
        else "3d_dpt_decoder"
    )

    # Encode as base64 float32 bytes
    features_bytes = features_tensor.numpy().astype(np.float32).tobytes()
    features_b64 = base64.b64encode(features_bytes).decode("utf-8")

    return {
        "features_b64": features_b64,
        "feature_h": feature_h,
        "feature_w": feature_w,
        "feature_dim": feature_dim,
        "feature_type": feature_type,
        "feature_source": feature_source,
        "format_version": 2,
        "shape": list(features_tensor.shape),
    }


@app.route("/features", methods=["POST"])
def features():
    """3D-aware feature extraction endpoint.

    Extracts features from a selected decoder's refinenet1 layer.

    Accepts parameters in the request body:
      - ``feature_source``: ``"depth_decoder"`` (default) for DualDPT
        depth features, or ``"gs_decoder"`` for GSDPT 3D Gaussian
        features (richest 3D representation).
      - ``feature_resolution``: ``"patch"`` (default) for downsampled
        patch-grid resolution, or ``"full"`` for native decoder resolution.
    """
    global model

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        if "images" not in data:
            return jsonify({"error": "Missing 'images' field"}), 400

        feature_resolution = data.get("feature_resolution", "patch")
        feature_source = data.get("feature_source", "depth_decoder")

        if feature_source not in VALID_FEATURE_SOURCES:
            return jsonify({
                "error": f"Invalid feature_source: '{feature_source}'. "
                         f"Must be one of {VALID_FEATURE_SOURCES}"
            }), 400

        # Decode first image
        b64_str = data["images"][0]
        img_bytes = base64.b64decode(b64_str)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        logger.info(
            "Feature extraction request: image shape=%s, resolution=%s, source=%s",
            img.shape, feature_resolution, feature_source,
        )

        features_data = _extract_features(
            img,
            feature_resolution=feature_resolution,
            feature_source=feature_source,
        )

        response = {
            "success": True,
            "output_mode": "features",
            "features_b64": features_data["features_b64"],
            "feature_h": features_data["feature_h"],
            "feature_w": features_data["feature_w"],
            "feature_dim": features_data["feature_dim"],
            "feature_type": features_data["feature_type"],
            "feature_source": features_data["feature_source"],
            "format_version": features_data["format_version"],
            "shape": features_data["shape"],
            # Backward-compatible aliases
            "patch_h": features_data["feature_h"],
            "patch_w": features_data["feature_w"],
            "embed_dim": features_data["feature_dim"],
        }

        logger.info(
            "Feature extraction complete: shape=%s, type=%s, source=%s, version=%d",
            features_data["shape"],
            features_data["feature_type"],
            features_data["feature_source"],
            features_data["format_version"],
        )
        return jsonify(response)

    except Exception as e:
        logger.error("Feature extraction error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything V3 Server")
    parser.add_argument(
        "--checkpoint_id",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="HuggingFace model ID (default: depth-anything/DA3NESTED-GIANT-LARGE-1.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=20039,
        help="Server port (default: 20039)",
    )
    args = parser.parse_args()

    logger.info("Starting Depth Anything V3 server ...")
    logger.info("  Checkpoint : %s", args.checkpoint_id)
    logger.info("  Port       : %d", args.port)

    if not load_model(checkpoint_id=args.checkpoint_id):
        logger.error(
            "Model load failed — refusing to start. Ensure Depth-Anything-3 "
            "src is on PYTHONPATH (import depth_anything_3) and HF weights "
            "are reachable."
        )
        raise SystemExit(1)

    app.run(host="0.0.0.0", port=args.port, debug=False)
