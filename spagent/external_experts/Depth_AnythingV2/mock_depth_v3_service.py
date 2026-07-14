"""
Mock Depth Anything V3 service for offline testing.

Returns synthetic depth maps, point clouds, gaussians, and metrics
without needing a GPU or checkpoint. Matches the same interface as
the real DepthV3Client so the tool can switch transparently.
"""

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDepthV3Service:
    """Mock V3 depth estimation service — no GPU or checkpoint needed."""

    VALID_OUTPUT_MODES = ("depth", "metric_depth", "point_cloud", "gaussians", "features")
    VALID_FEATURE_SOURCES = ("depth_decoder", "gs_decoder")

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy (mock)",
            "model_loaded": True,
            "model_type": "DA3NESTED-GIANT-LARGE-1.1 (mock)",
            "device": "mock",
        }

    # ------------------------------------------------------------------
    # Main entry point — mirrors DepthV3Client.infer
    # ------------------------------------------------------------------

    def infer(
        self,
        image_path: str,
        output_mode: str = "depth",
        return_metrics: bool = False,
        render_views: bool = False,
        feature_source: str = "depth_decoder",
    ) -> Dict[str, Any]:
        """
        Mock inference that returns synthetic results for every output mode.

        Args:
            image_path: Path to input image (single or multi-view list).
            output_mode: One of ``depth``, ``metric_depth``, ``point_cloud``,
                         ``gaussians``, ``features``.
            return_metrics: If True, include spatial metrics in the result.
            render_views: If True, include rendered base64 view images.
            feature_source: ``"depth_decoder"`` or ``"gs_decoder"`` — selects
                            which decoder's features to extract.

        Returns:
            Result dict matching the real client's return shape.
        """
        try:
            if isinstance(image_path, list):
                image_path = image_path[0]

            if not os.path.exists(image_path):
                return {"success": False, "error": f"Image file not found: {image_path}"}

            if output_mode not in self.VALID_OUTPUT_MODES:
                return {
                    "success": False,
                    "error": f"Invalid output_mode: '{output_mode}'. "
                             f"Must be one of {self.VALID_OUTPUT_MODES}",
                }

            stem = Path(image_path).stem
            timestamp = int(time.time())
            base_name = f"depth_v3_{stem}_{timestamp}"

            # Synthetic metric depth (meters) used for 16-bit export
            depth_m, shape = self._mock_metric_depth_array()
            camera_pose = self._mock_camera_pose()

            result: Dict[str, Any] = {
                "success": True,
                "backend": "v3",
                "output_mode": output_mode,
                "shape": shape,
                "camera_pose": camera_pose,
                "is_metric": 1,
            }

            # -- color visualization (not metric values) -----------------
            if output_mode in ("depth", "metric_depth", "point_cloud", "gaussians", "features"):
                depth_path = os.path.join(self.output_dir, f"{base_name}_depth.png")
                self._create_mock_depth_image(depth_path, output_mode, depth_m=depth_m)
                result["output_path"] = depth_path

            # -- 16-bit metric depth PNG + scale (depth modes) -----------
            if output_mode in ("depth", "metric_depth"):
                u16_path = os.path.join(
                    self.output_dir, f"{base_name}_metric_depth_16bit.png"
                )
                scale_info = self._save_metric_depth_16bit(depth_m, u16_path)
                scale_info["is_metric"] = 1
                result["metric_depth_16bit_path"] = u16_path
                result["metric_depth_scale"] = scale_info

            # -- point cloud / gaussians modes ---------------------------
            if output_mode in ("point_cloud", "gaussians"):
                ply_path = os.path.join(self.output_dir, f"{base_name}.ply")
                self._create_mock_ply(ply_path, shape)
                result["ply_path"] = ply_path

            if output_mode == "gaussians":
                gs_ply_path = os.path.join(self.output_dir, f"{base_name}_gs.ply")
                self._create_mock_ply(gs_ply_path, shape)
                result["gs_ply_path"] = gs_ply_path

            # -- rendered views ------------------------------------------
            if render_views and output_mode in ("point_cloud", "gaussians"):
                result["rendered_views"] = self._create_mock_rendered_views(
                    shape, base_name
                )

            # -- features -----------------------------------------------
            if output_mode == "features":
                features_path = os.path.join(self.output_dir, f"{base_name}_features.pth")
                self._create_mock_features(
                    features_path, stem, shape, feature_source=feature_source
                )
                result["features_path"] = features_path

            # -- metrics (summary dict + 16-bit scale) -------------------
            if return_metrics or output_mode == "metric_depth":
                metrics = self._mock_metrics(shape, depth_m=depth_m)
                if result.get("metric_depth_scale"):
                    metrics.update({
                        "scale": result["metric_depth_scale"]["scale"],
                        "offset": result["metric_depth_scale"]["offset"],
                        "dtype": result["metric_depth_scale"]["dtype"],
                        "formula": result["metric_depth_scale"]["formula"],
                        "is_metric": 1,
                    })
                result["metrics"] = metrics

            logger.info(
                "Mock V3 depth: mode=%s, image=%s", output_mode, image_path
            )
            return result

        except Exception as e:
            logger.error("Mock V3 depth error: %s", e)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mock_metric_depth_array(self, h: int = 480, w: int = 640):
        """Synthetic metric depth map in meters: vertical gradient 0.5–12.0 m."""
        col = np.linspace(0.5, 12.0, h, dtype=np.float32).reshape(h, 1)
        depth_m = np.broadcast_to(col, (h, w)).copy()
        return depth_m, [h, w]

    @staticmethod
    def _save_metric_depth_16bit(depth_m: np.ndarray, output_path: str) -> Dict[str, Any]:
        """Encode float meters to uint16 PNG; return scale info for recovery."""
        d_min = float(depth_m.min())
        d_max = float(depth_m.max())
        if d_max > d_min:
            scale = (d_max - d_min) / 65535.0
            offset = d_min
            depth_u16 = np.clip(
                np.round((depth_m - offset) / scale), 0, 65535
            ).astype(np.uint16)
        else:
            scale = 1.0
            offset = d_min
            depth_u16 = np.zeros_like(depth_m, dtype=np.uint16)

        # 16-bit grayscale PNG (Pillow 13+: fromarray(uint16) → I;16)
        Image.fromarray(depth_u16).save(output_path)

        return {
            "scale": scale,
            "offset": offset,
            "depth_min_m": d_min,
            "depth_max_m": d_max,
            "dtype": "uint16",
            "formula": "depth_m = scale * uint16 + offset",
        }

    def _create_mock_depth_image(
        self,
        output_path: str,
        output_mode: str,
        depth_m: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Generate a synthetic depth visualization (color only) and save it."""
        if depth_m is not None:
            h, w = depth_m.shape
            d_norm = (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min() + 1e-8)
            # Simple Spectral-like RGB from normalized depth (vis only)
            r = (255 * (1.0 - d_norm)).astype(np.uint8)
            g = (255 * np.sin(np.pi * d_norm)).astype(np.uint8)
            b = (255 * d_norm).astype(np.uint8)
            arr = np.stack([r, g, b], axis=-1)
            image = Image.fromarray(arr, mode="RGB")
        else:
            w, h = 640, 480
            image = Image.new("RGB", (w, h), color=(50, 50, 50))

        draw = ImageDraw.Draw(image)

        # Overlay text
        mode_label = {
            "depth": "Relative Depth (mock vis)",
            "metric_depth": "Metric Depth (mock vis) — see *_16bit.png for meters",
            "point_cloud": "Point Cloud (mock)",
            "gaussians": "3D Gaussians (mock)",
            "features": "3D DPT Decoder Features (mock)",
        }.get(output_mode, output_mode)

        draw.rectangle([10, 10, 520, 80], fill=(0, 0, 0, 128))
        draw.text((20, 20), f"DA3 V3 Mock — {mode_label}", fill=(255, 255, 255))
        draw.text((20, 45), f"Resolution: {w}x{h}", fill=(200, 200, 200))

        # Draw some fake objects with depth labels
        objects = [
            (120, 180, 80, 60, "Chair: 2.1m"),
            (350, 150, 120, 80, "Table: 3.5m"),
            (220, 300, 60, 40, "Cup: 1.2m"),
        ]
        for ox, oy, ow, oh, label in objects:
            draw.rectangle([ox, oy, ox + ow, oy + oh], outline=(255, 200, 100), width=2)
            draw.text((ox + 5, oy + oh + 5), label, fill=(255, 255, 200))

        image.save(output_path)
        return [h, w]

    def _create_mock_ply(self, ply_path: str, shape: List[int]) -> None:
        """Write a minimal synthetic PLY file."""
        h, w = shape
        header = f"""ply
format ascii 1.0
comment Mock DA3 V3 point cloud
element vertex {h * w // 100}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(ply_path, "w") as f:
            f.write(header)
            for i in range(h * w // 100):
                x = (i % (w // 10)) * 0.1
                y = (i // (w // 10)) * 0.1
                z = np.sin(x * 2) * np.cos(y * 2) * 2.0
                r, g, b = int(128 + 64 * np.sin(x)), int(128 + 64 * np.cos(y)), 200
                f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")

    def _create_mock_rendered_views(
        self, shape: List[int], base_name: str
    ) -> List[Dict[str, Any]]:
        """Generate mock rendered views (front, top, side) as base64 PNGs."""
        views = []
        for view_name in ("front", "top", "side"):
            img = Image.new("RGB", (512, 512), color=(40, 40, 80))
            draw = ImageDraw.Draw(img)

            # Draw some 3D-ish geometry
            center = 256
            draw.ellipse([center - 80, center - 60, center + 80, center + 60],
                         fill=(80, 80, 180), outline=(150, 150, 255), width=3)
            draw.rectangle([center - 40, center - 100, center + 40, center - 30],
                           fill=(180, 80, 80), outline=(255, 150, 150), width=2)

            if view_name == "top":
                draw.text((200, 20), "Top View", fill=(255, 255, 255))
            elif view_name == "side":
                draw.text((200, 20), "Side View", fill=(255, 255, 255))
            else:
                draw.text((200, 20), "Front View", fill=(255, 255, 255))

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            views.append({
                "view": view_name,
                "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
            })

        return views

    def _mock_camera_pose(self) -> Dict[str, Any]:
        """Return a mock identity-ish camera pose."""
        return {
            "extrinsics": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.5],
            ],
            "intrinsics": [
                [525.0, 0.0, 320.0],
                [0.0, 525.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
        }

    def _mock_metrics(
        self, shape: List[int], depth_m: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Return synthetic spatial metrics (summary dict)."""
        h, w = shape
        if depth_m is not None:
            return {
                "depth_min_m": float(depth_m.min()),
                "depth_max_m": float(depth_m.max()),
                "depth_mean_m": float(depth_m.mean()),
                "point_count": int(depth_m.size),
                "coverage_percent": float((depth_m > 0).sum() / depth_m.size * 100),
                "is_metric": 1,
            }
        return {
            "depth_min_m": 0.5,
            "depth_max_m": 12.0,
            "depth_mean_m": 6.25,
            "point_count": h * w,
            "coverage_percent": 100.0,
            "is_metric": 1,
        }

    def _create_mock_features(
        self,
        output_path: str,
        image_id: str,
        shape: List[int],
        feature_source: str = "depth_decoder",
    ) -> None:
        """Save a synthetic 3D-aware feature tensor as .pth.

        Produces a feature tensor [1, num_patches, 256] where the last
        dimension is the 3D-aware feature vector (consistent with DINOv2
        convention).

        Args:
            output_path: Path to save the .pth file.
            image_id: Image identifier string.
            shape: Image shape [H, W].
            feature_source: ``"depth_decoder"`` for DualDPT depth features,
                            or ``"gs_decoder"`` for GSDPT 3D Gaussian features.
        """
        import torch

        if feature_source not in self.VALID_FEATURE_SOURCES:
            logger.warning(
                "Unknown feature_source '%s'; defaulting to 'depth_decoder'",
                feature_source,
            )
            feature_source = "depth_decoder"

        h, w = shape
        feature_dim = 256
        feature_h = h // 14
        feature_w = w // 14
        num_patches = feature_h * feature_w

        # Synthetic feature tensor: [1, num_patches, feature_dim]
        features = np.random.randn(1, num_patches, feature_dim).astype(
            np.float32
        )

        feature_type = (
            "3d_gs_decoder" if feature_source == "gs_decoder"
            else "3d_dpt_decoder"
        )

        data = {
            "image_id": image_id,
            "features": torch.from_numpy(features),
            "feature_h": feature_h,
            "feature_w": feature_w,
            "feature_dim": feature_dim,
            "feature_type": feature_type,
            "feature_source": feature_source,
            "format_version": 2,
        }
        torch.save(data, output_path)
        logger.info("Mock V3 features saved: %s (source=%s, type=%s)", output_path, feature_source, feature_type)
