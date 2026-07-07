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

    VALID_OUTPUT_MODES = ("depth", "metric_depth", "point_cloud", "gaussians")

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
    ) -> Dict[str, Any]:
        """
        Mock inference that returns synthetic results for every output mode.

        Args:
            image_path: Path to input image (single or multi-view list).
            output_mode: One of ``depth``, ``metric_depth``, ``point_cloud``,
                         ``gaussians``.
            return_metrics: If True, include spatial metrics in the result.
            render_views: If True, include rendered base64 view images.

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

            # -- always generate the primary depth visualization ----------
            depth_path = os.path.join(self.output_dir, f"{base_name}_depth.png")
            shape = self._create_mock_depth_image(depth_path, output_mode)
            camera_pose = self._mock_camera_pose()

            result: Dict[str, Any] = {
                "success": True,
                "backend": "v3",
                "output_mode": output_mode,
                "output_path": depth_path,
                "shape": shape,
                "camera_pose": camera_pose,
            }

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

            # -- metrics -------------------------------------------------
            if return_metrics:
                result["metrics"] = self._mock_metrics(shape)

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

    def _create_mock_depth_image(
        self, output_path: str, output_mode: str
    ) -> List[int]:
        """Generate a synthetic depth visualization and save it."""
        w, h = 640, 480

        if output_mode == "metric_depth":
            # Blue gradient with metric annotations
            image = Image.new("RGB", (w, h), color=(30, 30, 80))
        else:
            image = Image.new("RGB", (w, h), color=(50, 50, 50))

        draw = ImageDraw.Draw(image)

        # Fake depth gradient bars
        for y in range(h):
            val = int(255 * y / h)
            for x in range(w):
                if output_mode == "metric_depth":
                    color = (val // 2, val // 3, 255 - val // 3)
                else:
                    color = (val, val // 2, 255 - val)
                draw.point((x, y), fill=color)

        # Overlay text
        mode_label = {
            "depth": "Relative Depth (mock)",
            "metric_depth": "Metric Depth (mock) — values in meters",
            "point_cloud": "Point Cloud (mock)",
            "gaussians": "3D Gaussians (mock)",
        }.get(output_mode, output_mode)

        draw.rectangle([10, 10, 400, 80], fill=(0, 0, 0, 128))
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

    def _mock_metrics(self, shape: List[int]) -> Dict[str, Any]:
        """Return synthetic spatial metrics."""
        h, w = shape
        return {
            "depth_min_m": 0.51,
            "depth_max_m": 12.34,
            "depth_mean_m": 3.17,
            "point_count": h * w,
            "coverage_percent": 98.7,
        }
