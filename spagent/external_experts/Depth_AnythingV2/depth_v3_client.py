"""
HTTP client for Depth Anything V3 server.

Sends images to the V3 Flask server and returns depth maps,
metric depth, point clouds, 3D gaussians, and spatial metrics.
"""

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthV3Client:
    """Client for the Depth Anything V3 (DA3NESTED-GIANT-LARGE-1.1) server."""

    VALID_OUTPUT_MODES = ("depth", "metric_depth", "point_cloud", "gaussians")

    def __init__(self, server_url: str = "http://127.0.0.1:20039", output_dir: str = "outputs"):
        self.server_url = server_url.rstrip("/")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self) -> Optional[Dict[str, Any]]:
        """Check server health status."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("V3 health check failed: %s", e)
            return None

    def infer(
        self,
        image_path: str,
        output_mode: str = "depth",
        return_metrics: bool = False,
        render_views: bool = False,
    ) -> Dict[str, Any]:
        """
        Send image(s) to the V3 server for depth estimation.

        Args:
            image_path: Path to input image, or list of paths for multi-view.
            output_mode: ``"depth"`` | ``"metric_depth"`` | ``"point_cloud"`` | ``"gaussians"``.
            return_metrics: Include spatial metrics in response.
            render_views: Include base64 rendered view images.

        Returns:
            Result dict matching the tool's expected return shape.
        """
        try:
            if output_mode not in self.VALID_OUTPUT_MODES:
                return {
                    "success": False,
                    "error": f"Invalid output_mode: '{output_mode}'. "
                             f"Must be one of {self.VALID_OUTPUT_MODES}",
                }

            # Normalize to list
            paths = [image_path] if isinstance(image_path, str) else list(image_path)

            # Read and encode images
            images_b64 = []
            for p in paths:
                if not os.path.exists(p):
                    return {"success": False, "error": f"Image file not found: {p}"}
                img = cv2.imread(p)
                if img is None:
                    return {"success": False, "error": f"Cannot read image: {p}"}
                _, buf = cv2.imencode(".jpg", img)
                images_b64.append(base64.b64encode(buf).decode("utf-8"))

            # Send request
            payload = {
                "images": images_b64,
                "output_mode": output_mode,
                "return_metrics": return_metrics,
                "render_views": render_views,
            }

            logger.info(
                "Sending V3 infer request: mode=%s, images=%d, url=%s",
                output_mode, len(images_b64), self.server_url,
            )
            resp = requests.post(
                f"{self.server_url}/infer",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            resp.raise_for_status()
            server_result = resp.json()

            if not server_result.get("success"):
                return {
                    "success": False,
                    "error": server_result.get("error", "Server returned failure"),
                }

            # Save and normalize the result
            return self._process_server_result(server_result, paths[0])

        except requests.exceptions.ConnectionError:
            return {"success": False, "error": f"V3 server unreachable: {self.server_url}"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "V3 server request timed out"}
        except Exception as e:
            logger.error("V3 client error: %s", e)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_server_result(
        self, server_result: Dict[str, Any], first_image_path: str
    ) -> Dict[str, Any]:
        """Save server outputs to disk and build the normalized result dict."""
        stem = Path(first_image_path).stem
        timestamp = int(time.time())
        base_name = f"depth_v3_{stem}_{timestamp}"

        result: Dict[str, Any] = {
            "success": True,
            "backend": "v3",
            "output_mode": server_result.get("output_mode", "depth"),
            "shape": server_result.get("shape"),
            "camera_pose": server_result.get("camera_pose"),
        }

        # -- depth visualization ---------------------------------------
        if server_result.get("depth_image"):
            depth_path = os.path.join(self.output_dir, f"{base_name}_depth.png")
            self._save_b64_image(server_result["depth_image"], depth_path)
            result["output_path"] = depth_path

        # -- PLY data --------------------------------------------------
        if server_result.get("ply_data"):
            ply_path = os.path.join(self.output_dir, f"{base_name}.ply")
            with open(ply_path, "w") as f:
                f.write(server_result["ply_data"])
            result["ply_path"] = ply_path

        # -- GS PLY data -----------------------------------------------
        if server_result.get("gs_ply_data"):
            gs_ply_path = os.path.join(self.output_dir, f"{base_name}_gs.ply")
            with open(gs_ply_path, "w") as f:
                f.write(server_result["gs_ply_data"])
            result["gs_ply_path"] = gs_ply_path

        # -- rendered views --------------------------------------------
        if server_result.get("rendered_views"):
            result["rendered_views"] = server_result["rendered_views"]

        # -- metrics ---------------------------------------------------
        if server_result.get("metrics"):
            result["metrics"] = server_result["metrics"]

        logger.info("V3 client: result saved, mode=%s", result.get("output_mode"))
        return result

    @staticmethod
    def _save_b64_image(b64_data: str, output_path: str) -> None:
        """Decode a base64 image and write it to disk."""
        img_bytes = base64.b64decode(b64_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(output_path, img)
