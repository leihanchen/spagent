"""
Mock Depth Anything V2 service for offline testing.

Referenced by depth_tool.py when use_mock=True with backend="v2".
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDepthService:
    """Mock V2 depth estimation service — no GPU or checkpoint needed."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def infer(self, image_path: str) -> Dict[str, Any]:
        """Mock inference returning a synthetic depth map."""
        try:
            if not os.path.exists(image_path):
                return {"success": False, "error": f"Image file not found: {image_path}"}

            stem = Path(image_path).stem
            w, h = 640, 480

            # Synthetic grayscale depth gradient
            img = Image.new("RGB", (w, h), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)
            for y in range(h):
                val = int(255 * y / h)
                for x in range(w):
                    draw.point((x, y), fill=(val, val // 2, 255 - val))

            draw.rectangle([10, 10, 380, 60], fill=(0, 0, 0, 128))
            draw.text((20, 20), "Depth Anything V2 (mock)", fill=(255, 255, 255))
            draw.text((20, 40), f"Resolution: {w}x{h}", fill=(200, 200, 200))

            output_path = os.path.join(self.output_dir, f"depth_v2_{stem}.png")
            img.save(output_path)

            logger.info("Mock V2 depth: saved %s", output_path)
            return {
                "success": True,
                "output_path": output_path,
                "shape": [h, w],
                "depth_data": None,
            }

        except Exception as e:
            logger.error("Mock V2 depth error: %s", e)
            return {"success": False, "error": str(e)}
