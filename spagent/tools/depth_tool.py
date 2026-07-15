"""
Depth Estimation Tool

This module contains the DepthEstimationTool that wraps
Depth-AnythingV2 and Depth-Anything-V3 functionality for the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Union

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class DepthEstimationTool(Tool):
    """Tool for depth estimation using Depth-Anything V2 or V3.

    Backend selection:
        - ``backend="v2"`` (default): Depth-Anything V2 — relative depth only.
        - ``backend="v3"``: DA3NESTED-GIANT-LARGE-1.1 — metric depth,
          point clouds, 3D gaussians, and spatial metrics.
    """

    BACKENDS = ("v2", "v3")
    V3_OUTPUT_MODES = ("depth", "metric_depth", "point_cloud", "gaussians", "features")

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://10.8.131.51:20019",
        backend: str = "v2",
    ):
        """
        Initialize depth estimation tool.

        Args:
            use_mock: Whether to use mock client for testing.
            server_url: URL of the depth estimation server.  For V3 the
                        default is ``http://127.0.0.1:20039`` unless
                        overridden.
            backend: ``"v2"`` (default) or ``"v3"``.
        """
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Must be one of {self.BACKENDS}"
            )

        # --- description varies by backend ---
        if backend == "v3":
            description = (
                "Generate depth maps, metric depth (meters), point clouds, 3D "
                "Gaussian splats, or 3D-aware features from one or more images using "
                "Depth Anything V3 (DA3NESTED-GIANT-LARGE-1.1). Supports "
                "multi-view input for better geometry.\n\n"
                "When to use: spatial relationship questions (closer/farther), "
                "metric distance estimation, 3D scene reconstruction, occlusion "
                "reasoning, scene layout analysis, or extracting 3D-aware DPT "
                "decoder features for downstream tasks.\n"
                "When NOT to use: object naming/counting (prefer detection), "
                "pixel masks (prefer segmentation), or novel camera viewpoints "
                "(prefer pi3/pi3x).\n"
                "Example: call with image_path='scene.jpg' and output_mode='metric_depth' "
                "to get absolute distance information, or output_mode='features' "
                "to extract 3D-aware spatial features from the DPT depth decoder."
            )
        else:
            description = (
                "Generate a monocular depth map for one input image to analyze relative depth, "
                "near/far ordering, and 3D layout cues.\n\n"
                "When to use: spatial relationship questions (closer/farther), occlusion reasoning, "
                "scene layout, or when depth ordering helps answer the question.\n"
                "When NOT to use: object naming/counting (prefer detection), pixel masks (prefer segmentation), "
                "or novel camera viewpoints (prefer pi3/pi3x).\n"
                "Example: call with image_path='scene.jpg' to compare which object is nearer to the camera."
            )

        super().__init__(name="depth_estimation_tool", description=description)

        self.use_mock = use_mock
        self.backend = backend

        # V3 defaults to its own port unless explicitly overridden
        if backend == "v3" and server_url == "http://10.8.131.51:20019":
            self.server_url = "http://127.0.0.1:20039"
        else:
            self.server_url = server_url

        self._client = None
        self._init_client()

    # ------------------------------------------------------------------
    # Client init
    # ------------------------------------------------------------------

    def _init_client(self):
        """Initialize the appropriate depth client (V2 or V3)."""
        if self.backend == "v3":
            self._init_v3_client()
        else:
            self._init_v2_client()

    def _init_v2_client(self):
        if self.use_mock:
            try:
                from external_experts.Depth_AnythingV2.mock_depth_service import (
                    MockDepthService,
                )
                self._client = MockDepthService()
                logger.info("Using mock V2 depth estimation service")
            except ImportError as e:
                logger.error("Failed to import mock V2 depth service: %s", e)
                raise
        else:
            try:
                from external_experts.Depth_AnythingV2.depth_client import DepthClient
                self._client = DepthClient(server_url=self.server_url)
                logger.info("Using real V2 depth service at %s", self.server_url)
            except ImportError as e:
                logger.error("Failed to import real V2 depth client: %s", e)
                raise

    def _init_v3_client(self):
        if self.use_mock:
            try:
                from external_experts.Depth_AnythingV2.mock_depth_v3_service import (
                    MockDepthV3Service,
                )
                self._client = MockDepthV3Service()
                logger.info("Using mock V3 depth estimation service")
            except ImportError as e:
                logger.error("Failed to import mock V3 depth service: %s", e)
                raise
        else:
            try:
                from external_experts.Depth_AnythingV2.depth_v3_client import (
                    DepthV3Client,
                )
                self._client = DepthV3Client(server_url=self.server_url)
                logger.info("Using real V3 depth service at %s", self.server_url)
            except ImportError as e:
                logger.error("Failed to import real V3 depth client: %s", e)
                raise

    # ------------------------------------------------------------------
    # Parameters schema
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema — varies by backend."""
        if self.backend == "v3":
            return self._v3_parameters()
        return self._v2_parameters()

    @staticmethod
    def _v2_parameters() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image for depth estimation.",
                }
            },
            "required": ["image_path"],
        }

    def _v3_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": (
                        "Path to the input image, or list of image paths for "
                        "multi-view depth estimation. Multi-view improves pose "
                        "and geometry accuracy."
                    ),
                },
                "output_mode": {
                    "type": "string",
                    "enum": list(self.V3_OUTPUT_MODES),
                    "description": (
                        "Output mode: 'depth' — relative depth map (color vis + float32 .npy); "
                        "'metric_depth' — absolute depth in meters (float32 .npy, color is vis only); "
                        "'point_cloud' — PLY point cloud; "
                        "'gaussians' — 3D Gaussian Splatting (GS PLY + rendered views); "
                        "'features' — 3D-aware features from depth or Gaussian decoder "
                        "(256-dim per patch, controlled by feature_source)."
                    ),
                    "default": "depth",
                },
                "feature_source": {
                    "type": "string",
                    "enum": ["depth_decoder", "gs_decoder"],
                    "description": (
                        "When output_mode='features', selects which decoder to extract "
                        "features from: 'depth_decoder' (default) for DualDPT depth "
                        "features, or 'gs_decoder' for GSDPT 3D Gaussian features "
                        "(richest 3D representation, trained for novel view synthesis)."
                    ),
                    "default": "depth_decoder",
                },
                "return_metrics": {
                    "type": "boolean",
                    "description": (
                        "When true, include computed spatial metrics: depth "
                        "min/max/mean (meters), point count, coverage percent."
                    ),
                    "default": False,
                },
                "render_views": {
                    "type": "boolean",
                    "description": (
                        "When true, render the 3D output from canonical "
                        "viewpoints (front, top, side) as base64 images. "
                        "Only meaningful with point_cloud or gaussians modes."
                    ),
                    "default": False,
                },
            },
            "required": ["image_path"],
        }

    # ------------------------------------------------------------------
    # call()
    # ------------------------------------------------------------------

    def call(
        self,
        image_path: Union[str, List[str]],
        output_mode: str = "depth",
        return_metrics: bool = False,
        render_views: bool = False,
        feature_source: str = "depth_decoder",
    ) -> Dict[str, Any]:
        """
        Execute depth estimation.

        Args:
            image_path: Path to input image (V2: single string; V3: string or list).
            output_mode: V3 only — one of ``"depth"``, ``"metric_depth"``,
                         ``"point_cloud"``, ``"gaussians"``, ``"features"``.
            return_metrics: V3 only — include spatial metrics.
            render_views: V3 only — include rendered view images.
            feature_source: V3 only — ``"depth_decoder"`` or ``"gs_decoder"``.
                            Selects which decoder to extract features from
                            when output_mode='features'.

        Returns:
            Depth estimation result dictionary.
        """
        if self.backend == "v3":
            return self._call_v3(
                image_path, output_mode, return_metrics, render_views, feature_source
            )
        return self._call_v2(image_path)

    def _call_v2(self, image_path: str) -> Dict[str, Any]:
        """Execute V2 depth estimation (original behaviour)."""
        try:
            logger.info("Running V2 depth estimation on: %s", image_path)

            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}",
                }

            if hasattr(self._client, "infer"):
                result = self._client.infer(image_path)
            else:
                result = self._client.process_image(image_path)

            if result and result.get("success"):
                logger.info("V2 depth estimation completed successfully")
                return {
                    "success": True,
                    "backend": "v2",
                    "result": result,
                    "output_path": result.get("output_path"),
                    "shape": result.get("shape"),
                    "depth_data": result.get("depth_data"),
                }
            else:
                error_msg = (
                    result.get("error", "Unknown error")
                    if result
                    else "No result returned"
                )
                logger.error("V2 depth estimation failed: %s", error_msg)
                return {
                    "success": False,
                    "error": f"Depth estimation failed: {error_msg}",
                }

        except Exception as e:
            logger.error("V2 depth estimation tool error: %s", e)
            return {"success": False, "error": str(e)}

    def _call_v3(
        self,
        image_path: Union[str, List[str]],
        output_mode: str,
        return_metrics: bool,
        render_views: bool,
        feature_source: str = "depth_decoder",
    ) -> Dict[str, Any]:
        """Execute V3 depth estimation."""
        try:
            # Validate image existence
            paths = [image_path] if isinstance(image_path, str) else list(image_path)
            for p in paths:
                if not Path(p).exists():
                    return {
                        "success": False,
                        "error": f"Image file not found: {p}",
                    }

            if output_mode not in self.V3_OUTPUT_MODES:
                return {
                    "success": False,
                    "error": (
                        f"Invalid output_mode: '{output_mode}'. "
                        f"Must be one of {self.V3_OUTPUT_MODES}"
                    ),
                }

            logger.info(
                "Running V3 depth estimation: mode=%s, images=%d, feature_source=%s",
                output_mode,
                len(paths),
                feature_source,
            )

            # Delegate to client (or mock)
            use_first = paths[0]  # client handles multi-view internally
            result = self._client.infer(
                image_path=use_first if len(paths) == 1 else paths,
                output_mode=output_mode,
                return_metrics=return_metrics,
                render_views=render_views,
                feature_source=feature_source,
            )

            if result and result.get("success"):
                logger.info("V3 depth estimation completed successfully")
                return result
            else:
                error_msg = (
                    result.get("error", "Unknown error")
                    if result
                    else "No result returned"
                )
                logger.error("V3 depth estimation failed: %s", error_msg)
                return {
                    "success": False,
                    "error": f"Depth estimation failed: {error_msg}",
                }

        except Exception as e:
            logger.error("V3 depth estimation tool error: %s", e)
            return {"success": False, "error": str(e)}
