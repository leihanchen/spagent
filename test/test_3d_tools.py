"""
Unit tests for Depth Anything V3 Tool and other 3D tools.

This file contains pytest-based unit tests for:
- DepthEstimationTool (V2 and V3 backends)
- VGGTTool
- Pi3Tool
- Pi3XTool

All tests run in mock mode by default, so no GPU or server is required.
For real-service tests, set the appropriate environment variable:

    DEPTH_V3_REAL_TEST=1  pytest -q test/test_3d_tools.py
    VGGT_REAL_TEST=1      pytest -q test/test_3d_tools.py
    PI3_REAL_TEST=1       pytest -q test/test_3d_tools.py

Usage:
    # Run all mock tests
    pytest -q test/test_3d_tools.py

    # Run specific test class
    pytest -q test/test_3d_tools.py::TestDepthV3Tool -v

    # Run with real server (requires GPU and running server)
    DEPTH_V3_REAL_TEST=1 pytest -q test/test_3d_tools.py::TestDepthV3Tool::test_v3_real_service
"""

import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# =============================================================================
# DepthEstimationTool - V2 Backend
# =============================================================================

class TestDepthV2Tool:
    """Tests for DepthEstimationTool with backend='v2'."""

    def test_v2_tool_instantiation(self):
        """V2 backend should instantiate successfully."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v2")
        assert tool.name == "depth_estimation_tool"
        assert tool.backend == "v2"

    def test_v2_parameters_schema(self):
        """V2 parameters should only require image_path (string)."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v2")
        params = tool.parameters

        assert params["type"] == "object"
        assert "image_path" in params["properties"]
        assert params["properties"]["image_path"]["type"] == "string"
        assert params["required"] == ["image_path"]

    def test_v2_mock_depth_success(self):
        """V2 mock mode should return success with output_path."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v2")
        result = tool.call(image_path="assets/example.png")

        assert result["success"] is True
        assert "output_path" in result
        assert result["output_path"] is not None
        assert os.path.exists(result["output_path"])

    def test_v2_missing_image_error(self):
        """V2 should return error for missing image."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v2")
        result = tool.call(image_path="nonexistent_image.jpg")

        assert result["success"] is False
        assert "Image file not found" in result["error"]


# =============================================================================
# DepthEstimationTool - V3 Backend
# =============================================================================

class TestDepthV3Tool:
    """Tests for DepthEstimationTool with backend='v3'."""

    def test_v3_tool_instantiation(self):
        """V3 backend should instantiate successfully."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        assert tool.name == "depth_estimation_tool"
        assert tool.backend == "v3"

    def test_v3_parameters_schema(self):
        """V3 parameters should include output_mode, return_metrics, render_views."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        params = tool.parameters

        assert params["type"] == "object"
        assert "image_path" in params["properties"]
        assert "output_mode" in params["properties"]
        assert "return_metrics" in params["properties"]
        assert "render_views" in params["properties"]
        assert params["required"] == ["image_path"]

        # Check output_mode enum
        assert params["properties"]["output_mode"]["enum"] == [
            "depth", "metric_depth", "point_cloud", "gaussians", "features"
        ]

    def test_v3_invalid_backend_error(self):
        """Invalid backend should raise ValueError."""
        from spagent.tools import DepthEstimationTool

        with pytest.raises(ValueError) as exc_info:
            DepthEstimationTool(use_mock=True, backend="v4")

        assert "Unknown backend" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # Output mode tests
    # -------------------------------------------------------------------------

    def test_v3_depth_mode(self):
        """V3 depth mode should return colored depth visualization."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        result = tool.call(image_path="assets/example.png", output_mode="depth")

        assert result["success"] is True
        assert result["backend"] == "v3"
        assert result["output_mode"] == "depth"
        assert "output_path" in result
        assert os.path.exists(result["output_path"])
        assert "camera_pose" in result
        assert "extrinsics" in result["camera_pose"]
        assert "intrinsics" in result["camera_pose"]

    def test_v3_metric_depth_mode(self):
        """V3 metric_depth mode should return float32 metric depth .npy."""
        import numpy as np

        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        result = tool.call(
            image_path="assets/example.png",
            output_mode="metric_depth",
            return_metrics=True,
        )

        assert result["success"] is True
        assert result["output_mode"] == "metric_depth"
        assert "metrics" in result
        assert "depth_min_m" in result["metrics"]
        assert "depth_max_m" in result["metrics"]
        assert "depth_mean_m" in result["metrics"]
        assert "scale" not in result["metrics"]
        assert "offset" not in result["metrics"]
        # Float32 HxW depth in meters (PNG cannot store IEEE floats)
        assert "metric_depth_path" in result
        assert result["metric_depth_path"] is not None
        assert os.path.exists(result["metric_depth_path"])
        assert result["metric_depth_path"].endswith(".npy")

        depth = np.load(result["metric_depth_path"])
        assert depth.dtype == np.float32
        assert depth.ndim == 2
        assert abs(float(depth.min()) - float(result["metrics"]["depth_min_m"])) < 1e-5
        assert abs(float(depth.max()) - float(result["metrics"]["depth_max_m"])) < 1e-5

    def test_v3_point_cloud_mode(self):
        """V3 point_cloud mode should return PLY file."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        result = tool.call(image_path="assets/example.png", output_mode="point_cloud")

        assert result["success"] is True
        assert result["output_mode"] == "point_cloud"
        assert "ply_path" in result
        assert result["ply_path"] is not None
        assert os.path.exists(result["ply_path"])

    def test_v3_gaussians_mode(self):
        """V3 gaussians mode should return PLY and GS PLY files."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        result = tool.call(
            image_path="assets/example.png",
            output_mode="gaussians",
            return_metrics=True,
            render_views=True,
        )

        assert result["success"] is True
        assert result["output_mode"] == "gaussians"
        assert "ply_path" in result
        assert "gs_ply_path" in result
        assert os.path.exists(result["ply_path"])
        assert os.path.exists(result["gs_ply_path"])
        assert "rendered_views" in result
        assert len(result["rendered_views"]) == 3  # front, top, side
        assert result["rendered_views"][0]["view"] in ("front", "top", "side")

    def test_v3_features_depth_decoder(self):
        """V3 features with depth_decoder should return .pth with 3D DPT decoder features."""
        import torch

        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        result = tool.call(
            image_path="assets/example.png",
            output_mode="features",
            feature_source="depth_decoder",
        )

        assert result["success"] is True
        assert result["output_mode"] == "features"
        assert "features_path" in result
        assert result["features_path"] is not None
        assert os.path.exists(result["features_path"])

        # Load and verify the .pth file
        data = torch.load(result["features_path"], map_location="cpu", weights_only=False)
        assert "image_id" in data
        assert "features" in data
        assert "feature_h" in data
        assert "feature_w" in data
        assert "feature_dim" in data
        assert "feature_type" in data
        assert "feature_source" in data
        assert "format_version" in data
        # Format: [1, num_patches, feature_dim] — last dim is feature vector
        assert data["features"].dim() == 3
        assert data["features"].shape[0] == 1  # batch dim
        assert data["features"].shape[2] == 256  # feature dim
        assert data["feature_dim"] == 256
        assert data["feature_type"] == "3d_dpt_decoder"
        assert data["feature_source"] == "depth_decoder"
        assert data["format_version"] == 2
        # num_patches = feature_h * feature_w
        assert data["features"].shape[1] == data["feature_h"] * data["feature_w"]

    def test_v3_features_gs_decoder(self):
        """V3 features with gs_decoder should return .pth with 3D GS decoder features."""
        import torch

        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        result = tool.call(
            image_path="assets/example.png",
            output_mode="features",
            feature_source="gs_decoder",
        )

        assert result["success"] is True
        assert result["output_mode"] == "features"
        assert "features_path" in result
        assert result["features_path"] is not None
        assert os.path.exists(result["features_path"])

        # Load and verify the .pth file
        data = torch.load(result["features_path"], map_location="cpu", weights_only=False)
        assert "features" in data
        assert "feature_type" in data
        assert "feature_source" in data
        # Same shape convention as depth_decoder: [1, num_patches, 256]
        assert data["features"].dim() == 3
        assert data["features"].shape[0] == 1
        assert data["features"].shape[2] == 256
        assert data["feature_dim"] == 256
        assert data["feature_type"] == "3d_gs_decoder"
        assert data["feature_source"] == "gs_decoder"
        assert data["format_version"] == 2

    def test_v3_features_spatial_structure(self):
        """3D features should vary across patches (not uniform)."""
        import torch

        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        result = tool.call(
            image_path="assets/example.png",
            output_mode="features",
        )

        assert result["success"] is True
        data = torch.load(result["features_path"], map_location="cpu", weights_only=False)
        features = data["features"]
        # Features should vary across patches
        assert features[0, 0].norm() != features[0, -1].norm()
        # Different patches should have different feature vectors
        assert not torch.allclose(features[0, 0, :], features[0, -1, :])

    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------

    def test_v3_invalid_output_mode_error(self):
        """V3 should return error for invalid output_mode."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        result = tool.call(image_path="assets/example.png", output_mode="invalid_mode")

        assert result["success"] is False
        assert "Invalid output_mode" in result["error"]

    def test_v3_missing_image_error(self):
        """V3 should return error for missing image."""
        from spagent.tools import DepthEstimationTool

        tool = DepthEstimationTool(use_mock=True, backend="v3")
        result = tool.call(image_path="nonexistent.jpg", output_mode="depth")

        assert result["success"] is False
        assert "Image file not found" in result["error"]

    # -------------------------------------------------------------------------
    # Real server test (optional)
    # -------------------------------------------------------------------------

    @pytest.mark.skipif(
        os.environ.get("DEPTH_V3_REAL_TEST") != "1",
        reason="Set DEPTH_V3_REAL_TEST=1 to run against a live V3 server.",
    )
    def test_v3_real_service(self):
        """Optional smoke test for a live V3 server."""
        from spagent.tools import DepthEstimationTool

        server_url = os.environ.get("DEPTH_V3_SERVER_URL", "http://127.0.0.1:20039")
        tool = DepthEstimationTool(use_mock=False, backend="v3", server_url=server_url)

        result = tool.call(
            image_path="assets/example.png",
            output_mode="metric_depth",
            return_metrics=True,
        )

        assert result["success"] is True
        assert os.path.exists(result["output_path"])


# =============================================================================
# VGGTTool
# =============================================================================

class TestVGGTTool:
    """Tests for VGGT 3D reconstruction tool."""

    def test_vggt_tool_instantiation(self):
        """VGGTTool should instantiate successfully."""
        from spagent.tools import VGGTTool

        tool = VGGTTool(use_mock=True)
        assert tool.name == "vggt_tool"

    def test_vggt_parameters_schema(self):
        """VGGT parameters should include image_path, angles, camera_view."""
        from spagent.tools import VGGTTool

        tool = VGGTTool(use_mock=True)
        params = tool.parameters

        assert params["type"] == "object"
        assert "image_path" in params["properties"]
        assert "azimuth_angle" in params["properties"]
        assert "elevation_angle" in params["properties"]
        assert "rotation_reference_camera" in params["properties"]
        assert "camera_view" in params["properties"]

    def test_vggt_mock_reconstruction(self):
        """VGGT mock mode should return point cloud visualization."""
        from spagent.tools import VGGTTool

        tool = VGGTTool(use_mock=True)
        result = tool.call(
            image_path=["assets/example.png"],
            azimuth_angle=45,
            elevation_angle=30,
        )

        assert result["success"] is True
        assert "output_path" in result
        assert os.path.exists(result["output_path"])
        assert "points_count" in result
        assert result["points_count"] > 0

    def test_vggt_angle_validation(self):
        """VGGT should validate angle ranges."""
        from spagent.tools import VGGTTool

        tool = VGGTTool(use_mock=True)

        # Out of range azimuth
        result = tool.call(image_path=["assets/example.png"], azimuth_angle=200)
        assert result["success"] is False
        assert "azimuth_angle" in result["error"]

        # Out of range elevation
        result = tool.call(image_path=["assets/example.png"], elevation_angle=100)
        assert result["success"] is False
        assert "elevation_angle" in result["error"]

    def test_vggt_missing_image_error(self):
        """VGGT should return error for missing image."""
        from spagent.tools import VGGTTool

        tool = VGGTTool(use_mock=True)
        result = tool.call(image_path=["nonexistent.jpg"])

        assert result["success"] is False
        assert "Image file not found" in result["error"]

    @pytest.mark.skipif(
        os.environ.get("VGGT_REAL_TEST") != "1",
        reason="Set VGGT_REAL_TEST=1 to run against a live VGGT server.",
    )
    def test_vggt_real_service(self):
        """Optional smoke test for a live VGGT server."""
        from spagent.tools import VGGTTool

        server_url = os.environ.get("VGGT_SERVER_URL", "http://127.0.0.1:20032")
        tool = VGGTTool(use_mock=False, server_url=server_url)

        result = tool.call(
            image_path=["assets/example.png"],
            azimuth_angle=45,
            elevation_angle=30,
        )

        assert result["success"] is True


# =============================================================================
# Pi3Tool
# =============================================================================

class TestPi3Tool:
    """Tests for Pi3 3D reconstruction tool."""

    def test_pi3_tool_instantiation(self):
        """Pi3Tool should instantiate successfully."""
        from spagent.tools import Pi3Tool

        tool = Pi3Tool(use_mock=True)
        assert tool.name == "pi3_tool"

    def test_pi3_parameters_schema(self):
        """Pi3 parameters should include image_path and angles."""
        from spagent.tools import Pi3Tool

        tool = Pi3Tool(use_mock=True)
        params = tool.parameters

        assert params["type"] == "object"
        assert "image_path" in params["properties"]
        assert "azimuth_angle" in params["properties"]
        assert "elevation_angle" in params["properties"]

    def test_pi3_mock_reconstruction(self):
        """Pi3 mock mode should return point cloud visualization."""
        from spagent.tools import Pi3Tool

        tool = Pi3Tool(use_mock=True)
        result = tool.call(
            image_path=["assets/example.png"],
            azimuth_angle=45,
            elevation_angle=0,
        )

        assert result["success"] is True
        assert "output_path" in result
        assert os.path.exists(result["output_path"])

    @pytest.mark.skipif(
        os.environ.get("PI3_REAL_TEST") != "1",
        reason="Set PI3_REAL_TEST=1 to run against a live Pi3 server.",
    )
    def test_pi3_real_service(self):
        """Optional smoke test for a live Pi3 server."""
        from spagent.tools import Pi3Tool

        server_url = os.environ.get("PI3_SERVER_URL", "http://127.0.0.1:20030")
        tool = Pi3Tool(use_mock=False, server_url=server_url)

        result = tool.call(
            image_path=["assets/example.png"],
            azimuth_angle=45,
            elevation_angle=0,
        )

        assert result["success"] is True


# =============================================================================
# Pi3XTool
# =============================================================================

class TestPi3XTool:
    """Tests for Pi3X (enhanced Pi3) 3D reconstruction tool."""

    def test_pi3x_tool_instantiation(self):
        """Pi3XTool should instantiate successfully."""
        from spagent.tools import Pi3XTool

        tool = Pi3XTool(use_mock=True)
        assert tool.name == "pi3x_tool"

    def test_pi3x_parameters_schema(self):
        """Pi3X parameters should match Pi3 interface."""
        from spagent.tools import Pi3XTool

        tool = Pi3XTool(use_mock=True)
        params = tool.parameters

        assert params["type"] == "object"
        assert "image_path" in params["properties"]
        assert "azimuth_angle" in params["properties"]
        assert "elevation_angle" in params["properties"]

    def test_pi3x_mock_reconstruction(self):
        """Pi3X mock mode should return point cloud visualization."""
        from spagent.tools import Pi3XTool

        tool = Pi3XTool(use_mock=True)
        result = tool.call(
            image_path=["assets/example.png"],
            azimuth_angle=30,
            elevation_angle=-15,
        )

        assert result["success"] is True
        assert "output_path" in result
        assert os.path.exists(result["output_path"])

    @pytest.mark.skipif(
        os.environ.get("PI3X_REAL_TEST") != "1",
        reason="Set PI3X_REAL_TEST=1 to run against a live Pi3X server.",
    )
    def test_pi3x_real_service(self):
        """Optional smoke test for a live Pi3X server."""
        from spagent.tools import Pi3XTool

        server_url = os.environ.get("PI3X_SERVER_URL", "http://127.0.0.1:20031")
        tool = Pi3XTool(use_mock=False, server_url=server_url)

        result = tool.call(
            image_path=["assets/example.png"],
            azimuth_angle=30,
            elevation_angle=-15,
        )

        assert result["success"] is True


# =============================================================================
# Catalog integration tests
# =============================================================================

class TestCatalogIntegration:
    """Tests for catalog registration and build_tools."""

    def test_depth_v2_registered_in_catalog(self):
        """Depth tool should be registered with V2 backend by default."""
        from spagent.tools.catalog import build_tools

        tools, skipped = build_tools(tool_keys=["depth"], use_mock=True)
        assert len(tools) == 1
        assert tools[0].backend == "v2"

    def test_depth_v3_via_override(self):
        """V3 backend should be selectable via catalog override."""
        from spagent.tools.catalog import build_tools

        tools, skipped = build_tools(
            tool_keys=["depth"],
            use_mock=True,
            overrides={"depth": {"backend": "v3"}},
        )
        assert len(tools) == 1
        assert tools[0].backend == "v3"

    def test_vggt_registered_in_catalog(self):
        """VGGT should be in the 3d tool group."""
        from spagent.tools.catalog import get_catalog_by_group, build_tools

        by_group = get_catalog_by_group()
        assert "3d" in by_group

        keys_3d = [e.key for e in by_group["3d"]]
        assert "vggt" in keys_3d
        assert "pi3" in keys_3d
        assert "pi3x" in keys_3d

    def test_build_all_3d_tools(self):
        """All 3d tools should be buildable in mock mode."""
        from spagent.tools.catalog import build_tools, get_catalog_by_group

        by_group = get_catalog_by_group()
        keys_3d = [e.key for e in by_group["3d"]]

        tools, skipped = build_tools(tool_keys=keys_3d, use_mock=True)
        # Some tools may skip if dependencies are missing
        assert len(tools) >= 1


# =============================================================================
# Entry point for CLI usage
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])