"""
Depth-AnythingV2 Expert Module

This module provides depth estimation capabilities using the Depth-AnythingV2
and Depth-Anything-V3 models.

Imports are lazy to avoid requiring all dependencies (cv2, torch, flask) at
import time — only the chosen backend's dependencies are needed at runtime.
"""


def _get_depth_client():
    from .depth_client import DepthClient
    return DepthClient


def _get_mock_depth_service():
    from .mock_depth_service import MockDepthService
    return MockDepthService


def _get_depth_v3_client():
    from .depth_v3_client import DepthV3Client
    return DepthV3Client


def _get_mock_depth_v3_service():
    from .mock_depth_v3_service import MockDepthV3Service
    return MockDepthV3Service


__all__ = [
    'DepthClient',
    'MockDepthService',
    'DepthV3Client',
    'MockDepthV3Service',
]
