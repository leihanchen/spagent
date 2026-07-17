#!/usr/bin/env python3
"""
Batch inference script for Depth Anything V3 on the CoMM dataset.

Loads CoMM test_data.pth, extracts all unique image paths, and runs
Depth V3 on each image with configurable output selection.

Usage:
    # Mock mode (no GPU needed)
    python scripts/run_depth_v3_batch.py \
        --data_path comm/test_data.pth \
        --image_root comm/val_and_test_images \
        --output_dir outputs/depth_v3_batch \
        --save-depth --save-metric-depth --save-gaussians --save-features \
        --use_mock

    # Real server
    python scripts/run_depth_v3_batch.py \
        --data_path comm/test_data.pth \
        --image_root comm/val_and_test_images \
        --output_dir outputs/depth_v3_batch \
        --save-depth --save-features \
        --server_url http://127.0.0.1:20039

    # Resume interrupted run (skips images with existing outputs)
    python scripts/run_depth_v3_batch.py \
        --data_path comm/test_data.pth \
        --image_root comm/val_and_test_images \
        --output_dir outputs/depth_v3_batch \
        --save-depth --save-features \
        --use_mock
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# Dataset loading
# =============================================================================

def load_comm_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load CoMM dataset from .pth file."""
    logger.info("Loading dataset: %s", data_path)
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    logger.info("Loaded %d samples", len(data))
    return data


def extract_image_entries(
    data: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Extract all unique image entries from the CoMM dataset.

    Returns list of dicts with keys:
        - image_rel_path: relative path from .pth (e.g., "images/WikiHow/565261.jpg")
        - source: dataset source (e.g., "WikiHow")
        - image_id: unique ID (e.g., "WikiHow_565261")
    """
    seen: Set[str] = set()
    entries = []

    for item in data:
        dataset_type = item.get("dataset_type", "unknown")
        for step in item.get("step_info", []):
            for element in step:
                if not isinstance(element, dict):
                    continue
                if element.get("type") != "image":
                    continue

                image_rel_path = element.get("image_path", "")
                if not image_rel_path or image_rel_path in seen:
                    continue
                seen.add(image_rel_path)

                # Parse: "images/WikiHow/565261.jpg" -> source=WikiHow, id=565261
                parts = Path(image_rel_path)
                source = parts.parent.name if parts.parent.name != "images" else "unknown"
                image_id = f"{source}_{parts.stem}"

                entries.append({
                    "image_rel_path": image_rel_path,
                    "source": source,
                    "image_id": image_id,
                })

    logger.info("Extracted %d unique image entries", len(entries))
    return entries


def resolve_image_path(
    image_rel_path: str, image_root: str
) -> Optional[str]:
    """
    Resolve the actual image file path.

    The .pth stores paths like "images/WikiHow/565261.jpg".
    The actual files are in "comm/val_and_test_images/WikiHow/565261.jpg".
    """
    # Strip leading "images/" prefix if present
    rel = image_rel_path
    if rel.startswith("images/"):
        rel = rel[len("images/"):]

    full_path = os.path.join(image_root, rel)
    if os.path.exists(full_path):
        return full_path

    # Fallback: try with the original path
    full_path = os.path.join(image_root, image_rel_path)
    if os.path.exists(full_path):
        return full_path

    return None


# =============================================================================
# Resume support
# =============================================================================

def check_existing_outputs(
    output_dir: str, source: str, image_id: str, save_flags: Dict[str, bool]
) -> bool:
    """Check if all requested outputs already exist and are consistent.

    When both metric depth and features are requested, features that are
    older than the float metric map are treated as stale (e.g. left over
    from a previous failed run) so the image is reprocessed.
    """
    source_dir = os.path.join(output_dir, source)

    expected = []
    if save_flags.get("depth"):
        expected.append(os.path.join(source_dir, f"{image_id}_depth.png"))
    metric_depth_npy = os.path.join(source_dir, f"{image_id}_metric_depth.npy")
    if save_flags.get("metric_depth"):
        # Float32 HxW depth in meters (not a color PNG)
        expected.append(metric_depth_npy)
    if save_flags.get("gaussians"):
        expected.append(os.path.join(source_dir, f"{image_id}_gs.ply"))
    features_path = os.path.join(source_dir, f"{image_id}_features.pth")
    features_vis_path = os.path.join(source_dir, f"{image_id}_features_vis.png")
    if save_flags.get("features"):
        expected.append(features_path)
        expected.append(features_vis_path)

    if not all(os.path.exists(p) for p in expected):
        return False

    # Stale features: predate a newer metric map from a partial re-run
    if (
        save_flags.get("features")
        and save_flags.get("metric_depth")
        and os.path.exists(features_path)
        and os.path.exists(metric_depth_npy)
    ):
        if os.path.getmtime(features_path) < os.path.getmtime(metric_depth_npy) - 1.0:
            logger.info(
                "Stale features for %s (older than metric depth); will reprocess",
                image_id,
            )
            return False

    return True


# =============================================================================
# Batch inference
# =============================================================================

def run_batch(
    entries: List[Dict[str, str]],
    image_root: str,
    output_dir: str,
    save_flags: Dict[str, bool],
    use_mock: bool = False,
    server_url: str = "http://127.0.0.1:20039",
) -> Dict[str, Any]:
    """
    Run Depth V3 batch inference on all image entries.

    Returns summary dict with stats and per-image status.
    """
    from spagent.tools import DepthEstimationTool

    tool = DepthEstimationTool(
        use_mock=use_mock,
        backend="v3",
        server_url=server_url,
    )

    stats = {
        "total": len(entries),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "start_time": time.time(),
    }
    per_image_status: List[Dict[str, Any]] = []
    # Abort early if the Flask server dies mid-run (avoid burning walltime
    # on thousands of "unreachable" failures after a crash/OOM kill).
    consecutive_unreachable = 0
    max_consecutive_unreachable = 5
    aborted = False

    for entry in tqdm(entries, desc="Depth V3 batch"):
        image_rel_path = entry["image_rel_path"]
        source = entry["source"]
        image_id = entry["image_id"]

        # Resolve image path
        full_path = resolve_image_path(image_rel_path, image_root)
        if full_path is None:
            logger.warning("Image not found: %s", image_rel_path)
            stats["failed"] += 1
            per_image_status.append({
                "image_id": image_id,
                "status": "not_found",
                "path": image_rel_path,
            })
            continue

        # Resume: skip if all outputs exist
        if check_existing_outputs(output_dir, source, image_id, save_flags):
            stats["skipped"] += 1
            per_image_status.append({
                "image_id": image_id,
                "status": "skipped",
                "path": full_path,
            })
            continue

        # Create output directory
        source_dir = os.path.join(output_dir, source)
        os.makedirs(source_dir, exist_ok=True)

        # Process each requested output mode
        image_success = True
        image_errors = []

        # --- Depth map ---
        if save_flags.get("depth"):
            try:
                result = tool.call(
                    image_path=full_path,
                    output_mode="depth",
                    return_metrics=True,
                )
                if result.get("success"):
                    _copy_output(result.get("output_path"), source_dir, f"{image_id}_depth.png")
                    # Save metrics alongside
                    if result.get("metrics"):
                        metrics_path = os.path.join(source_dir, f"{image_id}_metrics.json")
                        with open(metrics_path, "w") as f:
                            json.dump(result["metrics"], f, indent=2)
                else:
                    image_errors.append(f"depth: {result.get('error', 'unknown')}")
            except Exception as e:
                image_errors.append(f"depth: {e}")

        # --- Metric depth (float32 .npy meters + optional color vis) ---
        if save_flags.get("metric_depth"):
            try:
                result = tool.call(
                    image_path=full_path,
                    output_mode="metric_depth",
                    return_metrics=True,
                )
                if result.get("success"):
                    # Color visualization only (not metric values)
                    _copy_output(
                        result.get("output_path"),
                        source_dir,
                        f"{image_id}_metric_depth_vis.png",
                    )
                    # Float32 HxW depth in meters (no scale/offset needed)
                    _copy_output(
                        result.get("metric_depth_path"),
                        source_dir,
                        f"{image_id}_metric_depth.npy",
                    )
                    if result.get("metrics"):
                        metrics_path = os.path.join(
                            source_dir, f"{image_id}_metrics.json"
                        )
                        with open(metrics_path, "w") as f:
                            json.dump(result["metrics"], f, indent=2)
                else:
                    image_errors.append(f"metric_depth: {result.get('error', 'unknown')}")
            except Exception as e:
                image_errors.append(f"metric_depth: {e}")

        # --- Gaussians ---
        if save_flags.get("gaussians"):
            try:
                result = tool.call(
                    image_path=full_path,
                    output_mode="gaussians",
                    render_views=True,
                )
                if result.get("success"):
                    _copy_output(result.get("ply_path"), source_dir, f"{image_id}.ply")
                    _copy_output(result.get("gs_ply_path"), source_dir, f"{image_id}_gs.ply")
                    # Save rendered views
                    if result.get("rendered_views"):
                        _save_rendered_views(
                            result["rendered_views"], source_dir, image_id
                        )
                else:
                    image_errors.append(f"gaussians: {result.get('error', 'unknown')}")
            except Exception as e:
                image_errors.append(f"gaussians: {e}")

        # --- Features ---
        if save_flags.get("features"):
            try:
                result = tool.call(
                    image_path=full_path,
                    output_mode="features",
                    feature_source=save_flags.get("feature_source", "depth_decoder"),
                )
                if result.get("success"):
                    _copy_output(
                        result.get("features_path"),
                        source_dir,
                        f"{image_id}_features.pth",
                    )
                    _copy_output(
                        result.get("features_vis_path"),
                        source_dir,
                        f"{image_id}_features_vis.png",
                    )
                else:
                    image_errors.append(f"features: {result.get('error', 'unknown')}")
            except Exception as e:
                image_errors.append(f"features: {e}")

        # Record status
        if image_errors:
            image_success = False
            stats["failed"] += 1
            per_image_status.append({
                "image_id": image_id,
                "status": "failed",
                "path": full_path,
                "errors": image_errors,
            })
            err_blob = " ".join(image_errors).lower()
            if "unreachable" in err_blob or "connection" in err_blob:
                consecutive_unreachable += 1
            else:
                consecutive_unreachable = 0
            if consecutive_unreachable >= max_consecutive_unreachable:
                logger.error(
                    "Server unreachable for %d consecutive images; aborting batch "
                    "to preserve walltime. Resume will continue remaining work.",
                    consecutive_unreachable,
                )
                aborted = True
                stats["aborted"] = True
                break
        else:
            consecutive_unreachable = 0
            stats["processed"] += 1
            per_image_status.append({
                "image_id": image_id,
                "status": "success",
                "path": full_path,
            })

    stats["end_time"] = time.time()
    stats["elapsed_seconds"] = stats["end_time"] - stats["start_time"]
    if aborted:
        stats["remaining_unprocessed"] = max(
            0, stats["total"] - stats["processed"] - stats["skipped"] - stats["failed"]
        )

    return {
        "stats": stats,
        "per_image": per_image_status,
    }


# =============================================================================
# File helpers
# =============================================================================

def _copy_output(
    src_path: Optional[str], dest_dir: str, dest_name: str
) -> None:
    """Copy an output file to the destination directory."""
    import shutil

    if src_path and os.path.exists(src_path):
        dst = os.path.join(dest_dir, dest_name)
        shutil.copy2(src_path, dst)


def _save_rendered_views(
    rendered_views: List[Dict[str, Any]],
    source_dir: str,
    image_id: str,
) -> None:
    """Save rendered view images to disk."""
    import base64

    for view_data in rendered_views:
        view_name = view_data.get("view", "unknown")
        img_b64 = view_data.get("image", "")
        if not img_b64:
            continue

        view_path = os.path.join(source_dir, f"{image_id}_view_{view_name}.png")
        img_bytes = base64.b64decode(img_b64)
        with open(view_path, "wb") as f:
            f.write(img_bytes)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Depth V3 inference on CoMM dataset."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="comm/test_data.pth",
        help="Path to CoMM .pth data file (default: comm/test_data.pth).",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="comm/val_and_test_images",
        help="Root directory for image files (default: comm/val_and_test_images).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/depth_v3_batch",
        help="Output directory (default: outputs/depth_v3_batch).",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://127.0.0.1:20039",
        help="V3 server URL (default: http://127.0.0.1:20039).",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use mock service (no GPU or server needed).",
    )

    # Output selection flags
    output_group = parser.add_argument_group("Output selection")
    output_group.add_argument(
        "--save-depth",
        action="store_true",
        help="Save colored depth map (PNG).",
    )
    output_group.add_argument(
        "--save-metric-depth",
        action="store_true",
        help=(
            "Save metric depth as float32 *_metric_depth.npy (meters) + "
            "optional color vis PNG + *_metrics.json summary."
        ),
    )
    output_group.add_argument(
        "--save-gaussians",
        action="store_true",
        help="Save 3D Gaussian Splatting output (GS PLY + rendered views).",
    )
    output_group.add_argument(
        "--save-features",
        action="store_true",
        help="Save 3D-aware features (PTH, 256-dim per patch).",
    )
    output_group.add_argument(
        "--feature-source",
        type=str,
        default="depth_decoder",
        choices=["depth_decoder", "gs_decoder"],
        help="Feature decoder: 'depth_decoder' (default) or 'gs_decoder' (3D Gaussian).",
    )

    # Limit for testing
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images to process (for testing).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    save_flags = {
        "depth": args.save_depth,
        "metric_depth": args.save_metric_depth,
        "gaussians": args.save_gaussians,
        "features": args.save_features,
        "feature_source": args.feature_source,
    }

    if not any(
        save_flags[k] for k in ("depth", "metric_depth", "gaussians", "features")
    ):
        logger.error(
            "No output flags specified. Use at least one of: "
            "--save-depth, --save-metric-depth, --save-gaussians, --save-features"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Depth V3 Batch Inference on CoMM Dataset")
    logger.info("=" * 60)
    logger.info("  Data path    : %s", args.data_path)
    logger.info("  Image root   : %s", args.image_root)
    logger.info("  Output dir   : %s", args.output_dir)
    logger.info("  Server URL   : %s", args.server_url)
    logger.info("  Use mock     : %s", args.use_mock)
    logger.info("  Save flags   : %s", {k: v for k, v in save_flags.items() if v})
    logger.info("-" * 60)

    # Load dataset
    data = load_comm_dataset(args.data_path)
    entries = extract_image_entries(data)

    # Limit for testing
    if args.max_images:
        entries = entries[: args.max_images]
        logger.info("Limited to %d images", args.max_images)

    # Run batch
    os.makedirs(args.output_dir, exist_ok=True)
    summary = run_batch(
        entries=entries,
        image_root=args.image_root,
        output_dir=args.output_dir,
        save_flags=save_flags,
        use_mock=args.use_mock,
        server_url=args.server_url,
    )

    # Save summary
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    stats = summary["stats"]
    logger.info("=" * 60)
    logger.info("Batch Inference Summary")
    logger.info("=" * 60)
    logger.info("  Total images : %d", stats["total"])
    logger.info("  Processed    : %d", stats["processed"])
    logger.info("  Skipped      : %d", stats["skipped"])
    logger.info("  Failed       : %d", stats["failed"])
    logger.info("  Elapsed      : %.1f seconds", stats["elapsed_seconds"])
    logger.info("  Summary saved: %s", summary_path)

    if stats.get("aborted"):
        logger.error(
            "Batch aborted early (server unreachable). remaining≈%s — resubmit to resume.",
            stats.get("remaining_unprocessed", "?"),
        )
        sys.exit(2)
    if stats["failed"] > 0:
        logger.warning("%d images failed — see batch_summary.json for details", stats["failed"])
        sys.exit(1)


if __name__ == "__main__":
    main()
