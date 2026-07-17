#!/usr/bin/env python3
"""
Visualization script for Depth Anything V3 batch inference outputs.

Reads metric depth images (.png/.npy) and 3D latent features (.pth)
produced by run_depth_v3_batch.py, and generates composite
visualizations showing depth maps alongside PCA-projected feature maps.

Usage:
    # Visualize all images in the output directory
    python scripts/visualize_depth_v3_outputs.py \
        --output_dir outputs/depth_v3_batch \
        --image_root comm/val_and_test_images

    # Visualize specific source datasets only
    python scripts/visualize_depth_v3_outputs.py \
        --output_dir outputs/depth_v3_batch \
        --image_root comm/val_and_test_images \
        --sources WikiHow eHow

    # Limit number of images and set figure size
    python scripts/visualize_depth_v3_outputs.py \
        --output_dir outputs/depth_v3_batch \
        --image_root comm/val_and_test_images \
        --max_images 20 \
        --fig_width 18

    # Generate a grid summary of all images
    python scripts/visualize_depth_v3_outputs.py \
        --output_dir outputs/depth_v3_batch \
        --image_root comm/val_and_test_images \
        --mode grid
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
# Data loading
# =============================================================================

def find_image_entries(output_dir: str, sources: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Scan the output directory for images that have both depth and features.

    Returns list of dicts with keys: source, image_id, depth_path,
    metric_path, features_path, orig_image_path.
    """
    entries = []
    output_path = Path(output_dir)

    for source_dir in sorted(output_path.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        if sources and source not in sources:
            continue

        # Group files by image_id
        files_by_id: Dict[str, Dict[str, str]] = {}
        for f in sorted(source_dir.iterdir()):
            if not f.is_file():
                continue
            # Parse: WikiHow_565261_depth.png -> image_id = WikiHow_565261
            name = f.stem
            # Find the last underscore-separated suffix
            parts = name.rsplit("_", 1)
            if len(parts) < 2:
                continue
            image_id = parts[0]
            suffix = parts[1]

            if image_id not in files_by_id:
                files_by_id[image_id] = {"source": source, "image_id": image_id}
            files_by_id[image_id][f"{suffix}_path"] = str(f)

        for image_id, file_dict in sorted(files_by_id.items()):
            # Only include entries that have at least features or depth
            if "features_path" in file_dict or "depth_path" in file_dict:
                entries.append(file_dict)

    logger.info("Found %d image entries in %s", len(entries), output_dir)
    return entries


def load_depth_image(path: str) -> Optional[np.ndarray]:
    """Load a depth PNG visualization image."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        logger.warning("Failed to load depth image %s: %s", path, e)
        return None


def load_metric_depth(path: str) -> Optional[np.ndarray]:
    """Load raw metric depth from .npy file."""
    try:
        return np.load(path)
    except Exception as e:
        logger.warning("Failed to load metric depth %s: %s", path, e)
        return None


def load_features(path: str) -> Optional[Dict[str, Any]]:
    """Load 3D latent features from .pth file."""
    try:
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        return data
    except Exception as e:
        logger.warning("Failed to load features %s: %s", path, e)
        return None


def load_original_image(image_id: str, source: str, image_root: str) -> Optional[np.ndarray]:
    """Load the original input image."""
    try:
        from PIL import Image
        # image_id like "WikiHow_565261" -> source=WikiHow, stem=565261
        stem = image_id.replace(f"{source}_", "", 1)
        # Try common extensions
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            img_path = os.path.join(image_root, source, f"{stem}{ext}")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                return np.array(img)
        logger.warning("Original image not found for %s/%s", source, stem)
        return None
    except Exception as e:
        logger.warning("Failed to load original image: %s", e)
        return None


# =============================================================================
# Feature visualization
# =============================================================================

def features_to_pca_rgb(features: np.ndarray, feature_h: int, feature_w: int) -> np.ndarray:
    """
    Project 3D latent features to 3-channel RGB via PCA.

    Args:
        features: Array of shape [1, num_patches, feature_dim] or [num_patches, feature_dim]
        feature_h: Spatial height of the patch grid
        feature_w: Spatial width of the patch grid

    Returns:
        RGB image of shape [feature_h, feature_w, 3] with values in [0, 255]
    """
    from sklearn.decomposition import PCA

    # Flatten to [num_patches, feature_dim]
    if features.ndim == 3:
        patches = features[0]  # [num_patches, feature_dim]
    else:
        patches = features

    # PCA to 3 components
    n_components = min(3, patches.shape[1], patches.shape[0])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(patches)  # [num_patches, 3 or less]

    # Pad to 3 channels if needed
    if n_components < 3:
        pad = np.zeros((projected.shape[0], 3 - n_components))
        projected = np.concatenate([projected, pad], axis=1)

    # Normalize each channel to [0, 1]
    for c in range(3):
        col = projected[:, c]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-8:
            projected[:, c] = (col - col_min) / (col_max - col_min)
        else:
            projected[:, c] = 0

    # Reshape to spatial grid
    rgb = projected[:, :3].reshape(feature_h, feature_w, 3)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb


def features_to_similarity_map(features: np.ndarray, feature_h: int, feature_w: int,
                                query_patch: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Compute cosine similarity of every patch to a query patch,
    returning a single-channel heatmap.

    Args:
        features: [1, num_patches, feature_dim] or [num_patches, feature_dim]
        feature_h, feature_w: spatial grid dimensions
        query_patch: (row, col) of the query patch

    Returns:
        Heatmap of shape [feature_h, feature_w] with values in [0, 1]
    """
    if features.ndim == 3:
        patches = features[0]
    else:
        patches = features

    # Normalize
    norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8
    patches_norm = patches / norms

    # Query vector
    qr, qc = query_patch
    qidx = qr * feature_w + qc
    if qidx >= patches_norm.shape[0]:
        qidx = 0
    query_vec = patches_norm[qidx]  # [feature_dim]

    # Cosine similarity
    sim = patches_norm @ query_vec  # [num_patches]
    sim = (sim + 1) / 2  # map [-1, 1] -> [0, 1]
    sim = sim.reshape(feature_h, feature_w)
    return sim


# =============================================================================
# Plotting
# =============================================================================

def plot_single_image(
    entry: Dict[str, str],
    image_root: str,
    save_dir: str,
    fig_width: int = 15,
) -> Optional[str]:
    """
    Generate a composite visualization for a single image:
    [Original | Depth Map | Metric Depth | Feature PCA | Feature Similarity]

    Returns the output path, or None on failure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    source = entry["source"]
    image_id = entry["image_id"]

    # Load data
    orig_img = load_original_image(image_id, source, image_root)
    depth_img = load_depth_image(entry.get("depth_path", ""))
    metric_data = load_metric_depth(entry.get("metric_path", ""))

    feat_data = None
    if "features_path" in entry and os.path.exists(entry["features_path"]):
        feat_data = load_features(entry["features_path"])

    # Determine which panels we have
    panels = []
    if orig_img is not None:
        panels.append(("Original", orig_img, "image"))
    if depth_img is not None:
        panels.append(("Depth Map", depth_img, "image"))
    if metric_data is not None:
        panels.append(("Metric Depth (m)", metric_data, "heatmap"))
    if feat_data is not None:
        panels.append(("3D Features (PCA)", None, "features_pca"))
        panels.append(("Feature Similarity", None, "features_sim"))

    if not panels:
        logger.warning("No data to visualize for %s/%s", source, image_id)
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_width // n_panels + 1))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"{source}/{image_id}", fontsize=14, fontweight="bold")

    for ax, (title, data, ptype) in zip(axes, panels):
        ax.set_title(title, fontsize=10)
        ax.axis("off")

        if ptype == "image":
            ax.imshow(data)

        elif ptype == "heatmap":
            im = ax.imshow(data, cmap="turbo")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, format="%.1f")

        elif ptype == "features_pca" and feat_data is not None:
            features = feat_data["features"].numpy() if hasattr(feat_data["features"], "numpy") else feat_data["features"]
            fh = feat_data.get("feature_h", 0)
            fw = feat_data.get("feature_w", 0)
            if fh > 0 and fw > 0:
                rgb = features_to_pca_rgb(features, fh, fw)
                # Upsample to reasonable display size
                from PIL import Image as PILImage
                display_h = max(fh * 8, 224)
                display_w = max(fw * 8, 224)
                rgb_pil = PILImage.fromarray(rgb).resize(
                    (display_w, display_h), PILImage.NEAREST
                )
                ax.imshow(np.array(rgb_pil))
                ax.set_xlabel(f"dim={feat_data.get('feature_dim', '?')}, "
                              f"type={feat_data.get('feature_type', '?')}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No spatial info", ha="center", va="center")

        elif ptype == "features_sim" and feat_data is not None:
            features = feat_data["features"].numpy() if hasattr(feat_data["features"], "numpy") else feat_data["features"]
            fh = feat_data.get("feature_h", 0)
            fw = feat_data.get("feature_w", 0)
            if fh > 0 and fw > 0:
                sim = features_to_similarity_map(features, fh, fw, query_patch=(fh // 2, fw // 2))
                from PIL import Image as PILImage
                display_h = max(fh * 8, 224)
                display_w = max(fw * 8, 224)
                sim_pil = PILImage.fromarray((sim * 255).astype(np.uint8)).resize(
                    (display_w, display_h), PILImage.NEAREST
                )
                im = ax.imshow(np.array(sim_pil), cmap="hot", vmin=0, vmax=1)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)
                ax.set_xlabel("cosine sim to center patch", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No spatial info", ha="center", va="center")

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{image_id}_vis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grid_summary(
    entries: List[Dict[str, str]],
    image_root: str,
    save_path: str,
    max_images: int = 36,
    cols: int = 6,
) -> None:
    """
    Generate a grid summary showing depth + feature PCA for many images.
    Each cell shows: depth map on top, feature PCA on bottom.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(entries), max_images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else np.array([[axes]])

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        if idx < n:
            entry = entries[idx]
            source = entry["source"]
            image_id = entry["image_id"]

            # Build a 2-row composite: depth on top, features on bottom
            composite_parts = []

            depth_img = load_depth_image(entry.get("depth_path", ""))
            if depth_img is not None:
                # Resize to fixed height
                from PIL import Image as PILImage
                h_target = 112
                w_target = int(depth_img.shape[1] * h_target / depth_img.shape[0])
                depth_resized = np.array(
                    PILImage.fromarray(depth_img).resize((w_target, h_target))
                )
                composite_parts.append(depth_resized)

            feat_data = None
            if "features_path" in entry and os.path.exists(entry["features_path"]):
                feat_data = load_features(entry["features_path"])

            if feat_data is not None:
                features = feat_data["features"].numpy() if hasattr(feat_data["features"], "numpy") else feat_data["features"]
                fh = feat_data.get("feature_h", 0)
                fw = feat_data.get("feature_w", 0)
                if fh > 0 and fw > 0:
                    rgb = features_to_pca_rgb(features, fh, fw)
                    from PIL import Image as PILImage
                    w_target = composite_parts[0].shape[1] if composite_parts else 112
                    h_feat = int(fh * w_target / fw) if fw > 0 else 56
                    h_feat = min(h_feat, 112)
                    rgb_resized = np.array(
                        PILImage.fromarray(rgb).resize((w_target, h_feat), PILImage.NEAREST)
                    )
                    composite_parts.append(rgb_resized)

            if composite_parts:
                composite = np.concatenate(composite_parts, axis=0)
                ax.imshow(composite)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)

            # Truncate label
            label = f"{source[:6]}/{image_id.split('_')[-1][:8]}"
            ax.set_title(label, fontsize=7)
        ax.axis("off")

    plt.suptitle("Depth V3 Batch — Depth + Feature PCA Grid", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grid summary saved: %s", save_path)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Depth V3 batch inference outputs."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/depth_v3_batch",
        help="Directory containing batch inference outputs (default: outputs/depth_v3_batch).",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="comm/val_and_test_images",
        help="Root directory for original input images (default: comm/val_and_test_images).",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: <output_dir>/visualizations).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="individual",
        choices=["individual", "grid", "both"],
        help="Visualization mode: 'individual' per-image composites, 'grid' summary, or 'both' (default: individual).",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Only visualize these source datasets (e.g., WikiHow eHow). Default: all.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images to visualize.",
    )
    parser.add_argument(
        "--fig_width",
        type=int,
        default=15,
        help="Figure width in inches for individual mode (default: 15).",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=6,
        help="Number of columns in grid mode (default: 6).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    vis_dir = args.vis_dir or os.path.join(args.output_dir, "visualizations")

    logger.info("=" * 60)
    logger.info("Depth V3 Output Visualization")
    logger.info("=" * 60)
    logger.info("  Output dir  : %s", args.output_dir)
    logger.info("  Image root  : %s", args.image_root)
    logger.info("  Vis dir     : %s", vis_dir)
    logger.info("  Mode        : %s", args.mode)
    logger.info("  Sources     : %s", args.sources or "all")
    logger.info("-" * 60)

    # Find entries
    entries = find_image_entries(args.output_dir, args.sources)

    if not entries:
        logger.error("No image entries found in %s", args.output_dir)
        sys.exit(1)

    if args.max_images:
        entries = entries[: args.max_images]
        logger.info("Limited to %d images", args.max_images)

    # Individual mode
    if args.mode in ("individual", "both"):
        logger.info("Generating individual visualizations...")
        os.makedirs(vis_dir, exist_ok=True)
        success, failed = 0, 0
        for entry in tqdm(entries, desc="Visualizing"):
            source = entry["source"]
            image_id = entry["image_id"]
            save_subdir = os.path.join(vis_dir, source)
            try:
                out_path = plot_single_image(
                    entry, args.image_root, save_subdir, args.fig_width
                )
                if out_path:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error("Failed to visualize %s/%s: %s", source, image_id, e)
                failed += 1

        logger.info("Individual: %d succeeded, %d failed", success, failed)

    # Grid mode
    if args.mode in ("grid", "both"):
        logger.info("Generating grid summary...")
        grid_path = os.path.join(vis_dir, "grid_summary.png")
        try:
            plot_grid_summary(
                entries, args.image_root, grid_path,
                max_images=args.max_images or 36,
                cols=args.grid_cols,
            )
        except Exception as e:
            logger.error("Grid summary failed: %s", e)

    logger.info("Done. Visualizations saved to: %s", vis_dir)


if __name__ == "__main__":
    main()
