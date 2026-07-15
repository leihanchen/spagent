#!/bin/bash
#SBATCH --job-name=depth_v3_batch
#SBATCH --account=def-wangcs
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/depth_v3_batch_%j.out
#SBATCH --error=logs/depth_v3_batch_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leihan.chen@torontomu.ca

# Batch Depth Anything V3 inference on the CoMM dataset.
# Saves metric 16-bit depth (+ color vis) and 3D-aware features per image.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/run_depth_v3_batch.slurm
#   sbatch scripts/run_depth_v3_batch.slurm --max-images 100 --feature-source depth_decoder

set -euo pipefail

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo ">>> Cleanup: stopping server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM

# =====================================================================
# Configuration (override via args after sbatch)
# =====================================================================
DATA_PATH="comm/test_data.pth"
IMAGE_ROOT="comm/val_and_test_images"
OUTPUT_DIR="outputs/depth_v3_batch"
SERVER_PORT=20039
FEATURE_SOURCE="depth_decoder"   # depth_decoder | gs_decoder (needs gsplat)
MAX_IMAGES=""                 # empty string = all images
CHECKPOINT_ID="depth-anything/DA3NESTED-GIANT-LARGE-1.1"

while [[ $# -gt 0 ]]; do
  case $1 in
    --data-path)      DATA_PATH="$2";      shift 2 ;;
    --image-root)     IMAGE_ROOT="$2";     shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
    --feature-source) FEATURE_SOURCE="$2"; shift 2 ;;
    --max-images)     MAX_IMAGES="$2";     shift 2 ;;
    --checkpoint-id)  CHECKPOINT_ID="$2";  shift 2 ;;
    --server-port)    SERVER_PORT="$2";    shift 2 ;;
    *) echo "Unknown option: $1"; shift ;;
  esac
done

echo "============================================================"
echo " Depth Anything V3 — CoMM Batch Inference"
echo "============================================================"
echo " Job ID:          ${SLURM_JOB_ID:-local}"
echo " Node:            ${SLURMD_NODENAME:-$(hostname)}"
echo " GPU:             ${CUDA_VISIBLE_DEVICES:-}"
echo " Data path:       $DATA_PATH"
echo " Image root:      $IMAGE_ROOT"
echo " Output dir:      $OUTPUT_DIR"
echo " Feature source:  $FEATURE_SOURCE"
echo " Max images:      ${MAX_IMAGES:-all}"
echo " Checkpoint:      $CHECKPOINT_ID"
echo "============================================================"

# =====================================================================
# Environment
# =====================================================================
PROJECT_ROOT="/scratch/leihan/spagent"
cd "$PROJECT_ROOT"

module load python/3.12.4
module load cuda/13.2
module load opencv/4.13.0   # system cv2 — do not pip install opencv-python

export HF_HOME="/scratch/leihan/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" logs "$OUTPUT_DIR"
echo ">>> HF cache: $HF_HOME"

VENV_DIR="$PROJECT_ROOT/.venv_depth_v3"
if [[ ! -d "$VENV_DIR" ]]; then
  echo ">>> Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------
# Core deps (idempotent)
# ---------------------------------------------------------------------
echo ">>> Ensuring pip tooling..."
pip install --no-build-isolation --upgrade pip setuptools wheel packaging -q

if ! python3 -c "import torch" 2>/dev/null; then
  echo ">>> Installing torch + torchvision (cu132)..."
  pip install --no-build-isolation \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu132
else
  echo ">>> torch already installed, skipping"
fi

# Runtime deps for our Flask server + DA3 monocular inference path.
# Intentionally omit: opencv-python, xformers, open3d, pycolmap, gsplat
# (module opencv supplies cv2; heavy optional DA3 export deps not needed).
echo ">>> Installing runtime deps..."
pip install --no-build-isolation -q \
  flask requests Pillow numpy tqdm matplotlib \
  huggingface_hub safetensors einops omegaconf addict evo \
  imageio "moviepy==1.0.3" plyfile trimesh

# ---------------------------------------------------------------------
# Install Depth Anything 3 (official package, no heavy optional deps)
# ---------------------------------------------------------------------
DA3_DIR="$PROJECT_ROOT/.depth_anything_3_src"
if [[ ! -d "$DA3_DIR/src/depth_anything_3" ]]; then
  echo ">>> Cloning Depth-Anything-3..."
  rm -rf "$DA3_DIR"
  git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3_DIR"
else
  echo ">>> Depth-Anything-3 source present: $DA3_DIR"
fi

# Full `pip install -e .` pulls xformers/open3d/pycolmap/opencv-python.
# Install package only (--no-deps) + runtime deps above.
echo ">>> Installing depth-anything-3 (editable, --no-deps)..."
pip install --no-build-isolation hatchling hatch-vcs editables -q
if ! pip install --no-build-isolation -e "$DA3_DIR" --no-deps; then
  echo ">>> Editable install failed; using PYTHONPATH only"
fi
export PYTHONPATH="$DA3_DIR/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Eager import of export.colmap requires pycolmap (unavailable on CC).
# Lazy-load export formats so monocular depth inference can import DA3.
EXPORT_INIT="$DA3_DIR/src/depth_anything_3/utils/export/__init__.py"
if [[ -f "$EXPORT_INIT" ]] && grep -q "from \.colmap import export_to_colmap" "$EXPORT_INIT" 2>/dev/null; then
  echo ">>> Patching DA3 export/__init__.py for lazy optional imports..."
  cat > "$EXPORT_INIT" <<'PY'
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Lazy export imports (spagent HPC patch): avoid requiring pycolmap/gsplat at import time.
from depth_anything_3.specs import Prediction


def export(prediction: Prediction, export_format: str, export_dir: str, **kwargs):
    if "-" in export_format:
        for fmt in export_format.split("-"):
            export(prediction, fmt, export_dir, **kwargs)
        return
    if export_format == "glb":
        from .glb import export_to_glb
        export_to_glb(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "mini_npz":
        from .npz import export_to_mini_npz
        export_to_mini_npz(prediction, export_dir)
    elif export_format == "npz":
        from .npz import export_to_npz
        export_to_npz(prediction, export_dir)
    elif export_format == "feat_vis":
        from .feat_vis import export_to_feat_vis
        export_to_feat_vis(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "depth_vis":
        from .depth_vis import export_to_depth_vis
        export_to_depth_vis(prediction, export_dir)
    elif export_format == "gs_ply":
        from depth_anything_3.utils.export.gs import export_to_gs_ply
        export_to_gs_ply(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "gs_video":
        from depth_anything_3.utils.export.gs import export_to_gs_video
        export_to_gs_video(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "colmap":
        from .colmap import export_to_colmap
        export_to_colmap(prediction, export_dir, **kwargs.get(export_format, {}))
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


__all__ = [export]
PY
fi

# Fail-closed: official package required (server has no untrained fallback)
echo ">>> Verifying depth_anything_3..."
python3 - <<'PY'
from depth_anything_3.api import DepthAnything3
print("  DA3 OK:", DepthAnything3)
PY

# gsplat only when GS decoder features are requested
if [[ "$FEATURE_SOURCE" == "gs_decoder" ]]; then
  module load gcc/12.3
  if python3 -c "import gsplat" 2>/dev/null; then
    echo ">>> gsplat already installed, skipping"
  else
    echo ">>> Installing gsplat (may take several minutes)..."
    timeout 600 pip install --no-build-isolation \
      git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0 || {
        echo ">>> gsplat install failed/timed out; falling back to depth_decoder"
        FEATURE_SOURCE="depth_decoder"
      }
  fi
fi

echo ">>> Checking GPU (fail-closed)..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found — no GPU visible on this node"
  exit 1
fi
nvidia-smi -L || { echo "ERROR: nvidia-smi -L failed"; exit 1; }
python3 - <<'PY'
import torch
print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"  device_count: {torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: CUDA not available to PyTorch — refusing to run on CPU")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  CUDA capability: {torch.cuda.get_device_capability(0)}")
PY

# =====================================================================
# Start V3 server
# =====================================================================
echo ""
echo ">>> Starting Depth Anything V3 server on port $SERVER_PORT..."
python3 -m spagent.external_experts.Depth_AnythingV2.depth_v3_server \
  --checkpoint_id "$CHECKPOINT_ID" \
  --port "$SERVER_PORT" \
  &> "logs/depth_v3_server_${SLURM_JOB_ID}.log" &
SERVER_PID=$!

echo ">>> Waiting for server health..."
MAX_WAIT=300
ELAPSED=0
while (( ELAPSED < MAX_WAIT )); do
  if curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null; then
    echo ">>> Server ready (${ELAPSED}s)"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server process exited early"
    tail -50 "logs/depth_v3_server_${SLURM_JOB_ID}.log" || true
    exit 1
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  echo "    ... waiting (${ELAPSED}s / ${MAX_WAIT}s)"
done

if (( ELAPSED >= MAX_WAIT )); then
  echo "ERROR: Server not ready within ${MAX_WAIT}s"
  tail -50 "logs/depth_v3_server_${SLURM_JOB_ID}.log" || true
  exit 1
fi

# =====================================================================
# Batch inference
# =====================================================================
echo ""
echo ">>> Running batch inference..."

BATCH_ARGS=(
  --data_path "$DATA_PATH"
  --image_root "$IMAGE_ROOT"
  --output_dir "$OUTPUT_DIR"
  --server_url "http://127.0.0.1:${SERVER_PORT}"
  --save-metric-depth
  --save-features
  --feature-source "$FEATURE_SOURCE"
)
if [[ -n "$MAX_IMAGES" ]]; then
  BATCH_ARGS+=(--max_images "$MAX_IMAGES")
fi

echo ">>> Command: python3 scripts/run_depth_v3_batch.py ${BATCH_ARGS[*]}"
set +e
python3 scripts/run_depth_v3_batch.py "${BATCH_ARGS[@]}"
BATCH_EXIT=$?
set -e

# =====================================================================
# Finish
# =====================================================================
if [[ $BATCH_EXIT -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo " Batch inference completed successfully!"
  echo " Output directory: $OUTPUT_DIR"
  echo " Summary:          $OUTPUT_DIR/batch_summary.json"
  echo " Server log:       logs/depth_v3_server_${SLURM_JOB_ID}.log"
  echo "============================================================"
  echo " Metric artifacts per image:"
  echo "   *_metric_depth.npy        (float32 meters, HxW)"
  echo "   *_metric_depth_vis.png    (color visualization only)"
  echo "   *_features.pth            (3D-aware features, 256-dim per patch)"
  echo "   *_features_vis.png        (PCA→RGB feature visualization)"
  echo "   *_metrics.json            (optional min/max summary)"
  echo "============================================================"
else
  echo ""
  echo "ERROR: Batch inference failed with exit code $BATCH_EXIT"
  tail -50 "logs/depth_v3_server_${SLURM_JOB_ID}.log" || true
  exit "$BATCH_EXIT"
fi
