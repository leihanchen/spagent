#!/bin/bash
#SBATCH --job-name=depth_v3_batch
#SBATCH --account=def-wangcs
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/depth_v3_batch_%j.out
#SBATCH --error=logs/depth_v3_batch_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leihan.chen@torontomu.ca

# Batch Depth Anything V3 inference on the CoMM dataset.
# Produces metric depth images + 3D Gaussian decoder features per image.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/run_depth_v3_batch.slurm
#
# With custom options:
#   sbatch scripts/run_depth_v3_batch.slurm
#     --max-images 100
#     --feature-source depth_decoder
#     --output-dir outputs/depth_v3_batch_run2
#
# Multi-GPU: To use 2 GPUs, change --gpus-per-node=h100:2 and set
#   NUM_SERVERS=2 below. The script will launch 2 server instances
#  (ports 20039, 20040) and split the image list across them.

set -euo pipefail

# Graceful cleanup on exit/timeout — kill the server process
cleanup() {
  echo ""
  echo ">>> Cleanup: stopping server (PID ${SERVER_PID:-unknown})..."
  kill ${SERVER_PID:-} 2>/dev/null || true
  wait ${SERVER_PID:-} 2>/dev/null || true
}
trap cleanup EXIT TERM

# =====================================================================
# Configuration (override via command-line args after sbatch)
# =====================================================================
DATA_PATH="comm/test_data.pth"
IMAGE_ROOT="comm/val_and_test_images"
OUTPUT_DIR="outputs/depth_v3_batch"
SERVER_PORT=20039
FEATURE_SOURCE="depth_decoder"    # "depth_decoder" or "gs_decoder" (gs_decoder requires gsplat)
MAX_IMAGES="100"                  # start with 100 images; set to "" for all 12K
CHECKPOINT_ID="depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# Parse extra args passed after sbatch
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-path)     DATA_PATH="$2";     shift 2 ;;
    --image-root)    IMAGE_ROOT="$2";    shift 2 ;;
    --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
    --feature-source) FEATURE_SOURCE="$2"; shift 2 ;;
    --max-images)    MAX_IMAGES="$2";    shift 2 ;;
    --checkpoint-id) CHECKPOINT_ID="$2"; shift 2 ;;
    --server-port)   SERVER_PORT="$2";   shift 2 ;;
    *) echo "Unknown option: $1"; shift ;;
  esac
done

echo "============================================================"
echo " Depth Anything V3 — CoMM Batch Inference"
echo "============================================================"
echo " Job ID:          $SLURM_JOB_ID"
echo " Node:            $SLURMD_NODENAME"
echo " GPU:             $CUDA_VISIBLE_DEVICES"
echo " Data path:       $DATA_PATH"
echo " Image root:      $IMAGE_ROOT"
echo " Output dir:      $OUTPUT_DIR"
echo " Feature source:  $FEATURE_SOURCE"
echo " Max images:      ${MAX_IMAGES:-all}"
echo " Checkpoint:      $CHECKPOINT_ID"
echo "============================================================"

# =====================================================================
# Environment setup
# =====================================================================
PROJECT_ROOT="/scratch/leihan/spagent"
cd "$PROJECT_ROOT"

# Load system modules
module load python/3.12.4
module load cuda/13.2
module load opencv/4.13.0
module load gcc/12.3             # needed for compiling gsplat CUDA extensions

# --- HuggingFace cache on /scratch (home quota is limited) ---
HF_CACHE_DIR="/scratch/leihan/.cache/huggingface"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
echo ">>> HF cache: $HF_HOME"

# Create virtualenv if it doesn't exist
# NOTE: opencv module must be loaded BEFORE venv activation so that
#       cv2 is available inside the venv (ComputeCanada convention).
VENV_DIR="$PROJECT_ROOT/.venv_depth_v3"
if [ ! -d "$VENV_DIR" ]; then
  echo ">>> Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install dependencies
# Heavy deps (torch, torchvision) are skipped if already present.
# Lightweight deps (flask, requests, etc.) are always installed (fast, idempotent).
echo ">>> Installing dependencies..."
pip install --no-build-isolation --upgrade pip setuptools wheel

# Core ML dependencies — skip if torch is already installed (saves ~5 min)
if ! python3 -c "import torch" 2>/dev/null; then
  pip install --no-build-isolation \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu132
else
  echo ">>> torch already installed, skipping"
fi

# Project dependencies (NO opencv pip — loaded via module above)
# Always run: fast and idempotent; ensures missing packages like 'requests' get installed
pip install --no-build-isolation \
  flask \
  requests \
  Pillow \
  numpy \
  huggingface_hub \
  safetensors \
  tqdm \
  matplotlib \
  scipy \
  scikit-learn \
  lxml \
  einops \
  omegaconf \
  addict

# Depth Anything V3 — not on PyPI, and its full pip install pulls in heavy
# deps (xformers, pycolmap, e3nn, open3d, opencv-python) that conflict with
# ComputeCanada modules.  Instead, clone the source and add to PYTHONPATH
# so the server can import depth_anything_3 directly.
echo ">>> Setting up Depth Anything V3 source..."
DA3_DIR="$PROJECT_ROOT/.depth_anything_3_src"
if [ ! -d "$DA3_DIR" ]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3_DIR"
fi
# Install only the lightweight build deps needed for the package metadata
pip install --no-build-isolation \
  hatchling hatch-vcs \
  2>/dev/null || true
# Add source to PYTHONPATH (avoids full dependency install)
export PYTHONPATH="$DA3_DIR/src:$PROJECT_ROOT:${PYTHONPATH:-}"

# Optional: gsplat for Gaussian head (needed for feature_source=gs_decoder)
# Only install if using gs_decoder; skip if already installed; timeout after 10 min
if [ "$FEATURE_SOURCE" = "gs_decoder" ]; then
  if python3 -c "import gsplat" 2>/dev/null; then
    echo ">>> gsplat already installed, skipping build"
  else
    echo ">>> Installing gsplat (this may take several minutes to compile CUDA kernels)..."
    timeout 600 pip install --no-build-isolation \
      git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0 || {
      echo ">>> gsplat install failed or timed out; GS decoder features may not be available"
      echo ">>> Falling back to depth_decoder features"
      FEATURE_SOURCE="depth_decoder"
    }
  fi
else
  echo ">>> Skipping gsplat install (feature_source=$FEATURE_SOURCE, gsplat only needed for gs_decoder)"
fi

# Install the project itself (editable so scripts/ imports work)
pip install --no-build-isolation -e . 2>/dev/null || {
  echo ">>> Project pip install skipped (not a package); using PYTHONPATH"
}

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Verify GPU access
echo ">>> Checking GPU..."
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Create log directory
mkdir -p logs "$OUTPUT_DIR"

# =====================================================================
# Step 1: Start the V3 server in the background
# =====================================================================
echo ""
echo ">>> Starting Depth Anything V3 server on port $SERVER_PORT..."

python3 -m spagent.external_experts.Depth_AnythingV2.depth_v3_server \
  --checkpoint_id "$CHECKPOINT_ID" \
  --port "$SERVER_PORT" \
  &> "logs/depth_v3_server_${SLURM_JOB_ID}.log" &
SERVER_PID=$!

# Wait for server to be ready (up to 5 minutes)
echo ">>> Waiting for server to be ready..."
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
  if curl -s "http://127.0.0.1:$SERVER_PORT/health" > /dev/null 2>&1; then
    echo ">>> Server is ready (took ${ELAPSED}s)"
    break
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  echo "    ... waiting (${ELAPSED}s / ${MAX_WAIT}s)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
  echo "ERROR: Server did not become ready within ${MAX_WAIT}s"
  echo "Server log tail:"
  tail -30 "logs/depth_v3_server_${SLURM_JOB_ID}.log"
  kill $SERVER_PID 2>/dev/null || true
  exit 1
fi

# =====================================================================
# Step 2: Run batch inference
# =====================================================================
echo ""
echo ">>> Running batch inference..."

BATCH_CMD="python3 scripts/run_depth_v3_batch.py \
  --data_path $DATA_PATH \
  --image_root $IMAGE_ROOT \
  --output_dir $OUTPUT_DIR \
  --server_url http://127.0.0.1:$SERVER_PORT \
  --save-metric-depth \
  --save-features \
  --feature-source $FEATURE_SOURCE"

# Add max_images if specified
if [ -n "$MAX_IMAGES" ]; then
  BATCH_CMD="$BATCH_CMD --max_images $MAX_IMAGES"
fi

echo ">>> Command: $BATCH_CMD"
eval $BATCH_CMD

BATCH_EXIT=$?

# =====================================================================
# Step 3: Visualize outputs (no GPU needed)
# =====================================================================
if [ $BATCH_EXIT -eq 0 ]; then
  echo ""
  echo ">>> Generating visualizations..."
  python3 scripts/visualize_depth_v3_outputs.py \
    --output_dir "$OUTPUT_DIR" \
    --image_root "$IMAGE_ROOT" \
    --mode both \
    --max_images 50 \
    2>&1 || {
      echo ">>> Visualization failed (non-fatal, outputs are still valid)"
    }
fi

# =====================================================================
# Step 4: Cleanup
# =====================================================================
echo ""
echo ">>> Stopping server (PID $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

if [ $BATCH_EXIT -eq 0 ]; then
  echo ""
  echo "============================================================"
  echo " Batch inference completed successfully!"
  echo "============================================================"
  echo " Output directory: $OUTPUT_DIR"
  echo " Summary:          $OUTPUT_DIR/batch_summary.json"
  echo " Server log:       logs/depth_v3_server_${SLURM_JOB_ID}.log"
  echo "============================================================"
else
  echo ""
  echo "ERROR: Batch inference failed with exit code $BATCH_EXIT"
  echo "Server log tail:"
  tail -50 "logs/depth_v3_server_${SLURM_JOB_ID}.log"
  exit $BATCH_EXIT
fi
