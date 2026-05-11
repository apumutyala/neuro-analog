#!/bin/bash
set -euo pipefail

# =============================================================================
# neuro-analog Pilot Toy Tasks — Full RunPod Pipeline
# =============================================================================
# This script trains all 7 pilot models, runs the complete sweep matrix,
# collects layer sensitivity and transient stability data, and generates
# all figures. Run this inside a RunPod GPU pod after uploading the
# neuro-analog repository.
#
# Usage:
#   cd /workspace/neuro-analog
#   bash experiments/cross_arch_tolerance/runpod_pilot.sh
#
# Estimated runtime: ~3–4 hours on an RTX A4000/3090 class GPU.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}"
ROOT_DIR="$(cd "${EXP_DIR}/../.." && pwd)"
RESULTS_DIR="${EXP_DIR}/results"
FIGURES_DIR="${EXP_DIR}/figures"
CKPT_DIR="${EXP_DIR}/checkpoints"

N_TRIALS_MAIN=100
N_TRIALS_LAYER=50
N_INFER_TRANSIENT=5000

echo "========================================"
echo "neuro-analog Pilot RunPod Pipeline"
echo "========================================"
echo "Repo root:    ${ROOT_DIR}"
echo "Experiment:   ${EXP_DIR}"
echo "N trials:     ${N_TRIALS_MAIN}"
echo "========================================"

# ---------------------------------------------------------------------------
# 0. Environment setup
# ---------------------------------------------------------------------------
echo ""
echo "[0/6] Setting up environment..."
cd "${ROOT_DIR}"
pip install -q -e .
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# 1. Train all 7 architectures
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Training all pilot models..."
cd "${EXP_DIR}"
python train_all.py

# Verify checkpoints exist
for arch in neural_ode transformer diffusion flow ebm deq ssm; do
    if [[ ! -f "${CKPT_DIR}/${arch}.pt" ]]; then
        echo "ERROR: Missing checkpoint for ${arch}" >&2
        exit 1
    fi
done
echo "  All checkpoints verified."

# ---------------------------------------------------------------------------
# 2. Main sweep matrix: all substrates × both domains
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Running full sweep matrix (substrates=all, domains=both, n_trials=${N_TRIALS_MAIN})..."
python sweep_all.py \
    --analog-substrate all \
    --analog-domain both \
    --n-trials "${N_TRIALS_MAIN}" \
    --force

echo "  Sweep matrix complete."

# ---------------------------------------------------------------------------
# 3. Layer sensitivity (per-layer mismatch attribution)
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Running layer sensitivity analysis (n_trials=${N_TRIALS_LAYER})..."
python layer_sensitivity.py \
    --n-trials "${N_TRIALS_LAYER}" \
    --force

echo "  Layer sensitivity complete."

# ---------------------------------------------------------------------------
# 4. Transient stability (long-term drift, 2 architectures)
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Running transient stability experiment (N=${N_INFER_TRANSIENT})..."
python transient_stability.py \
    --n-infer "${N_INFER_TRANSIENT}" \
    --sigma 0.05

echo "  Transient stability complete."

# ---------------------------------------------------------------------------
# 5. Generate all figures
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Generating figures..."

# Core figures from sweep results (default domain=conservative)
python plot_results.py

# Substrate comparison figure (uses existing results)
python plot_substrate_comparison.py

# Heatmaps (analog/digital partition maps)
python plot_heatmaps.py

echo "  All figures generated."

# ---------------------------------------------------------------------------
# 6. Package outputs for download
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Packaging results..."
OUTPUT_TAR="/workspace/neuro-analog-pilot-results.tar.gz"

tar -czf "${OUTPUT_TAR}" \
    -C "${ROOT_DIR}" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "experiments/cross_arch_tolerance/results" \
    "experiments/cross_arch_tolerance/figures" \
    "experiments/cross_arch_tolerance/checkpoints"

echo ""
echo "========================================"
echo "PIPELINE COMPLETE"
echo "========================================"
echo "Results + figures + checkpoints packaged:"
echo "  ${OUTPUT_TAR}"
echo ""
echo "To download to your local machine:"
echo "  scp root@<runpod-ip>:${OUTPUT_TAR} ."
echo ""
echo "Key outputs:"
echo "  - results/           : All JSON sweep results (mismatch, ablation, ADC, MSE, profiles)"
echo "  - figures/           : fig1–fig9, heatmaps, transient stability"
echo "  - checkpoints/       : Trained model weights"
echo "========================================"
