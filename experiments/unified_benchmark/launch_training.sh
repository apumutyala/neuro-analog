#!/bin/bash
#
# Launch all 7 architecture training jobs in parallel on RunPod.
# Each architecture gets its own GPU.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================="
echo "CIFAR-10 Unified Training - Parallel Launch"
echo "======================================================================="

# Architecture list
ARCHS=("neural_ode" "s4d" "deq" "diffusion" "flow" "ebm" "transformer")

# Check GPU availability
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ $NUM_GPUS -lt 7 ]; then
    echo "WARNING: Only $NUM_GPUS GPUs available, training will be serialized"
fi

# Create checkpoint directory
mkdir -p checkpoints/cifar10
mkdir -p logs

# Launch training jobs
for i in "${!ARCHS[@]}"; do
    ARCH="${ARCHS[$i]}"
    GPU_ID=$((i % NUM_GPUS))
    
    echo ""
    echo "Launching $ARCH on GPU $GPU_ID..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_cifar10.py \
        --arch $ARCH \
        --device cuda \
        > logs/${ARCH}_train.log 2>&1 &
    
    # Store PID for monitoring
    echo $! > logs/${ARCH}_train.pid
    
    # Small delay to avoid simultaneous data downloads
    sleep 5
done

echo ""
echo "======================================================================="
echo "All training jobs launched."
echo "Monitor progress with: tail -f logs/*.log"
echo "======================================================================="

# Wait for all background jobs
wait

echo ""
echo "======================================================================="
echo "Training complete for all architectures."
echo "======================================================================="

# Print final results
echo ""
echo "Final accuracies:"
for ARCH in "${ARCHS[@]}"; do
    if [ -f "checkpoints/cifar10/${ARCH}_cifar10_best.pt" ]; then
        ACC=$(grep "best acc:" logs/${ARCH}_train.log | tail -1 | grep -oP '\d+\.\d+')
        echo "  $ARCH: ${ACC}%"
    else
        echo "  $ARCH: FAILED (no checkpoint found)"
    fi
done
