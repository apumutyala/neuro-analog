#!/bin/bash

# Parallel training launcher for WikiText-2 language models
# Launches 7 training jobs (one per architecture) on separate GPUs

echo "========================================"
echo "WikiText-2 Training Launcher"
echo "========================================"
echo "Starting parallel training for 7 architectures..."
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints/wikitext2

# Architecture list
ARCHS=("transformer" "s4d" "neural_ode" "deq" "flow" "ebm" "diffusion")

# Launch training jobs in background
for i in "${!ARCHS[@]}"; do
    ARCH="${ARCHS[$i]}"
    GPU=$i  # Assign one GPU per architecture
    
    echo "[$i/7] Launching $ARCH on GPU $GPU..."
    
    CUDA_VISIBLE_DEVICES=$GPU python train_wikitext2.py \
        --arch $ARCH \
        --device cuda \
        --checkpoint-dir checkpoints/wikitext2 \
        --log-dir logs \
        --batch-size 32 \
        --learning-rate 3e-4 \
        --max-epochs 100 \
        > logs/${ARCH}_lm_stdout.log 2>&1 &
    
    # Store process ID
    PID=$!
    echo "  → PID: $PID"
    echo "  → Log: logs/${ARCH}_lm_train.log"
    echo ""
    
    # Small delay to avoid race conditions
    sleep 2
done

echo "========================================"
echo "All training jobs launched!"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/transformer_lm_train.log"
echo "  tail -f logs/s4d_lm_train.log"
echo "  ... etc"
echo ""
echo "Or check all at once:"
echo "  watch -n 30 'tail -n 3 logs/*_lm_train.log'"
echo ""
echo "Kill all jobs:"
echo "  pkill -f train_wikitext2.py"
echo "========================================"

# Wait for all background jobs
wait

echo ""
echo "========================================"
echo "All training jobs completed!"
echo "========================================"
echo ""
echo "Check results:"
echo "  ls -lh checkpoints/wikitext2/"
echo ""
echo "View summaries:"
for ARCH in "${ARCHS[@]}"; do
    if [ -f "checkpoints/wikitext2/${ARCH}_lm_results.json" ]; then
        echo "  cat checkpoints/wikitext2/${ARCH}_lm_results.json"
    fi
done
