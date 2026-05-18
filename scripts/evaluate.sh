#!/bin/bash

# Dynamic3DGS Evaluation Script
# Usage: ./scripts/evaluate.sh [OPTIONS]

set -e  # Exit on any error

# Default configuration
CHECKPOINT_PATH="./checkpoints/latest.pth"
DATASET="nvidia_dynamic"
DATA_PATH="./data/nvidia_dynamic"
OUTPUT_DIR="./evaluation_results/$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=4
NUM_BATCHES=20
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-batches)
            NUM_BATCHES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --render-videos)
            RENDER_VIDEOS=true
            shift
            ;;
        --save-renders)
            SAVE_RENDERS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH    Checkpoint file to evaluate (default: ./checkpoints/latest.pth)"
            echo "  --dataset NAME       Dataset name (nvidia_dynamic, hypernerf, custom)"
            echo "  --data-path PATH     Dataset path"
            echo "  --output-dir DIR     Output directory for results"
            echo "  --batch-size SIZE    Batch size for evaluation (default: 4)"
            echo "  --num-batches N      Number of batches to evaluate (default: 20)"
            echo "  --device DEVICE      Device to use (default: cuda)"
            echo "  --metrics LIST       Comma-separated list of metrics to compute"
            echo "  --render-videos      Generate video visualizations"
            echo "  --save-renders       Save individual rendered frames"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Dynamic3DGS Evaluation Setup ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $DATASET"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path does not exist: $DATA_PATH"
    exit 1
fi

# Set up environment variables
export PYTHONPATH="${PWD}:$PYTHONPATH"

# Run evaluation
echo "Starting evaluation..."

python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --dataset "$DATASET" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --device "$DEVICE" \
    ${METRICS:+--metrics "$METRICS"} \
    ${RENDER_VIDEOS:+--render-videos} \
    ${SAVE_RENDERS:+--save-renders}

echo ""
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Available files:"
ls -la "$OUTPUT_DIR"

# Optional: Display results summary
if [ -f "$OUTPUT_DIR/metrics.json" ]; then
    echo ""
    echo "=== Results Summary ==="
    python -c "
import json
with open('$OUTPUT_DIR/metrics.json', 'r') as f:
    metrics = json.load(f)
for key, value in metrics.items():
    print(f'{key}: {value:.4f}')
"
fi

# Optional: Launch TensorBoard for evaluation results
read -p "Would you like to launch TensorBoard for evaluation results? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] && [ -d "$OUTPUT_DIR/tensorboard" ]; then
    echo "Launching TensorBoard..."
    tensorboard --logdir "$OUTPUT_DIR/tensorboard" --port=6007 &
    sleep 2
    echo "TensorBoard running at: http://localhost:6007"
    echo "Press Ctrl+C to stop TensorBoard when done."
    wait
fi