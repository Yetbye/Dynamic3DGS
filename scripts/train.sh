#!/bin/bash

# Dynamic3DGS Training Script
# Usage: ./scripts/train.sh [OPTIONS]

set -e  # Exit on any error

# Default configuration
CONFIG_FILE="configs/experiments/final_model.yaml"
DATA_PATH="./data/nvidia_dynamic"
EXPERIMENT_NAME="dynamic_3dgs_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="./checkpoints/$EXPERIMENT_NAME"
LOG_DIR="./logs/$EXPERIMENT_NAME"
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE          Configuration file (default: configs/experiments/final_model.yaml)"
            echo "  --data-path PATH     Dataset path (default: ./data/nvidia_dynamic)"
            echo "  --experiment-name NAME Experiment name (default: auto-generated)"
            echo "  --checkpoint-dir DIR   Checkpoint directory (default: ./checkpoints/NAME)"
            echo "  --log-dir DIR        Log directory (default: ./logs/NAME)"
            echo "  --device DEVICE      Device to use (default: cuda)"
            echo "  --resume CHECKPOINT  Resume from checkpoint"
            echo "  --debug              Enable debug mode"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Dynamic3DGS Training Setup ==="
echo "Config: $CONFIG_FILE"
echo "Data Path: $DATA_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Log Dir: $LOG_DIR"
echo "Device: $DEVICE"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Warning: Data path does not exist: $DATA_PATH"
    echo "Please prepare your dataset first."
    exit 1
fi

# Check for required Python packages
echo "Checking dependencies..."
python -c "
try:
    import torch, yaml, numpy, PIL, cv2
    print('✓ All required packages installed')
except ImportError as e:
    print(f'✗ Missing package: {e}')
    exit(1)
" || {
    echo "Installing required packages..."
    pip install -r requirements.txt || {
        echo "Failed to install packages. Please install manually:"
        echo "pip install torch torchvision numpy pyyaml opencv-python pillow"
        exit 1
    }
}

# Set up environment variables
export PYTHONPATH="${PWD}:$PYTHONPATH"

# Run training
echo ""
echo "Starting training..."

if [ "$DEBUG_MODE" = true ]; then
    echo "Running in debug mode with reduced settings..."
    python main.py \
        --config "$CONFIG_FILE" \
        --data-path "$DATA_PATH" \
        --experiment-name "$EXPERIMENT_NAME" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --log-dir "$LOG_DIR" \
        --device "$DEVICE" \
        --debug-mode
else
    python main.py \
        --config "$CONFIG_FILE" \
        --data-path "$DATA_PATH" \
        --experiment-name "$EXPERIMENT_NAME" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --log-dir "$LOG_DIR" \
        --device "$DEVICE" \
        ${RESUME_CHECKPOINT:+--resume "$RESUME_CHECKPOINT"}
fi

echo ""
echo "Training completed!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Logs available at: $LOG_DIR"

# Optional: Launch TensorBoard
read -p "Would you like to launch TensorBoard? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Launching TensorBoard..."
    tensorboard --logdir "$LOG_DIR" --port=6006 &
    sleep 2
    echo "TensorBoard running at: http://localhost:6006"
    echo "Press Ctrl+C to stop TensorBoard when done."
    wait
fi