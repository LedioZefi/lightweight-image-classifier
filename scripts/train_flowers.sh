#!/bin/bash
# Train lightweight image classifier on tf_flowers dataset
# 
# Usage:
#   ./scripts/train_flowers.sh [OPTIONS]
#
# Options:
#   --epochs N              Number of training epochs (default: 10)
#   --batch-size N          Training batch size (default: 32)
#   --img-size N            Image size (default: 224)
#   --artifacts-dir PATH    Directory to save models (default: artifacts)
#   --export-quantizations  TFLite quantization formats (default: float32 dynamic float16)
#   --no-export             Skip TFLite export
#   --help                  Show this help message

set -e

# Default values
EPOCHS=10
BATCH_SIZE=32
IMG_SIZE=224
ARTIFACTS_DIR="artifacts"
EXPORT_QUANTIZATIONS="float32 dynamic float16"
NO_EXPORT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --img-size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --artifacts-dir)
            ARTIFACTS_DIR="$2"
            shift 2
            ;;
        --export-quantizations)
            EXPORT_QUANTIZATIONS="$2"
            shift 2
            ;;
        --no-export)
            NO_EXPORT=true
            shift
            ;;
        --help)
            grep "^#" "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=========================================="
echo "Training Lightweight Image Classifier"
echo "=========================================="
echo "Dataset:              tf_flowers"
echo "Epochs:               $EPOCHS"
echo "Batch Size:           $BATCH_SIZE"
echo "Image Size:           ${IMG_SIZE}x${IMG_SIZE}"
echo "Artifacts Directory:  $ARTIFACTS_DIR"
if [ "$NO_EXPORT" = false ]; then
    echo "Export Quantizations: $EXPORT_QUANTIZATIONS"
else
    echo "Export:               Disabled"
fi
echo "=========================================="
echo ""

# Run training
cd "$(dirname "$0")/.."

PYTHON_CMD="python -m src.image_classifier"
CMD="$PYTHON_CMD --dataset tf_flowers --epochs $EPOCHS --batch-size $BATCH_SIZE --img-size $IMG_SIZE --artifacts-dir $ARTIFACTS_DIR"

if [ "$NO_EXPORT" = true ]; then
    CMD="$CMD --no-export"
else
    CMD="$CMD --export-quantizations $EXPORT_QUANTIZATIONS"
fi

echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "=========================================="
echo "Training complete!"
echo "Models saved to: $ARTIFACTS_DIR/"
echo "=========================================="

