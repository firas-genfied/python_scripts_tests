#!/bin/bash
set -e

echo "🔽 Downloading TransReID pretrained model..."
MODEL_PATH="/app/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"

if [ -f "$MODEL_PATH" ]; then
  echo "✅ Model already exists: $MODEL_PATH"
else
  echo "🔽 Downloading backbone model..."
  mkdir -p $(dirname "$MODEL_PATH")
  curl -L https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth \
    -o "$MODEL_PATH"
fi

# # Create the required directory for PyTorch checkpoints
# mkdir -p /app/.cache/torch/checkpoints

# # Download the vision transformer backbone weights
# curl -L https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth \
#   -o /app/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth

# echo "✅ Backbone model downloaded."

echo "🔽 Downloading TransReID fine-tuned model..."
FINE_TUNED_PATH="/app/models/vit_base_msmt.pth"

if [ -f "$FINE_TUNED_PATH" ]; then
  echo "✅ Fine-tuned model already exists: $FINE_TUNED_PATH"
else
  echo "🔽 Downloading fine-tuned model..."
  mkdir -p $(dirname "$FINE_TUNED_PATH")
  curl -L -o "$FINE_TUNED_PATH" "https://genfied.blob.core.windows.net/models/vit_base_msmt.pth"
fi