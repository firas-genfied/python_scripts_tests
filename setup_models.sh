#!/usr/bin/env bash
set -e

# Backbone (ViT-JX) from timm releases
BACKBONE_URL="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
BACKBONE_FILE="jx_vit_base_p16_224-80ecf9dd.pth"
BACKBONE_DIR="/home/tanay/workstation/genfied_workstation/ObjectTracking/TransReID/.cache/torch/checkpoints"
BACKBONE_PATH="$BACKBONE_DIR/$BACKBONE_FILE"

# Fine-tuned TransReID model
FINETUNE_URL="https://genfied.blob.core.windows.net/models/vit_base_msmt.pth"
FINETUNE_FILE="vit_base_msmt.pth"
FINETUNE_DIR="/home/tanay/workstation/genfied_workstation/ObjectTracking/TransReID/models"
FINETUNE_PATH="$FINETUNE_DIR/$FINETUNE_FILE"

# Ensure directories exist
mkdir -p "$BACKBONE_DIR" "$FINETUNE_DIR"

# Download backbone if missing
if [ -f "$BACKBONE_PATH" ]; then
  echo "✅ Backbone model already exists: $BACKBONE_PATH"
else
  echo "🔽 Downloading backbone to $BACKBONE_PATH"
  curl -L "$BACKBONE_URL" -o "$BACKBONE_PATH"
fi

# Download fine-tuned if missing
if [ -f "$FINETUNE_PATH" ]; then
  echo "✅ Fine-tuned model already exists: $FINETUNE_PATH"
else
  echo "🔽 Downloading fine-tuned model to $FINETUNE_PATH"
  curl -L "$FINETUNE_URL" -o "$FINETUNE_PATH"
fi
