#!/bin/bash
set -e

# Set up CUDA visible devices if provided
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

# Configure memory limit for PyTorch
if [ ! -z "$GPU_MEMORY_FRACTION" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
fi

# Create log directory
mkdir -p /app/logs

# Run the processor with arguments passed to this script
echo "Starting RTSP processor with command: $@"
exec "$@"