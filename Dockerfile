# Use the provided NVIDIA CUDA base image with cuDNN 8 on Ubuntu 20.04
# FROM nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu20.04

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV BASE_DIR=/app

# Install prerequisites and add deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install Python 3.10, pip, and related packages along with other system dependencies
RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    software-properties-common \
    apt-transport-https \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ensure that "python3" points to Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1


# Download and run get-pip.py to install the latest pip (overriding the system pip)
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py --break-system-packages && \
    rm get-pip.py

# Install PostgreSQL client tools (useful for debugging)
# RUN apt-get update && apt-get install -y postgresql-client

# Set working directory
WORKDIR /app

# Clone your repository (with submodules)
RUN git clone --recursive https://github.com/anuj018/ObjectTracking.git . 

RUN echo "Python version:" && python3 --version
RUN echo "Pip version:" && pip3 --version

# Install Python dependencies from your curated requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt
# Install pytorch packages first
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0+cu126 \
    torchvision==0.21.0+cu126 \
    torchaudio==2.6.0+cu126
    
RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt

# Install Detectron2 from GitHub
RUN pip3 install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Set the working directory to TransReID (adjust the base path as needed)
WORKDIR /app/TransReID

# Create the necessary cache directories recursively
RUN mkdir -p .cache/torch/checkpoints

# Download the model and save it in the checkpoints directory
RUN curl -L https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth \
    -o .cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth

RUN mkdir -p models
RUN curl -L -o models/vit_base_msmt.pth "https://genfied.blob.core.windows.net/models/vit_base_msmt.pth"

WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
# COPY /app/entrypoint.sh /app/entrypoint.sh

# RUN sudo chmod +x /app/entrypoint.sh


# # Create entrypoint script
# RUN echo '#!/bin/bash\n\
# set -e\n\
# \n\
# # Set up CUDA visible devices if provided\n\
# if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then\n\
#     echo "Using GPUs: $CUDA_VISIBLE_DEVICES"\n\
# fi\n\
# \n\
# # Configure memory limit for PyTorch\n\
# if [ ! -z "$GPU_MEMORY_FRACTION" ]; then\n\
#     export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"\n\
# fi\n\
# \n\
# # Create log directory\n\
# mkdir -p /app/logs\n\
# \n\
# # Run the processor with arguments passed to this script\n\
# echo "Starting RTSP processor with command: $@"\n\
# exec "$@"\n\
# ' > /app/entrypoint.sh

# RUN chmod +x /app/entrypoint.sh


# Set default command to run your main processor script
# CMD ["python3", "processor_segment_with_transreid.py"]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "rtsp_stream_processor_multiple_cameras.py", "--config", "/app/config/camera_config.json"]
# CMD ["/bin/bash"]
