# Use the provided NVIDIA CUDA base image with cuDNN 8 on Ubuntu 20.04
# FROM nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu20.04
# FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu20.04
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu20.04



# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites and add deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install Python 3.10.12 and related packages along with other system dependencies
RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    build-essential \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure that "python3" points to Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Clone your repository (with submodules)
RUN git clone --recursive https://github.com/anuj018/ObjectTracking.git . 

# Install Python dependencies from your curated requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Detectron2 from GitHub
RUN pip3 install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Set default command to run your main processor script
# CMD ["python3", "processor_segment_with_transreid.py"]
CMD ["/bin/bash"]
