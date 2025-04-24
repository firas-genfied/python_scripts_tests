# Use the provided NVIDIA CUDA base image with cuDNN 8 on Ubuntu 20.04
# FROM nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV BASE_DIR=/app
WORKDIR /app

# Install prerequisites and add deadsnakes PPA for Python 3.10
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    lsb-release \
    gnupg2 \
    apt-transport-https \
    software-properties-common \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg
# Install Python 3.10, pip, and related packages along with other system dependencies
# RUN apt-get install -y --no-install-recommends \
#     python3.10 \
#     python3.10-dev \
#     python3.10-venv \
#     python3-pip \
#     build-essential \
#     software-properties-common \
#     apt-transport-https \
#     git \
#     curl \
#     ca-certificates \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgl1-mesa-glx \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

# Ensure that "python3" points to Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1


# Download and run get-pip.py to install the latest pip (overriding the system pip)
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py --break-system-packages && \
    rm get-pip.py

# Install PostgreSQL client tools (useful for debugging)
# RUN apt-get update && apt-get install -y postgresql-client

# Set working directory

# Copy your code into the container
COPY . /app

# Clone your repository (with submodules)
# RUN git clone --recursive https://github.com/anuj018/ObjectTracking.git . 

RUN echo "Python version:" && python3 --version
RUN echo "Pip version:" && pip3 --version

# Install Python dependencies from your curated requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt
# Install pytorch packages first
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0+cu126 \
    torchvision==0.21.0+cu126 \
    torchaudio==2.6.0+cu126
    
RUN pip3 install --no-cache-dir --ignore-installed --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt

# Install Detectron2 from GitHub
RUN pip3 install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

COPY setup_models.sh /app/setup_models.sh
RUN chmod +x /app/setup_models.sh && /app/setup_models.sh

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "rtsp_stream_processor_multiple_cameras.py", "--config", "/app/config/camera_config.json"]
# CMD ["/bin/bash"]
