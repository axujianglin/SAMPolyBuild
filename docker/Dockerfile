FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SAMPOLY_AUTO_CONFIG=/app/configs/auto_whumix.py \
    SAMPOLY_AUTO_CHECKPOINT=/app/checkpoints/auto_whumix.pth \
    SAMPOLY_RUNTIME_ROOT=/app/runtime \
    SAMPOLY_DEVICE=cuda \
    SAMPOLY_GPU=0 \
    SAMPOLY_BATCH_SIZE=1 \
    SAMPOLY_NUM_WORKERS=1 \
    SAMPOLY_DEFAULT_SCORE_THRESHOLD=0.1 \
    SAMPOLY_MAX_UPLOAD_MB=100

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libglib2.0-0 \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
COPY docker/requirements-api.txt /tmp/requirements-api.txt

# Keep the model runtime aligned with the versions documented by SAMPoly.
RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && python3.10 -m pip install \
        torch==2.0.0+cu117 \
        torchvision==0.15.1+cu117 \
        --index-url https://download.pytorch.org/whl/cu117 \
    && sed '/^mmcv==/d' /tmp/requirements.txt > /tmp/requirements-docker.txt \
    && python3.10 -m pip install \
        -r /tmp/requirements-docker.txt \
        -r /tmp/requirements-api.txt \
    && python3.10 -m pip install \
        mmcv==2.0.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

COPY pycocotools/PythonAPI /tmp/pycocotools
RUN python3.10 -m pip install /tmp/pycocotools

COPY . /app
RUN mkdir -p /app/runtime /app/checkpoints

EXPOSE 8000

CMD ["python3.10", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
