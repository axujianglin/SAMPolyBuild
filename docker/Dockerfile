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
        libgl1 \
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

# Preserve the vendored source layout because PythonAPI/setup.py compiles the
# sibling common/maskApi.c source file into the native _mask extension.
COPY pycocotools /tmp/pycocotools

# The legacy setup.py imports NumPy while preparing the wheel but does not
# declare NumPy as a build dependency. Build against the project NumPy already
# installed above and provide Cython explicitly for the _mask.pyx source.
RUN python3.10 -m pip install Cython==0.29.36 \
    && python3.10 -m pip install \
        --no-build-isolation \
        /tmp/pycocotools/PythonAPI

# Fail the image build before deployment if native graphics libraries or the
# pinned legacy Lightning Cloud API become incompatible with the runtime.
RUN python3.10 -c "import cv2; import lightning; import mmcv; import pycocotools._mask; from lightning_cloud.openapi import AppinstancesIdBody; print('SAMPoly Docker runtime imports OK')"

COPY . /app
RUN mkdir -p /app/runtime /app/checkpoints

EXPOSE 8000 8010

CMD ["python3.10", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
