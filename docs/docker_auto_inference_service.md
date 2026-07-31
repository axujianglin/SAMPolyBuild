# SAMPoly Docker Auto Inference Service

This service exposes the existing SAMPoly Auto prediction flow to a native
Windows client without changing the model, checkpoint, or post-processing
implementation.

## Architecture

```text
SHORTER (Windows Qt GUI)
  -> HTTP multipart upload
  -> Docker Desktop / WSL2 port 127.0.0.1:8000
  -> FastAPI adapter
  -> AutoInferenceService
  -> infer_auto.run_auto_inference()
  -> validated original-image polygon JSON
  -> SHORTER overlay rendering
```

The current service intentionally creates the existing runner for every
request. This preserves command-line inference behavior and minimizes changes
to Layer 2. A persistent Auto runtime is a separate future optimization.

## Prerequisites

1. Windows 10/11 with a supported NVIDIA driver.
2. Docker Desktop using the WSL2 Linux backend.
3. GPU access verified from Docker.
4. The WHU Auto checkpoint available at `checkpoints/auto_whumix.pth`.

Verify the GPU before building SAMPoly:

```powershell
docker run --rm --gpus all `
  nvidia/cuda:11.7.1-base-ubuntu22.04 `
  nvidia-smi
```

## Filesystem layout

The Compose configuration expects:

```text
SAMPolyBuild/
  checkpoints/
    auto_whumix.pth
  configs/
    auto_whumix.py
  docker/
    Dockerfile
    compose.yaml
```

The checkpoint directory is mounted read-only. Checkpoints, datasets, TIFFs,
and runtime outputs are excluded from the Docker build context.

## Build and start

Run from the repository root:

```powershell
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml up -d
docker compose -f docker/compose.yaml logs -f sampoly-auto-api
```

The API listens inside the container on `0.0.0.0:8000`, but Compose publishes
it only on the Windows loopback address:

```text
http://127.0.0.1:8000
```

It is not exposed to the local network by default.

## Health endpoint

```powershell
curl.exe http://127.0.0.1:8000/health
```

Ready response:

```json
{
  "status": "ready",
  "service": "sampoly-auto-inference",
  "config_exists": true,
  "checkpoint_exists": true,
  "cuda_available": true,
  "cuda_error": null,
  "device": "cuda:0",
  "busy": false,
  "inference_lifecycle": "runner_per_request"
}
```

`/health` does not load the model. A `not_ready` response identifies a missing
config, missing checkpoint, or unavailable CUDA runtime.

## Auto inference endpoint

The endpoint accepts one `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff` image.

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/auto/infer `
  -F "image=@D:\data\WHU\image001.tiff" `
  -F "score_threshold=0.1" `
  -o auto_response.json
```

Successful response:

```json
{
  "success": true,
  "request_id": "6fce7db0f13e4c328b637ca7f1656428",
  "image": {
    "file_name": "image001.tiff",
    "image_id": "image001",
    "width": 1024,
    "height": 1024
  },
  "coordinate_space": "original_image_pixel",
  "polygons": [
    {
      "id": "image001_poly_0001",
      "score": 0.91,
      "score_cls": 0.94,
      "bbox_xywh": [100.0, 120.0, 180.0, 190.0],
      "points": [[110.0, 130.0], [270.0, 125.0], [275.0, 300.0]],
      "metadata": {
        "raw_index": 0,
        "category_id": 1,
        "area": 24830
      }
    }
  ],
  "summary": {
    "polygon_count": 1,
    "score_threshold": 0.1
  }
}
```

`points` and `bbox_xywh` use original image pixel coordinates. The GUI should
draw these coordinates over the original image and apply zoom through its view
transform rather than modifying polygon coordinates.

## Error contract

All service failures use a stable envelope:

```json
{
  "success": false,
  "error": {
    "code": "unsupported_image_format",
    "message": "Supported suffixes: .jpeg, .jpg, .png, .tif, .tiff"
  }
}
```

Important error codes include:

- `missing_file_name`
- `unsupported_image_format`
- `empty_upload`
- `upload_too_large`
- `invalid_image`
- `invalid_score_threshold`
- `request_validation_error`
- `config_not_found`
- `checkpoint_not_found`
- `cuda_unavailable`
- `inference_failed`
- `invalid_inference_results`
- `internal_error`

## Configuration

The container supports the following environment variables:

| Variable | Default |
| --- | --- |
| `SAMPOLY_AUTO_CONFIG` | `/app/configs/auto_whumix.py` |
| `SAMPOLY_AUTO_CHECKPOINT` | `/app/checkpoints/auto_whumix.pth` |
| `SAMPOLY_RUNTIME_ROOT` | `/app/runtime` |
| `SAMPOLY_DEVICE` | `cuda` |
| `SAMPOLY_GPU` | `0` |
| `SAMPOLY_BATCH_SIZE` | `1` |
| `SAMPOLY_NUM_WORKERS` | `1` |
| `SAMPOLY_DEFAULT_SCORE_THRESHOLD` | `0.1` |
| `SAMPOLY_MAX_UPLOAD_MB` | `100` |

The service writes each upload and its intermediate artifacts into a unique
temporary runtime directory. The directory is removed after the response is
created, including failed requests.

## Stop and inspect

```powershell
docker compose -f docker/compose.yaml ps
docker compose -f docker/compose.yaml logs sampoly-auto-api
docker compose -f docker/compose.yaml down
```

Do not add `-v` to `down` unless the named runtime volume should also be
deleted.

## Model-free tests

Install only the API dependencies in the existing Linux `sampoly` environment:

```bash
python -m pip install -r docker/requirements-api.txt
```

Run:

```bash
PYTHONPATH=. python -m unittest \
  tests.test_auto_result_adapter \
  tests.test_auto_inference_service \
  tests.test_auto_api_contract
```

If the repository's `tests` directory is not importable as a package, use:

```bash
PYTHONPATH=. python -m unittest discover -s tests -p "test_auto_*.py"
```

## Current limitations

- The runner and checkpoint are loaded for every request.
- GPU inference is serialized by one process-local lock.
- `infer_auto.py` currently limits output to 100 instances per image.
- The endpoint is synchronous; large-image job queues are not part of this stage.
- Prompt, positive/negative points, and SHORTER integration are not included.
