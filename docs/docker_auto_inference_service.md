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

The API dependencies deliberately use FastAPI `0.88.0`, Uvicorn `0.20.0`, and
HTTPX `0.23.3`. The existing SAMPoly runtime pins `lightning==2.0.1`, whose
package metadata requires `fastapi<0.89.0`; using a newer FastAPI in the same
Python environment makes pip fail with `ResolutionImpossible`. Keeping the API
adapter on the compatible release line avoids changing the model and inference
runtime. After a successful build, verify dependency consistency with:

```shell
docker run --rm --entrypoint python3.10 sampoly-auto-api:cu117 -m pip check
```

The expected output is `No broken requirements found.`

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
- `unauthorized`
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
| `SAMPOLY_API_KEY` | empty (authentication disabled) |

The service writes each upload and its intermediate artifacts into a unique
temporary runtime directory. The directory is removed after the response is
created, including failed requests.

When `SAMPOLY_API_KEY` is non-empty, `/v1/auto/infer` requires the exact value
in the `X-API-Key` header. `/health` intentionally remains unauthenticated so
Docker and infrastructure health checks can determine readiness. Do not expose
an unauthenticated service to an untrusted network.

## Build once and deploy to another machine

Containerization has three separate deliverables:

1. `docker/Dockerfile` is the reproducible image recipe.
2. `sampoly-auto-api:cu117` is the built image containing SAMPoly, CUDA runtime,
   Python dependencies, configuration, and the HTTP service.
3. `auto_whumix.pth` is deployed separately and mounted read-only. It is not
   copied into the image because of its size and possible distribution terms.

The machine that builds the image needs access to Docker Hub, PyTorch,
OpenMMLab, Python package repositories, and the system package repositories.
After an image archive has been produced, an offline target host does not need
those package repositories.

### Export on the build machine

Build and export on Windows:

```powershell
docker compose -f docker/compose.yaml build --progress=plain
powershell -ExecutionPolicy Bypass -File docker/export-image.ps1 `
  -Image sampoly-auto-api:cu117 `
  -Output sampoly-auto-api-cu117.tar
```

Build and export on Linux:

```bash
docker compose -f docker/compose.yaml build --progress=plain
sh docker/export-image.sh sampoly-auto-api:cu117 sampoly-auto-api-cu117.tar
```

Copy the following deployment bundle to the target GPU server:

```text
sampoly-deploy/
  sampoly-auto-api-cu117.tar
  docker/
    compose.deploy.yaml
    .env.example
    deploy-image.ps1
    deploy-image.sh
    verify-remote.ps1
    verify-remote.sh
  checkpoints/
    auto_whumix.pth
```

Before distributing a checkpoint, verify that its license permits transfer to
the target machine.

### Configure the target host

The target host requires Docker with the Compose plugin, a compatible NVIDIA
driver, and NVIDIA container GPU access. It does not require a local SAMPoly
Python environment. First verify GPU access:

```shell
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi
```

Copy `docker/.env.example` to `docker/.env`, then set at least:

```dotenv
SAMPOLY_IMAGE=sampoly-auto-api:cu117
SAMPOLY_BIND_HOST=0.0.0.0
SAMPOLY_API_PORT=8000
SAMPOLY_CHECKPOINT_DIR=../checkpoints
SAMPOLY_CHECKPOINT_FILE=auto_whumix.pth
SAMPOLY_API_KEY=replace-with-a-long-random-secret
```

All paths supplied to Compose must be valid on the target host. Use an absolute
checkpoint directory if the deployment files can be launched from different
working directories.

### Import and start on Windows

```powershell
Copy-Item docker/.env.example docker/.env
# Edit docker/.env before starting the service.
powershell -ExecutionPolicy Bypass -File docker/deploy-image.ps1 `
  -Archive sampoly-auto-api-cu117.tar `
  -EnvFile docker/.env
```

### Import and start on Linux

```bash
cp docker/.env.example docker/.env
# Edit docker/.env before starting the service.
sh docker/deploy-image.sh sampoly-auto-api-cu117.tar docker/.env
```

The deployment Compose file uses `image:` without `build:`. Consequently, the
target machine uses the imported image and never attempts to rebuild SAMPoly.

### Deploy through a trusted registry instead

An organization with a private registry can replace archive transfer:

```shell
docker tag sampoly-auto-api:cu117 registry.example.com/sampoly-auto-api:1.0.0
docker push registry.example.com/sampoly-auto-api:1.0.0
```

Set `SAMPOLY_IMAGE=registry.example.com/sampoly-auto-api:1.0.0` in the target
`.env`, run `docker pull`, and start `compose.deploy.yaml`. Registry credentials
and TLS configuration are infrastructure responsibilities and must not be
stored in the repository.

## Remote client verification

Assume the GPU server address is `192.168.1.100` and TCP port `8000` is allowed
by its host firewall. From a different Windows machine:

```powershell
powershell -ExecutionPolicy Bypass -File docker/verify-remote.ps1 `
  -BaseUrl http://192.168.1.100:8000 `
  -ApiKey replace-with-a-long-random-secret `
  -Image D:\data\WHU\image001.tiff `
  -Output auto-response.json
```

The equivalent direct HTTP request is:

```powershell
curl.exe -X POST "http://192.168.1.100:8000/v1/auto/infer" `
  -H "X-API-Key: replace-with-a-long-random-secret" `
  -F "image=@D:\data\WHU\image001.tiff" `
  -F "score_threshold=0.1" `
  -o auto-response.json
```

From Linux:

```bash
sh docker/verify-remote.sh \
  http://192.168.1.100:8000 \
  replace-with-a-long-random-secret \
  /data/WHU/image001.tiff \
  auto-response.json
```

The client only needs an HTTP client. It does not need CUDA, Python, SAMPoly,
or the checkpoint. Returned polygon points use original image pixel
coordinates, so a desktop GUI can draw them directly over the uploaded image.

## Network and security boundary

`compose.yaml` remains restricted to `127.0.0.1` for local development.
`compose.deploy.yaml` defaults to `0.0.0.0` for cross-machine access. Opening a
host firewall port is an explicit deployment action; Docker configuration does
not replace Windows Defender Firewall, Linux firewall, router, or cloud security
group rules.

For a trusted LAN, use a strong API key and restrict port `8000` to the client
subnet. For Internet access, place the service behind an HTTPS reverse proxy or
API gateway with authentication, request-size limits, timeouts, and audit
logging. Do not publish the raw unauthenticated Uvicorn endpoint to the public
Internet.

## Portability limits

The provided image targets Linux `x86_64`, CUDA 11.7, PyTorch 2.0, and NVIDIA
GPU execution. It can run through Docker Desktop/WSL2 on Windows or Docker
Engine on Linux when the host driver supports the container CUDA runtime.
Running the same artifact on ARM hardware, a non-NVIDIA accelerator, or a host
without compatible GPU container support is outside the validated deployment
target. Client machines have no GPU requirement.

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
