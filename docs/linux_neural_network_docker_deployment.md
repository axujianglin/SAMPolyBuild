# Linux 环境下神经网络模型 Docker 构建与部署操作手册

本文档说明如何在一台 Linux 构建机上将神经网络推理服务封装为 Docker
镜像，完成本机验证后导出镜像，并部署到另一台 Linux GPU 服务器。示例以
CUDA、PyTorch 和 HTTP 推理接口为基础，同时给出 SAMPoly 项目的直接命令。

> 本手册使用简体中文编写。命令、文件名、环境变量、接口路径及错误码保留其
> 原始英文形式，以确保可以直接复制执行和准确排查问题。

## 1. 部署架构与验收边界

```text
构建机
  ├─ 项目源码、Python依赖、Dockerfile
  ├─ 构建并测试Docker镜像
  ├─ docker save导出tar
  └─ SHA256校验与rsync传输
          ↓
目标服务器
  ├─ 校验tar和模型权重
  ├─ docker load导入镜像
  ├─ Docker Compose启动GPU服务
  └─ 对外提供health和infer接口
          ↑
客户端/构建机
  └─ 通过HTTP完成远程健康检查和真实推理
```

完整验收必须同时证明：镜像能够构建、容器能够启动、GPU可用、真实模型可以
推理、镜像可以跨机器导入、远程客户端可以调用。只完成 `docker save` 不能证明
真实推理或跨机器部署已经成功。

## 2. 规划版本与目录

建议使用固定版本标签，避免只使用含义不明确的 `latest`：

```text
模型服务名：sampoly-auto-api
镜像标签：cu117-1.0.0
镜像全名：sampoly-auto-api:cu117-1.0.0
容器端口：8000
模型权重：auto_whumix.pth
```

推荐项目结构：

```text
project/
  api/
  services/
  configs/
  checkpoints/
    model.pth
  docker/
    Dockerfile
    compose.yaml
    compose.deploy.yaml
    requirements-api.txt
    .env.example
  requirements.txt
  .dockerignore
```

模型权重、数据集、运行结果和导出的镜像归档不应复制进镜像。权重应在启动
容器时通过只读卷挂载，这样可以独立升级代码镜像和权重。

## 3. 构建机环境检查

检查 Linux、Docker 和 Compose：

```bash
uname -a
uname -m
docker --version
docker compose version
```

GPU模型还需要 NVIDIA 驱动和 NVIDIA Container Toolkit：

```bash
nvidia-smi
docker info | grep -i runtime
```

使用与项目CUDA版本相符的基础镜像验证容器GPU：

```bash
docker run --rm --gpus all \
  nvidia/cuda:11.7.1-base-ubuntu22.04 \
  nvidia-smi
```

如果宿主机 `nvidia-smi` 成功而 Docker 命令失败，应先安装或修复 NVIDIA
Container Toolkit。Docker镜像携带CUDA用户态运行库，但不能代替宿主机驱动。

## 4. 编写 Dockerfile

下面是一个GPU推理服务的基础模板。Python、CUDA、PyTorch及系统库版本必须与
实际模型兼容，生产镜像应固定依赖版本。

```dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CHECKPOINT=/app/checkpoints/model.pth \
    RUNTIME_ROOT=/app/runtime \
    MODEL_DEVICE=cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
COPY docker/requirements-api.txt /tmp/requirements-api.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
       -r /tmp/requirements.txt \
       -r /tmp/requirements-api.txt

COPY . /app
RUN mkdir -p /app/checkpoints /app/runtime

# 构建期间尽早发现二进制依赖或版本冲突。
RUN python3 -c "import torch; import cv2; print('runtime imports OK')"

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

SAMPoly应直接使用仓库中的 `docker/Dockerfile`，其中已经固定 PyTorch 2.0、
CUDA 11.7、MMCV、Lightning和vendored pycocotools的兼容构建方式。

## 5. 配置 .dockerignore

构建前必须排除数据、权重、密钥、运行目录和镜像归档：

```dockerignore
.git
.gitignore
.codegraph
**/__pycache__
**/*.pyc
.pytest_cache
.idea
.vscode

checkpoints
dataset
data
runtime
work_dir
work_dir_*

*.pth
*.ckpt
*.tif
*.tiff
*.tar
*.tar.gz
*.tgz
*.zst

.env
docker/.env
auto-response*.json
```

这是重要的安全和体积控制措施。如果项目根目录存在一个12GB镜像tar，而
Dockerfile又执行 `COPY . /app`，没有排除 `*.tar` 会让下次构建上下文和镜像
异常膨胀。没有排除 `.env` 还可能把 API Key 复制进镜像。

## 6. 构建 Docker 镜像

在项目根目录执行：

```bash
docker compose -f docker/compose.yaml build --progress=plain
```

或直接构建：

```bash
docker build \
  -f docker/Dockerfile \
  -t sampoly-auto-api:cu117-1.0.0 \
  .
```

确认镜像：

```bash
docker image inspect sampoly-auto-api:cu117-1.0.0 \
  --format='Image={{.RepoTags}} Size={{.Size}} ID={{.Id}}'
docker image history sampoly-auto-api:cu117-1.0.0
```

CUDA、PyTorch、MMCV等依赖会使未压缩镜像达到数GB甚至十几GB。`docker save`
输出的是未压缩tar，所以归档较大并不等于构建失败。

## 7. 构建机本地验收

### 7.1 检查依赖一致性

```bash
docker run --rm --entrypoint python3.10 \
  sampoly-auto-api:cu117-1.0.0 \
  -m pip check
```

预期输出：

```text
No broken requirements found.
```

### 7.2 启动服务

确认权重存在：

```bash
ls -lh checkpoints/auto_whumix.pth
```

启动并查看日志：

```bash
docker compose -f docker/compose.yaml up -d
docker compose -f docker/compose.yaml ps
docker compose -f docker/compose.yaml logs --tail=200 sampoly-auto-api
```

### 7.3 健康检查

```bash
curl -fsS http://127.0.0.1:8000/health
echo
```

应重点确认：

```json
{
  "status": "ready",
  "config_exists": true,
  "checkpoint_exists": true,
  "cuda_available": true,
  "device": "cuda:0"
}
```

如果旧版curl不支持 `--fail-with-body`，使用兼容写法：

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
```

### 7.4 真实模型推理

不能只测试health，必须使用一张真实输入完成推理：

```bash
curl -fsS \
  -X POST http://127.0.0.1:8000/v1/auto/infer \
  -F "image=@/data/building_test.tif" \
  -F "score_threshold=0.1" \
  -o auto-response.json

python -m json.tool auto-response.json
```

验收时应确认请求成功、检测数量合理、坐标没有非有限值或越界，并对返回轮廓
进行可视化抽查。

## 8. 导出镜像和生成校验文件

建议把归档存放在项目目录之外：

```bash
mkdir -p ~/docker-images

docker save --output \
  ~/docker-images/sampoly-auto-api-cu117-1.0.0.tar \
  sampoly-auto-api:cu117-1.0.0
```

进入归档目录后使用文件名生成校验文件，避免写入构建机绝对路径：

```bash
cd ~/docker-images

sha256sum sampoly-auto-api-cu117-1.0.0.tar \
  > sampoly-auto-api-cu117-1.0.0.tar.sha256

sha256sum -c sampoly-auto-api-cu117-1.0.0.tar.sha256
```

预期：

```text
sampoly-auto-api-cu117-1.0.0.tar: OK
```

可选压缩。压缩后传输更小，但目标服务器需要先解压：

```bash
gzip -1 sampoly-auto-api-cu117-1.0.0.tar
sha256sum sampoly-auto-api-cu117-1.0.0.tar.gz \
  > sampoly-auto-api-cu117-1.0.0.tar.gz.sha256
```

## 9. 准备部署包

Linux目标服务器最少需要：

```text
model-deploy/
  sampoly-auto-api-cu117-1.0.0.tar
  sampoly-auto-api-cu117-1.0.0.tar.sha256
  docker/
    compose.deploy.yaml
    .env.example
    deploy-image.sh
    verify-remote.sh
  checkpoints/
    auto_whumix.pth
```

权重可能受许可证限制，传输前应确认授权范围。

## 10. 传输到目标服务器

以下示例把构建机作为之后的远程测试客户端：

```bash
DEPLOY_USER=xubo
DEPLOY_HOST=192.168.1.120
DEPLOY_ROOT=/home/xubo/model-deploy
```

创建目标目录：

```bash
ssh "$DEPLOY_USER@$DEPLOY_HOST" \
  "mkdir -p $DEPLOY_ROOT/docker $DEPLOY_ROOT/checkpoints"
```

推荐使用支持进度显示和断点续传的rsync：

```bash
rsync -avP \
  ~/docker-images/sampoly-auto-api-cu117-1.0.0.tar \
  ~/docker-images/sampoly-auto-api-cu117-1.0.0.tar.sha256 \
  "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_ROOT/"

rsync -avP \
  docker/compose.deploy.yaml \
  docker/.env.example \
  docker/deploy-image.sh \
  docker/verify-remote.sh \
  "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_ROOT/docker/"

rsync -avP checkpoints/auto_whumix.pth \
  "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_ROOT/checkpoints/"
```

传输中断后重新执行同一条 `rsync -avP` 命令即可继续。

## 11. 目标服务器环境与完整性校验

目标服务器同样需要Docker、Compose、NVIDIA驱动和NVIDIA Container Toolkit：

```bash
docker --version
docker compose version
nvidia-smi

docker run --rm --gpus all \
  nvidia/cuda:11.7.1-base-ubuntu22.04 \
  nvidia-smi
```

检查部署文件：

```bash
cd /home/xubo/model-deploy
ls -lh
ls -lh docker
ls -lh checkpoints/auto_whumix.pth
```

校验镜像归档：

```bash
sha256sum -c sampoly-auto-api-cu117-1.0.0.tar.sha256
```

只有输出 `OK` 才能继续。若校验文件记录了构建机绝对路径，例如：

```text
/home/xubo/docker-images/sampoly-auto-api-cu117-1.0.0.tar
```

而目标文件在当前目录，可以保留原哈希，仅在校验时去掉路径：

```bash
sed 's#  .*/#  #' \
  sampoly-auto-api-cu117-1.0.0.tar.sha256 \
  | sha256sum -c -
```

如果输出 `FAILED`，应重新传输归档，不要继续 `docker load`。

## 12. 配置目标服务器

```bash
cd /home/xubo/model-deploy
cp docker/.env.example docker/.env
nano docker/.env
```

SAMPoly示例：

```dotenv
SAMPOLY_IMAGE=sampoly-auto-api:cu117-1.0.0
SAMPOLY_BIND_HOST=0.0.0.0
SAMPOLY_API_PORT=8000

SAMPOLY_CHECKPOINT_DIR=../checkpoints
SAMPOLY_CHECKPOINT_FILE=auto_whumix.pth

SAMPOLY_DEVICE=cuda
SAMPOLY_GPU=0
SAMPOLY_BATCH_SIZE=1
SAMPOLY_NUM_WORKERS=1
SAMPOLY_DEFAULT_SCORE_THRESHOLD=0.1
SAMPOLY_MAX_UPLOAD_MB=100
SAMPOLY_SHM_SIZE=4gb

SAMPOLY_API_KEY=replace-with-a-long-random-secret
```

生成随机密钥：

```bash
openssl rand -hex 32
```

不要提交 `docker/.env`，也不要在截图、日志或聊天记录中暴露真实密钥。

## 13. 导入镜像并启动

```bash
cd /home/xubo/model-deploy
chmod +x docker/*.sh

sh docker/deploy-image.sh \
  sampoly-auto-api-cu117-1.0.0.tar \
  docker/.env
```

等价的手动命令：

```bash
docker load --input sampoly-auto-api-cu117-1.0.0.tar

docker compose \
  --env-file docker/.env \
  -f docker/compose.deploy.yaml \
  up -d
```

检查镜像、容器和日志：

```bash
docker image inspect sampoly-auto-api:cu117-1.0.0 \
  --format='Image={{.RepoTags}} Size={{.Size}} ID={{.Id}}'

docker compose \
  --env-file docker/.env \
  -f docker/compose.deploy.yaml \
  ps

docker compose \
  --env-file docker/.env \
  -f docker/compose.deploy.yaml \
  logs --tail=200 sampoly-auto-api
```

## 14. 目标服务器本机验证

```bash
curl -fsS http://127.0.0.1:8000/health
echo
```

必须确认状态为 `ready`，配置、权重和CUDA均可用。然后在目标服务器上执行
一次真实推理，排除网络和防火墙因素后再进行跨机器测试。

## 15. 防火墙和网络配置

建议只允许指定客户端访问推理端口。UFW示例：

```bash
sudo ufw allow from 192.168.1.113 to any port 8000 proto tcp
sudo ufw status
```

其中 `192.168.1.113` 是允许访问的客户端IP。云服务器还需要配置安全组；局域网
之外的服务应置于 HTTPS 反向代理或 API Gateway 后，不应直接暴露未认证的
Uvicorn端口。

## 16. 从另一台机器远程调用

先测试网络和health：

```bash
curl -i http://192.168.1.120:8000/health
```

再使用正确API Key执行真实推理：

```bash
curl -fsS \
  -X POST http://192.168.1.120:8000/v1/auto/infer \
  -H "X-API-Key: REPLACE_WITH_REAL_KEY" \
  -F "image=@/data/building_test.tif" \
  -F "score_threshold=0.1" \
  -o remote-response.json

python -m json.tool remote-response.json
```

还应使用错误密钥发送一次请求，确认接口返回HTTP 401，而 `/health` 仍可用于
基础设施健康检查。

## 17. 日常维护

查看状态和日志：

```bash
docker compose --env-file docker/.env \
  -f docker/compose.deploy.yaml ps

docker compose --env-file docker/.env \
  -f docker/compose.deploy.yaml logs -f sampoly-auto-api
```

重启：

```bash
docker compose --env-file docker/.env \
  -f docker/compose.deploy.yaml restart sampoly-auto-api
```

停止：

```bash
docker compose --env-file docker/.env \
  -f docker/compose.deploy.yaml down
```

不要随意给 `down` 添加 `-v`，否则会删除Compose命名卷。

升级时为新镜像使用新标签，完成导入和验证后修改 `.env` 中的镜像名并重新
执行 `up -d`。保留上一个已验证标签即可快速回滚。

## 18. 常见问题排查

### 18.1 curl不支持 --fail-with-body

原因是curl版本低于7.76.0。无需阻塞部署，可改用：

```bash
curl -fsS http://127.0.0.1:8000/health
```

### 18.2 SHA256提示没有文件或目录

通常是校验文件记录了构建机绝对路径。使用第11节的 `sed` 管道保留原哈希并
移除目录。该错误与真正的哈希 `FAILED` 不同。

### 18.3 CUDA不可用

```bash
nvidia-smi
docker run --rm --gpus all \
  nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi
docker logs CONTAINER_NAME
```

依次确认宿主机驱动、NVIDIA Container Toolkit、Compose GPU声明和容器CUDA
版本。

### 18.4 checkpoint_not_found

检查宿主机路径、文件名和只读挂载：

```bash
ls -lh checkpoints/auto_whumix.pth
docker inspect CONTAINER_NAME
```

相对挂载路径由Compose文件位置决定。启动目录经常变化时应使用绝对路径。

### 18.5 远程连接被拒绝或超时

在目标服务器执行：

```bash
curl -i http://127.0.0.1:8000/health
ss -lntp | grep 8000
sudo ufw status
```

如果本机成功而远程失败，应检查 `SAMPOLY_BIND_HOST`、宿主机防火墙、路由和
云安全组。

### 18.6 HTTP 401

确认 `.env` 中API Key与请求头完全一致：

```text
X-API-Key: REAL_KEY
```

密钥修改后需要重新创建或重启容器。

### 18.7 容器启动后退出

```bash
docker compose --env-file docker/.env \
  -f docker/compose.deploy.yaml ps -a
docker compose --env-file docker/.env \
  -f docker/compose.deploy.yaml logs --tail=300 sampoly-auto-api
```

重点查找Python导入错误、动态库缺失、配置路径、权重路径、显存不足和端口占用。

### 18.8 镜像或tar过大

```bash
docker image history IMAGE_NAME
du -sh ./*
```

确认 `.dockerignore` 排除了归档、数据集和权重；考虑多阶段构建、删除编译期
依赖、使用gzip或zstd压缩传输。优化前必须保留模型运行依赖并重新完成真实推理
验收。

## 19. 最终验收清单

- [ ] 构建机Docker可以访问GPU。
- [ ] 镜像构建成功，依赖检查没有冲突。
- [ ] 构建机health返回 `ready`。
- [ ] 构建机真实模型推理成功，结果坐标合法。
- [ ] 镜像tar和SHA256文件已生成。
- [ ] 目标服务器SHA256校验输出 `OK`。
- [ ] 目标服务器Docker可以访问GPU。
- [ ] `docker load`成功且镜像标签正确。
- [ ] 目标容器处于running/healthy状态。
- [ ] 目标服务器本机health和真实推理成功。
- [ ] 指定客户端能够访问，非授权客户端被防火墙阻止。
- [ ] 正确API Key推理成功，错误API Key返回401。
- [ ] 返回结果通过数量、范围和可视化抽查。
- [ ] 镜像标签、Git提交、权重版本、SHA256和验收结果已归档。

完成以上项目后，才能声明该神经网络镜像已通过Linux跨机器部署验收。
