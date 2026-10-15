# SAMPoly 多轮交互优化 HTTP 服务部署与联调手册

本服务实现 SHORTER 交互优化协议 `2.0`，与已有 Auto 服务相互独立。影像仅在创建 session 时上传一次，服务端缓存 image embedding；后续 refine 请求只传 JSON，并可通过 `interaction_token` 继续使用上一轮低分辨率 mask logits。

## 1. 接口

```text
POST   /v1/interactive/sessions
POST   /v1/interactive/sessions/{session_id}/refine
DELETE /v1/interactive/sessions/{session_id}
```

三个接口均要求 `X-API-Key`。创建和 refine 还要求非空的 `X-Request-ID`，并且 header 值必须与 JSON 中的 `request_id` 完全一致。

## 2. Linux 环境变量与启动

```bash
cd ~/projects/sampoly/SAMPolyBuild
conda activate sampoly

export SAMPOLY_API_KEY='请替换为足够长的随机密钥'
export SAMPOLY_CONFIG="$PWD/configs/prompt_instance_spacenet.json"
export SAMPOLY_CHECKPOINT="$PWD/prompt_interactive.pth"
export SAMPOLY_DEVICE='cuda'
export SAMPOLY_HOST='0.0.0.0'
export SAMPOLY_PORT='8010'
export SAMPOLY_MAX_UPLOAD_MB='100'
export SAMPOLY_MAX_CONCURRENCY='1'
export SAMPOLY_SESSION_TTL_SECONDS='1800'
export SAMPOLY_MAX_SESSIONS='4'

test -f "$SAMPOLY_CONFIG" && echo 'config OK'
test -f "$SAMPOLY_CHECKPOINT" && echo 'checkpoint OK'
python -m pip install -r docker/requirements-api.txt
python -m interactive_api.main
```

另一个终端检查：

```bash
curl -sS http://127.0.0.1:8010/health
```

预期 `status` 为 `ready`，`protocol_version` 为 `2.0`。

## 3. Docker 启动

```bash
cd ~/projects/sampoly/SAMPolyBuild
mkdir -p checkpoints
cp /实际路径/prompt_interactive.pth checkpoints/prompt_interactive.pth
cp docker/interactive.env.example docker/interactive.env
chmod 600 docker/interactive.env
```

编辑 `docker/interactive.env`，替换 API Key，然后执行：

```bash
docker compose -f docker/compose.interactive.yaml build
docker compose -f docker/compose.interactive.yaml up -d
docker compose -f docker/compose.interactive.yaml ps
docker compose -f docker/compose.interactive.yaml logs --tail=100
```

停止服务：

```bash
docker compose -f docker/compose.interactive.yaml down
```

## 4. curl 三组联调命令

以下命令从真实影像读取文件名和尺寸，并按其尺寸生成合法示例坐标。将路径和密钥替换为实际值。

```bash
export BASE_URL='http://127.0.0.1:8010'
export API_KEY='与SAMPOLY_API_KEY一致'
export TEST_IMAGE='/绝对路径/example.tif'
export FILE_NAME="$(basename "$TEST_IMAGE")"
read WIDTH HEIGHT < <(python3 -c 'from PIL import Image; import os; im=Image.open(os.environ["TEST_IMAGE"]); print(*im.size)')
export WIDTH HEIGHT FILE_NAME
```

### 4.1 创建 session：只上传一次影像

```bash
export CREATE_REQUEST_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
CREATE_JSON="$(python3 -c 'import json,os; print(json.dumps({"protocol_version":"2.0","request_id":os.environ["CREATE_REQUEST_ID"],"coordinate_space":"original_image_pixel","image":{"file_name":os.environ["FILE_NAME"],"width":int(os.environ["WIDTH"]),"height":int(os.environ["HEIGHT"])}}))')"

CREATE_RESPONSE="$(curl -sS -X POST "$BASE_URL/v1/interactive/sessions" \
  -H "Accept: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "X-Request-ID: $CREATE_REQUEST_ID" \
  -F "image=@$TEST_IMAGE" \
  -F "request=$CREATE_JSON;type=application/json")"

echo "$CREATE_RESPONSE" | python3 -m json.tool
export SESSION_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["session_id"])' <<<"$CREATE_RESPONSE")"
```

### 4.2 第一轮 refine

```bash
export ROUND1_REQUEST_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
ROUND1_JSON="$(python3 -c 'import json,os; w=float(os.environ["WIDTH"]); h=float(os.environ["HEIGHT"]); print(json.dumps({"protocol_version":"2.0","request_id":os.environ["ROUND1_REQUEST_ID"],"session_id":os.environ["SESSION_ID"],"polygon_id":"building_0006","revision":1,"coordinate_space":"original_image_pixel","image":{"file_name":os.environ["FILE_NAME"],"width":int(w),"height":int(h)},"bbox_xyxy":[.2*w,.2*h,.8*w,.8*h],"positive_points":[[.5*w,.5*h]],"negative_points":[],"original_polygon":[[.3*w,.3*h],[.7*w,.3*h],[.7*w,.7*h],[.3*w,.7*h]],"interaction_token":""}))')"

ROUND1_RESPONSE="$(curl -sS -X POST "$BASE_URL/v1/interactive/sessions/$SESSION_ID/refine" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "X-Request-ID: $ROUND1_REQUEST_ID" \
  --data "$ROUND1_JSON")"

echo "$ROUND1_RESPONSE" | python3 -m json.tool
export ROUND1_TOKEN="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["interaction_token"])' <<<"$ROUND1_RESPONSE")"
```

### 4.3 第二轮 refine：累计点和上一轮 token

```bash
export ROUND2_REQUEST_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
ROUND2_JSON="$(python3 -c 'import json,os; w=float(os.environ["WIDTH"]); h=float(os.environ["HEIGHT"]); print(json.dumps({"protocol_version":"2.0","request_id":os.environ["ROUND2_REQUEST_ID"],"session_id":os.environ["SESSION_ID"],"polygon_id":"building_0006","revision":2,"coordinate_space":"original_image_pixel","image":{"file_name":os.environ["FILE_NAME"],"width":int(w),"height":int(h)},"bbox_xyxy":[.2*w,.2*h,.8*w,.8*h],"positive_points":[[.5*w,.5*h],[.55*w,.55*h]],"negative_points":[[.75*w,.25*h]],"original_polygon":[[.3*w,.3*h],[.7*w,.3*h],[.7*w,.7*h],[.3*w,.7*h]],"interaction_token":os.environ["ROUND1_TOKEN"]}))')"

ROUND2_RESPONSE="$(curl -sS -X POST "$BASE_URL/v1/interactive/sessions/$SESSION_ID/refine" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "X-Request-ID: $ROUND2_REQUEST_ID" \
  --data "$ROUND2_JSON")"

echo "$ROUND2_RESPONSE" | python3 -m json.tool
```

### 4.4 DELETE 关闭 session

```bash
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' -X DELETE \
  "$BASE_URL/v1/interactive/sessions/$SESSION_ID" \
  -H "X-API-Key: $API_KEY"
```

预期输出 `HTTP 204`；重复 DELETE 仍返回 204。

## 5. 自动化与真实模型验证

普通测试全部使用 mock predictor，不读取 checkpoint、不需要 CUDA：

```bash
conda activate sampoly
cd ~/projects/sampoly/SAMPolyBuild
python -m unittest \
  tests.test_interactive_session_store \
  tests.test_interactive_predictor_adapter \
  tests.test_interactive_api -v
```

服务启动后，可选执行真实 checkpoint 三轮验证：

```bash
python tools/verify_interactive_api.py \
  --base-url http://127.0.0.1:8010 \
  --api-key "$SAMPOLY_API_KEY" \
  --image /绝对路径/example.tif \
  --bbox '[200,200,800,800]' \
  --positive '[[500,500],[550,550]]' \
  --negative '[[750,250]]' \
  --polygon '[[300,300],[700,300],[700,700],[300,700]]'
```

参数坐标必须根据真实影像和目标建筑调整。只有该验证通过，才能说明真实 checkpoint 的多轮交互链路可用。

## 6. SHORTER 配置

```ini
[sampoly_refine]
base_url=http://<Linux-IP>:8010
api_key=<与SAMPOLY_API_KEY一致>
session_endpoint=/v1/interactive/sessions
refine_endpoint_template=/v1/interactive/sessions/{session_id}/refine
connect_timeout_ms=5000
upload_timeout_ms=60000
response_timeout_ms=120000
max_upload_mb=100
max_response_mb=20
require_https=false
proxy_mode=none
proxy_url=
```

## 7. 状态与资源语义

- TTL 是成功访问后刷新的空闲超时，使用单调时钟计算。
- session 过期返回 HTTP 410 / `session_expired`；未知或主动删除的 session 返回 HTTP 404 / `session_not_found`。
- embedding 和 logits 保存在服务端，不编码进 token；session 删除、过期或服务关闭时释放。
- token 为空时，即使 revision 大于 1，也根据完整累计点从头推理。
- 只有输出 polygon、bbox、score 和 logits 全部验证成功后，才原子替换交互状态。
- 默认仅允许一个模型任务进入 GPU；繁忙时返回 HTTP 429 / `service_busy`。
- 生产网络建议由反向代理提供 HTTPS，并只向可信客户端开放 8010 端口。
