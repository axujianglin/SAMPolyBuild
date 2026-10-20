# SAMPoly Linux 开机自启与后台服务手册

本文档用于在不使用 Docker 的 Linux GPU 主机上，将 SAMPoly Auto API 和交互优化 API 注册为 systemd 服务。完成一次安装后，Linux 会在网络就绪后自动启动两个服务，SHORTER 无需等待用户登录服务器。

## 1. 服务布局

| 服务 | 地址 | 功能 |
|---|---|---|
| `sampoly-auto.service` | `0.0.0.0:8000` | 自动提取初始建筑轮廓 |
| `sampoly-interactive.service` | `0.0.0.0:8010` | 正负点多轮交互优化 |
| `sampoly.target` | 无 | 统一启动、停止两个服务 |

服务以指定普通用户运行，配置由 systemd 从 `/etc/sampoly/sampoly.env` 读取，日志保存在 systemd journal 中。

## 2. 安装前检查

```bash
cd ~/projects/sampoly/SAMPolyBuild
conda activate sampoly
git branch --show-current
nvidia-smi
```

确认以下文件存在，并根据服务器上的真实位置调整路径：

```bash
test -x /home/xubo/anaconda3/envs/sampoly/bin/python
test -f configs/auto_whumix.py
test -f configs/prompt_instance_spacenet.json
test -f checkpoints/auto_whumix.pth
test -f prompt_interactive.pth
```

安装主机需要使用 systemd，可执行：

```bash
systemctl --version
```

## 3. 创建配置

不要直接编辑示例文件。复制到仓库外的临时安全位置：

```bash
cp deploy/systemd/sampoly.env.example ~/sampoly.env
chmod 600 ~/sampoly.env
```

生成 API Key：

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

编辑配置：

```bash
nano ~/sampoly.env
```

必须修改：

- `SAMPOLY_PROJECT_ROOT`
- `SAMPOLY_PYTHON`
- `SAMPOLY_API_KEY`
- 两个模型配置和 checkpoint 的绝对路径
- 根据实际需求修改监听端口、GPU 和并发参数

配置文件使用 systemd `EnvironmentFile` 格式。路径必须使用绝对路径，不要使用 `$HOME`、`$PWD`、`~` 或命令替换。

Auto 服务的设备值只能是 `cuda` 或 `cpu`；Refine 服务可以使用 `cuda:0`。

## 4. 安装并启动

先运行静态检查：

```bash
bash scripts/validate-systemd-assets.sh
```

执行安装：

```bash
sudo bash scripts/install-systemd-services.sh \
  --user xubo \
  --config /home/xubo/sampoly.env
```

安装脚本会：

1. 检查必要文件和配置键。
2. 将配置复制到 `/etc/sampoly/sampoly.env`，权限设为 `0600`。
3. 安装普通用户运行的 Auto 和 Refine 服务。
4. 设置 `sampoly.target` 随 Linux 开机启动。
5. 立即启动两个服务。

安装脚本不会修改防火墙、网卡或固定 IP。

## 5. 状态与健康检查

```bash
sudo systemctl status sampoly.target --no-pager
sudo systemctl status sampoly-auto.service --no-pager
sudo systemctl status sampoly-interactive.service --no-pager
```

默认端口下可以执行：

```bash
bash scripts/check-systemd-services.sh
```

自定义端口时传入两个基础地址：

```bash
bash scripts/check-systemd-services.sh \
  http://127.0.0.1:18000 \
  http://127.0.0.1:18010
```

也可分别查看：

```bash
curl -sS http://127.0.0.1:8000/health | python -m json.tool
curl -sS http://127.0.0.1:8010/health | python -m json.tool
```

## 6. 常用管理命令

统一停止和启动：

```bash
sudo systemctl stop sampoly.target
sudo systemctl start sampoly.target
```

修改 `/etc/sampoly/sampoly.env` 后重启：

```bash
sudo systemctl restart sampoly-auto.service
sudo systemctl restart sampoly-interactive.service
```

查看最近日志：

```bash
sudo journalctl -u sampoly-auto.service -n 100 --no-pager
sudo journalctl -u sampoly-interactive.service -n 100 --no-pager
```

实时查看日志：

```bash
sudo journalctl -u sampoly-interactive.service -f
```

## 7. Windows SHORTER 连通性

查看 Linux IP：

```bash
hostname -I
```

在 Windows PowerShell 中执行：

```powershell
Test-NetConnection <Linux-IP> -Port 8000
Test-NetConnection <Linux-IP> -Port 8010
curl.exe http://<Linux-IP>:8000/health
curl.exe http://<Linux-IP>:8010/health
```

如果 Linux 本机可以访问但 Windows 不可以，只允许 SHORTER 所在主机访问端口：

```bash
sudo ufw allow from <Windows-IP> to any port 8000 proto tcp
sudo ufw allow from <Windows-IP> to any port 8010 proto tcp
```

若 Linux 使用 DHCP，应在路由器配置地址保留或为 Linux 配置固定 IP。systemd 能等待网络并启动服务，但不能保证 DHCP 地址不变化。

## 8. 重启验收

```bash
sudo reboot
```

服务器恢复后执行：

```bash
sudo systemctl is-enabled sampoly.target
sudo systemctl is-active sampoly-auto.service
sudo systemctl is-active sampoly-interactive.service
bash scripts/check-systemd-services.sh
```

预期：

```text
enabled
active
active
All SAMPoly services are active and ready.
```

随后在 Windows 再运行端口和 `/health` 检查，并启动 SHORTER 完成一次 Auto 与 Refine 请求。

## 9. 故障排查

服务启动失败时先查看：

```bash
sudo journalctl -u sampoly-auto.service -b --no-pager
sudo journalctl -u sampoly-interactive.service -b --no-pager
```

常见原因：

| 日志 | 原因 |
|---|---|
| `required configuration ... is empty` | 配置键缺失或值为空 |
| `file not found` | 配置或权重路径错误 |
| `configured Python is not executable` | Conda Python 路径错误 |
| `address already in use` | 端口被旧进程占用 |
| `CUDA device requested but CUDA is unavailable` | systemd 环境下 CUDA 不可用 |
| `CUDA out of memory` | 两个模型或其他程序占用显存过多 |
| Windows 无法连接 | 防火墙、IP、监听地址或网络隔离问题 |

配置修正后执行：

```bash
sudo systemctl restart sampoly.target
```

## 10. 卸载

保留 `/etc/sampoly/sampoly.env`：

```bash
sudo bash scripts/uninstall-systemd-services.sh
```

同时删除配置：

```bash
sudo bash scripts/uninstall-systemd-services.sh --purge-config
```

卸载不会删除源码、模型权重或运行结果。
