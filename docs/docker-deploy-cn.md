# Docker 部署说明

这个项目可以不用安装 `insightface` Python 包，直接使用 ONNX 引擎在 Docker 里跑 CPU 推理。

## CPU 推理速度

2000 人规模下，向量匹配本身很快。主要耗时在：

- 人脸检测
- ArcFace 特征提取
- 图片解码和缩放

普通 CPU 上，单张图片通常可以做到秒级以内；如果图片很大或一张图里多人脸，耗时会增加。2000 到 10000 人这种规模，当前 NumPy 余弦相似度匹配仍然很轻。

## 构建镜像

```bash
docker compose build
```

## 启动服务

```bash
docker compose up -d
```

首次启动会自动下载两个 ONNX 模型：

- InsightFace ArcFace `w600k_r50.onnx`
- OpenCV YuNet 人脸检测模型

启动后访问：

- Web 页面：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs

## 训练示例数据集

容器启动后，页面里填写训练路径：

```text
datasets/faces
```

也可以用 API：

```bash
curl -X POST "http://127.0.0.1:8000/api/train/path" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path":"datasets/faces"}'
```

训练结果会保存到宿主机：

```text
data/models/
```

## Linux 和 macOS

Linux 和 macOS 都可以使用同一套命令：

```bash
docker compose build
docker compose up -d
```

macOS 上 Docker Desktop 默认也是 CPU 推理。如果是 Apple Silicon，`python:3.10-slim` 会自动使用 arm64 镜像，`onnxruntime` 也会安装对应平台包。

## 常用命令

查看日志：

```bash
docker compose logs -f
```

停止服务：

```bash
docker compose down
```

重新训练后查看状态：

```bash
curl http://127.0.0.1:8000/api/health
```
