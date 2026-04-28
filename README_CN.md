# InsightFace 人脸识别训练项目中文说明

本项目是一个基于 InsightFace 的人脸识别系统，提供 Web 页面和 REST API，可以用于构建 2000+ 人员规模的人脸库。

## 功能概览

- 支持按人员目录训练人脸库
- 支持上传 zip 数据集训练
- 支持上传图片进行人脸识别
- 支持查看人员库、特征数量和服务状态
- 使用 InsightFace 提取人脸特征，并保存本地向量库

## 重要说明

这里的“训练”不是从零训练一个深度学习模型，而是使用 InsightFace 已训练好的 ArcFace 模型提取人脸特征，然后建立本地人脸特征库。

这种方式更适合实际业务：

- 训练速度快
- 2000+ 人员也能轻松扩展
- 后续新增人员不需要重新训练深度模型
- 更适合 Web/API 服务部署

## 项目结构

```text
app/
  main.py              FastAPI 后端入口
  config.py            项目配置
  schemas.py           API 返回结构
  services/
    face_engine.py     InsightFace 模型封装
    face_store.py      人脸特征库存储和检索
    trainer.py         数据集训练逻辑
  web/
    index.html         Web 页面
    styles.css         页面样式
    app.js             页面交互逻辑

datasets/
  faces/               训练图片目录
  README.md            数据集说明
  sample_celebrities_manifest.json

scripts/
  download_sample_dataset.py

data/
  models/              训练后生成的人脸库
  uploads/             上传文件临时目录
```

## 环境准备

推荐使用 Python 3.10。

```powershell
conda create -y -p .\.conda python=3.10
.\.conda\python.exe -m pip install -r requirements.txt
```

如果你只是想先启动 Web 页面和 API，不安装 InsightFace，也可以执行：

```powershell
.\.conda\python.exe -m pip install -r requirements-core.txt
```

不过这种情况下只能打开页面，训练和识别接口会提示 InsightFace 未安装。

## Windows 安装 InsightFace 的问题

如果安装依赖时出现：

```text
Microsoft Visual C++ 14.0 or greater is required
```

说明本机缺少 C++ 编译环境。请先安装 Microsoft C++ Build Tools，然后重新执行：

```powershell
.\.conda\python.exe -m pip install insightface==0.7.3
```

安装完成后再启动服务即可训练和识别。

## 不安装 InsightFace 包的替代方案

如果不想安装 Microsoft C++ Build Tools，可以使用项目内置的 ONNX 备用引擎。它使用 InsightFace ArcFace ONNX 模型提取人脸特征，不需要安装 `insightface` Python 包。

先下载 ONNX 模型：

```powershell
.\.conda\python.exe scripts\download_onnx_models.py
.\.conda\python.exe scripts\download_detector_model.py
```

然后启动服务：

```powershell
$env:FACE_ENGINE="onnx"
.\.conda\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

也可以不设置 `FACE_ENGINE`，系统会优先尝试 `insightface` 包；如果没安装，会自动切换到 ONNX 备用引擎。

注意：ONNX 备用引擎用 OpenCV 做基础人脸检测，没有完整 InsightFace 检测和五点对齐流程，适合开发、演示和小规模测试。正式生产建议安装完整 InsightFace 包。

## 启动服务

```powershell
.\.conda\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动后访问：

- Web 页面：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/api/health

## Docker 部署

如果要在 Linux 或 macOS 部署，推荐直接用 Docker：

```bash
docker compose build
docker compose up -d
```

详细说明见：

```text
docs/docker-deploy-cn.md
```

## 数据集格式

训练集必须按“一个人一个目录”的方式组织：

```text
datasets/faces/
  jackie_chan/
    001.jpg
    002.jpg
  jet_li/
    001.jpg
    002.jpg
```

目录名就是识别出来的人员名称。

建议：

- 每个人至少 5 张图片
- 尽量使用清晰人脸
- 正脸、半侧脸都可以放
- 避免多人合照作为训练图片
- 图片格式支持 jpg、jpeg、png、bmp、webp

## 示例明星数据集

当前项目已经生成了 10 个明星示例目录：

```text
datasets/faces/
  andy_lau/
  fan_bingbing/
  gong_li/
  jackie_chan/
  jay_chou/
  jet_li/
  leonardo_dicaprio/
  taylor_swift/
  tony_leung/
  zhang_ziyi/
```

重新生成示例数据集：

```powershell
.\.conda\python.exe scripts\download_sample_dataset.py
```

图片来源清单：

```text
datasets/sample_celebrities_manifest.json
```

## 使用 Web 页面训练

1. 打开 http://127.0.0.1:8000
2. 在“目录训练”里填写：

```text
datasets/faces
```

3. 点击“开始训练”
4. 等待结果区显示训练完成
5. 页面会刷新人员数和特征数

## 使用 API 训练

### 通过本地目录训练

```powershell
curl -X POST "http://127.0.0.1:8000/api/train/path" ^
  -H "Content-Type: application/json" ^
  -d "{\"dataset_path\":\"datasets/faces\"}"
```

返回示例：

```json
{
  "persons": 10,
  "images_seen": 10,
  "faces_indexed": 10,
  "skipped_images": 0,
  "store_version": "1.0",
  "message": "训练完成，特征库已更新"
}
```

### 通过 zip 上传训练

zip 内部也需要保持一个人一个目录的结构。

```powershell
curl -X POST "http://127.0.0.1:8000/api/train/upload" ^
  -F "file=@faces.zip"
```

## 使用 API 识别图片

```powershell
curl -X POST "http://127.0.0.1:8000/api/recognize" ^
  -F "file=@test.jpg"
```

返回示例：

```json
{
  "faces_detected": 1,
  "matches": [
    {
      "person_id": 0,
      "name": "jackie_chan",
      "score": 0.62,
      "threshold": 0.38,
      "accepted": true
    }
  ]
}
```

字段说明：

- `faces_detected`：检测到的人脸数量
- `name`：匹配到的人员目录名
- `score`：相似度分数，越高越像
- `threshold`：识别阈值
- `accepted`：是否超过阈值并被系统接受

## 查看人员库

```powershell
curl "http://127.0.0.1:8000/api/people"
```

## 查看服务状态

```powershell
curl "http://127.0.0.1:8000/api/health"
```

## 配置项

可以通过环境变量调整：

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `FACE_DATA_DIR` | `data` | 数据保存目录 |
| `INSIGHTFACE_MODEL` | `buffalo_l` | InsightFace 模型名称 |
| `FACE_DET_SIZE` | `640,640` | 人脸检测尺寸 |
| `FACE_THRESHOLD` | `0.38` | 识别阈值 |
| `FACE_CTX_ID` | `-1` | 推理设备，CPU 为 `-1`，GPU 0 为 `0` |

## 2000+ 人员训练建议

对于 2000 多个人脸身份，建议这样准备数据：

- 每个人 5 到 20 张图片
- 图片尽量清晰，脸部不要太小
- 每张训练图最好只有一个主要人脸
- 人员目录名保持稳定，不要频繁改名
- 训练完成后备份 `data/models/`

训练完成后主要文件：

```text
data/models/persons.json
data/models/embeddings.npz
```

这两个文件就是本地人脸库。

## 常见问题

### 1. 页面能打开，但训练失败

通常是 InsightFace 没有安装成功。请确认：

```powershell
.\.conda\python.exe -m pip show insightface
```

如果没有结果，需要安装：

```powershell
.\.conda\python.exe -m pip install insightface==0.7.3
```

### 2. 训练时 skipped_images 很多

说明很多图片没有检测到人脸。可以换更清晰、更正面、脸部更大的图片。

### 3. 识别结果不准

可以尝试：

- 每个人增加更多训练图
- 删除模糊、遮挡严重的图片
- 调整 `FACE_THRESHOLD`
- 使用更高质量的注册照

### 4. 想用 GPU

设置：

```powershell
$env:FACE_CTX_ID="0"
```

然后重新启动服务。GPU 环境还需要匹配合适的 onnxruntime-gpu、CUDA 和驱动版本。
