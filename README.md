# CIFAR10 Image Classifier API

这是一个面向推理部署场景的 CIFAR10 图像分类 API 项目，基于 FastAPI、PyTorch 和 Docker 构建，提供单张图片和批量图片的分类能力。

## 项目定位

- 将训练好的 CIFAR10 分类模型封装为 HTTP API
- 提供健康检查、单图预测和批量预测接口
- 支持 Docker 化部署
- 输出 Top-1 结果、置信度和 Top-3 概率分布

## 技术栈

- PyTorch
- FastAPI
- Uvicorn
- Docker

## 目录结构

```text
image_classifier_api/
├── app/
│   ├── main.py           # API 入口
│   ├── model.py          # 模型加载与预测
│   ├── schemas.py        # 响应结构定义
│   └── utils.py          # 图像预处理
├── models/
│   └── resnet.py         # ResNet18 结构
├── checkpoints/          # 模型权重目录
├── Dockerfile            # 容器部署文件
├── requirements.txt      # 依赖列表
└── README.md
```

## 接口说明

### 健康检查

- `GET /health`

### 单图预测

- `POST /predict`

### 批量预测

- `POST /predict_batch`

## 运行方式

本地启动方式：

```bash
uvicorn app.main:app --reload
```

启动后访问：

```text
http://127.0.0.1:8000/docs
```

Docker 启动方式：

```bash
docker build -t image-classifier-api .
docker run -p 8000:8000 image-classifier-api
```

## 模型说明

项目使用的是 ResNet18，训练数据为 CIFAR10。

API 服务在启动时会加载 `checkpoints/cifar10_resnet18.pth`，并在请求到达时完成图像预处理、前向推理和结果封装。

## 响应格式

单图预测结果包含：

- `top1`
- `confidence`
- `top3`
- `filename`

批量预测结果以 `results` 数组形式返回，每个元素对应一张图片的预测信息。

## 备注

- `examples/` 目录保存了测试图片
- `Dockerfile` 用于容器化部署
- 权重文件缺失时，接口无法正常启动

