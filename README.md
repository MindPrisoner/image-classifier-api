# 🧠 CIFAR10 Image Classifier API

A production-style image classification API built with **FastAPI + PyTorch + Docker**.

---

## 🚀 Features

- Upload image for classification
- Top-1 prediction + confidence
- Top-3 probability output
- Batch prediction support
- Dockerized deployment
- RESTful API interface

---

## 🧱 Tech Stack

- PyTorch
- FastAPI
- Uvicorn
- Docker

---

## 📦 Project Structure


app/
models/
checkpoints/
Dockerfile


---

## ⚙️ Run Locally

```bash
uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs
🐳 Run with Docker
Build image
docker build -t image-classifier-api .
Run container
docker run -p 8000:8000 image-classifier-api
📡 API Endpoints
Health Check
GET /health
Single Prediction
POST /predict
Batch Prediction
POST /predict_batch
📊 Example Response
{
  "top1": "cat",
  "confidence": 0.97,
  "top3": {
    "cat": 0.97,
    "dog": 0.02,
    "deer": 0.01
  }
}
🧠 Model
ResNet18
Trained on CIFAR10

---

