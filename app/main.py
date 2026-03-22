from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from app.model import predict_image
from app.schemas import PredictionResponse


app = FastAPI(title="CIFAR10 Image Classifier API")


@app.get("/")
def root():
    return {"message": "CIFAR10 Image Classifier API is running."}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = predict_image(image)
    return result


