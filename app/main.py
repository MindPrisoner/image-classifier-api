from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import io

from app.model import predict_image
from app.schemas import PredictionResponse
from app.schemas import BatchPredictionResponse

app = FastAPI(title="CIFAR10 Image Classifier API")


@app.get("/")
def root():
    return {"message": "CIFAR10 Image Classifier API is running."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process image.")

    result = predict_image(image)
    result["filename"] = file.filename
    return result



from typing import List


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process image: {file.filename}, error: {str(e)}")

        result = predict_image(image)
        result["filename"] = file.filename
        results.append(result)

    return {"results": results}