from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI(title="Brain Tumor Detection API")

# Allow Streamlit to connect to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "../model/brain_tumor_model.h5"
model = load_model(MODEL_PATH)

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        processed = preprocess(img)
        pred = model.predict(processed)[0][0]

        result = "Tumor Detected" if pred < 0.5 else "Healthy Brain"

        return {"result": result, "raw_score": float(pred)}

    except Exception as e:
        return {"error": str(e)}

