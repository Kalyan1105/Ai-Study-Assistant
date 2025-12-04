from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
app = FastAPI()

model = load_model("tb_model.h5")
IMG_SIZE = (100, 100)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-xray")
async def predict_xray(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L")  # grayscale
    img = img.resize(IMG_SIZE)

    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob_tb = float(model.predict(arr)[0][0])
    label = "Tuberculosis" if prob_tb >= 0.5 else "Normal"

    return {
        "prediction": label,
        "probability_tb": prob_tb,
        "probability_normal": 1.0 - prob_tb,
    }
