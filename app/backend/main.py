import io
import json
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

#Paths & constants
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "quickdraw_scratch_final_deploy.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_list_z.json"

IMG_SIZE = 224

DEBUG_DIR = BASE_DIR / "debug_inputs"
DEBUG_DIR.mkdir(exist_ok=True)

#App & model
app = FastAPI(title="QuickDraw Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# Helper functions
def crop_to_bbox(img: np.ndarray) -> np.ndarray:
    ys, xs = np.where(img < 0.95)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return img[y_min:y_max + 1, x_min:x_max + 1]

def pad_to_square(img: np.ndarray, fill: float = 1.0) -> np.ndarray:
    h, w = img.shape
    size = max(h, w)
    padded = np.full((size, size), fill, dtype=img.dtype)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = img
    return padded

# Preprocessing (MATCH TRAINING)
def preprocess_image(file_bytes: bytes, save_debug: bool = True) -> np.ndarray:
    #Load to grayscale
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = np.array(img).astype("float32") / 255.0

    #Auto-invert if background is dark
    if img.mean() < 0.5:
        img = 1.0 - img

    #Crop to strokes
    img = crop_to_bbox(img)

    #Pad to square
    img = pad_to_square(img, fill=1.0)

    #Resize
    img = Image.fromarray((img * 255).astype("uint8"))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img = np.array(img).astype("float32") / 255.0

    #Expand Channel
    img = np.expand_dims(img, axis=-1)

    #Save debug image
    if save_debug:
        dbg = (img.squeeze() * 255).astype("uint8")
        Image.fromarray(dbg).save(
            DEBUG_DIR / f"preprocessed_{uuid.uuid4().hex}.png"
        )

    #Add batch dim
    img = np.expand_dims(img, axis=0)

    return img

#API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    img = preprocess_image(image_bytes, save_debug=True)

    preds = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))

    top3_idx = np.argsort(preds)[-3:][::-1]

    return {
        "predicted_class": class_names[idx],
        "confidence": float(preds[idx]),
        "top3": [
            {
                "class": class_names[i],
                "confidence": float(preds[i]),
            }
            for i in top3_idx
        ],
    }
