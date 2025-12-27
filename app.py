import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
app = FastAPI()

os.makedirs('static', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def charbonnier_loss(y_true, y_pred, epsilon=1e-3):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + epsilon))


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255)


custom_objects = {'charbonnier_loss': charbonnier_loss, 'peak_signal_noise_ratio': peak_signal_noise_ratio}
model = tf.keras.models.load_model('model.h5', custom_objects=custom_objects)


def process_enhancement(image_path: str):
    image = Image.open(image_path).convert('RGB')
    image_array = keras.utils.img_to_array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    output = model.predict(image_array)
    output_image = (output[0] * 255.0).clip(0, 255).astype(np.uint8)

    enhanced_image = Image.fromarray(output_image)
    output_path = os.path.join('static', 'enhanced_result.png')
    enhanced_image.save(output_path)
    return "/static/enhanced_result.png"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/aboutus", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("aboutUs.html", {"request": request})


@app.post("/enhance")
async def enhance_route(file: UploadFile = File(...)):
    temp_path = os.path.join('static', 'original_preview.png')
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    enhanced_url = process_enhancement(temp_path)
    return JSONResponse({"enhanced_url": enhanced_url})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)