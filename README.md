# Image Enhancement Web Application

A FastAPI-based web application for enhancing images using a pre-trained deep learning model. Users can upload an image via a web UI and receive an enhanced version generated using a TensorFlow model.

---

## Project Structure

```
Colorizer/
│
├── app.py                 # FastAPI application entry point
├── model.h5               # Pre-trained image enhancement model
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates (UI pages)
├── static/                # Static files and output images
└── data/                  # Optional data directory
```

---

## Tech Stack

* **Python 3.10**
* **FastAPI**
* **TensorFlow / Keras**
* **Pillow (PIL)**
* **Jinja2 Templates**
* **Uvicorn**

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Pratham1008/IMAGE_ENHANCEMENT_HONOURS.git
cd IMAGE_ENHANCEMENT_HONOURS
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Model

Download `model.h5` from the link below and place it in the project root directory:

```
https://drive.google.com/file/d/1tEWI8hTpHkL41VqLqTMGBqsGQMh1oOQH/view
```

---

## Running the Application

```bash
python app.py
```

The server will start at:

```
http://127.0.0.1:8000
```

---

## Application Routes

| Endpoint   | Method | Description                          |
| ---------- | ------ | ------------------------------------ |
| `/`        | GET    | Home page with image upload UI       |
| `/aboutus` | GET    | About page                           |
| `/enhance` | POST   | Upload image and get enhanced output |

---

## How It Works

1. User uploads an image via the web interface.
2. Image is normalized and passed to the trained TensorFlow model.
3. Model enhances the image using learned features.
4. Enhanced image is saved and returned to the frontend.

Custom loss and evaluation metrics:

* **Charbonnier Loss**
* **Peak Signal-to-Noise Ratio (PSNR)**

---

## Output

* Enhanced image is saved in the `static/` directory.
* Result is displayed instantly on the UI.

---

## Notes

* Ensure `model.h5` exists before starting the app.
* The application currently supports **RGB images only**.
* Internet access is required for initial dependency installation.
