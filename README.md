# 💧 WaterEye: Intelligent Water Meter Reading System

WaterEye is an end-to-end computer vision solution designed to automate the process of reading water meters (both Analog and Digital). By leveraging deep learning, the system converts meter images into accurate digital readings, facilitating seamless data management and billing via a modern web interface.

## 👥 Team Roles & Responsibilities

| Name | Role & Core Responsibilities |
| --- | --- |
| **Ehab** | • **Project Architect:** Defined the business logic and problem statement.<br>

<br>• **Mathematics:** Developed the geometric formulas to calculate real-world readings from needle angles.<br>

<br>• **Deployment:** Created the model inference pipeline for high-scale processing.<br>

<br>• **Backend:** Developed the **FastAPI Framework** and integrated it with the website (Collaborated with Ali). |
| **Ali** | • **Full-Stack Development:** Built the Web Front-end and managed the Database.<br>

<br>• **Backend:** Developed the server-side logic (Collaborated with Ehab).<br>

<br>• **Data Engineering:** Managed the 70/15/15 data split (Train/Validation/Test) to ensure robust model evaluation. |
| **Atrees** | • **Preprocessing:** Handled data cleaning and preparation pipelines in collaboration with Ehab.<br>

<br>• **Web Dev:** Assisted in building the web interface components alongside Ali. |
| **Nasser, Amira, & Maisam** | • **AI Training:** Focused on the core Model Training phase.<br>

<br>• **Evaluation:** Performed rigorous Model Testing and Validation to ensure accuracy across different meter types. |

---

## 🛠️ Technical Overview

### 1. Data Pipeline

* **Split Strategy:** The dataset was manually organized into a **70% Train, 15% Validation, and 15% Test** structure to prevent overfitting and ensure reliable performance.
* **Augmentation:** Used `Albumentations` to increase model robustness against different lighting and angles.

### 2. Machine Learning Models

* **Object Detection (YOLOv8):** Used to detect critical meter components: the center, the needle tip, and scale markers (Min/Max).
* **Router Model:** An ensemble of `EfficientNet_B0`, `ResNet50`, and `MobileNetV3` is used to classify the meter type before processing.
* **OCR:** Integrated for digital meter reading extraction.

### 3. Mathematical Inference (Analog Meters)

The system calculates the reading by determining the angle of the needle relative to the zero-point of the scale:

```python
# Core logic used in the project
def calculate_angle(center, point):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    deg = math.degrees(math.atan2(dy, dx))
    return deg + 360 if deg < 0 else deg

```

### 4. Backend & API

* **Framework:** FastAPI for high-performance, asynchronous requests.
* **Endpoint:** `POST /predict/`
* **Features:** Supports multiple image formats (`.jpg`, `.png`, `.bmp`) and maps readings to specific `station_id` identifiers.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* GPU (Recommended for training) or CPU (for inference)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/watereye.git

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Run the Production API:
```bash
uvicorn PRODUCTION_READY_FASTAPI:app --reload

```



## 📁 File Structure

* `PRODUCTION_READY_FASTAPI.py`: The main entry point for the web-integrated backend.
* `Final-clean-notebook-WaterEye.ipynb`: The primary pipeline for data organization and YOLO training.
* `Untitled10 (1).ipynb`: contains the Meter Classification (Router) and Digital OCR logic.
