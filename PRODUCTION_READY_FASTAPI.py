# import os
# import cv2
# import math
# import numpy as np
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from ultralytics import YOLO
# import uvicorn

# # --- CONFIGURATION & MODEL LOADING ---
# MODEL_PATH = 'best.pt'
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# app = FastAPI(title="WaterEye API", description="Analog Meter Reading Service")

# # Load YOLO model once during startup
# if os.path.exists(MODEL_PATH):
#     model_yolo = YOLO(MODEL_PATH)
#     print(f"✅ Analog Model Loaded: {MODEL_PATH}")
# else:
#     print(f"⚠️ {MODEL_PATH} not found. Using generic yolov8n.pt")
#     model_yolo = YOLO('yolov8n.pt')

# # --- MATH LOGIC ---

# def calculate_angle(center, point):
#     dx = point[0] - center[0]
#     dy = point[1] - center[1]
#     deg = math.degrees(math.atan2(dy, dx))
#     return deg + 360 if deg < 0 else deg

# def get_analog_reading(boxes, names, max_scale_value):
#     center = tip = min_p = max_p = None
    
#     for box in boxes:
#         lbl = names[int(box[5])]
#         pt = ((box[0]+box[2])/2, (box[1]+box[3])/2)
#         if lbl == 'center': center = pt
#         elif lbl == 'tip': tip = pt
#         elif lbl == 'min': min_p = pt
#         elif lbl == 'max': max_p = pt

#     if not center or not tip:
#         return None, "Error (Missing Needle or Center)"

#     needle_angle = calculate_angle(center, tip)
#     min_angle = calculate_angle(center, min_p) if min_p else 135
#     max_angle = calculate_angle(center, max_p) if max_p else 45

#     def norm(a, s): return (a - s + 360) % 360
    
#     full_span = norm(max_angle, min_angle)
#     current_span = norm(needle_angle, min_angle)

#     if full_span < 10: full_span = 270

#     fraction = current_span / full_span
#     fraction = max(0.0, min(1.1, fraction)) # Allow slight overflow
    
#     final_value = round(fraction * max_scale_value, 2)
#     return final_value, "Success"

# # --- API ENDPOINT ---

# @app.post("/analyze")
# async def analyze_meter(
#     station_id: int = Form(...), 
#     max_scale: float = Form(...), 
#     file: UploadFile = File(...)
# ):
#     """
#     Accepts Station ID, Max Scale, and an Image File.
#     Returns the calculated reading.
#     """
#     # 1. Read the uploaded image
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     if img is None:
#         raise HTTPException(status_code=400, detail="Invalid image file")

#     # 2. YOLO Inference
#     results = model_yolo.predict(img, conf=0.15, verbose=False)[0]
#     names = results.names
#     data = results.boxes.data.cpu().numpy()
    
#     # 3. Calculate Reading
#     val, status = get_analog_reading(data, names, max_scale)

#     if val is None:
#         return {
#             "station_id": station_id,
#             "status": "Detection Failed",
#             "error": status,
#             "reading": 0
#         }

#     return {
#         "station_id": station_id,
#         "max_scale": max_scale,
#         "reading": val,
#         "status": status
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



import os
import cv2
import math
import numpy as np
from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
import uvicorn

# --- CONFIGURATION & MODEL LOADING ---
MODEL_PATH = 'best.pt'
IMAGES_DIR = "images"  # المجلد اللي فيه صور العدادات
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI(title="WaterEye API", description="Analog Meter Reading Service via JSON & Local Images")

# Load YOLO model once during startup
if os.path.exists(MODEL_PATH):
    model_yolo = YOLO(MODEL_PATH)
    print(f"✅ Analog Model Loaded: {MODEL_PATH}")
else:
    print(f"⚠️ {MODEL_PATH} not found. Using generic yolov8n.pt")
    model_yolo = YOLO('yolov8n.pt')

# تأكد من وجود فولدر الصور
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# --- MATH LOGIC ---
def calculate_angle(center, point):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    deg = math.degrees(math.atan2(dy, dx))
    return deg + 360 if deg < 0 else deg

def get_analog_reading(boxes, names, max_scale_value):
    center = tip = min_p = max_p = None
    
    for box in boxes:
        lbl = names[int(box[5])]
        pt = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        if lbl == 'center': center = pt
        elif lbl == 'tip': tip = pt
        elif lbl == 'min': min_p = pt
        elif lbl == 'max': max_p = pt

    if not center or not tip:
        return None, "Error (Missing Needle or Center)"

    needle_angle = calculate_angle(center, tip)
    # زوايا افتراضية لو الموديل ملقاش الـ min/max
    min_angle = calculate_angle(center, min_p) if min_p else 135
    max_angle = calculate_angle(center, max_p) if max_p else 45

    def norm(a, s): return (a - s + 360) % 360
    
    full_span = norm(max_angle, min_angle)
    current_span = norm(needle_angle, min_angle)

    if full_span < 10: full_span = 270

    fraction = current_span / full_span
    fraction = max(0.0, min(1.1, fraction)) # Allow slight overflow
    
    # 2. هنا بنستخدم الـ max_scale اللي جاي من الـ JSON في الحسابات
    final_value = round(fraction * max_scale_value, 2)
    return final_value, "Success"

# --- API ENDPOINT ---

@app.post("/analyze")
async def analyze_meter(request: MeterRequest):
    """
    Accepts JSON: {"station_id": "123", "max_scale": 10.0}
    Loads image from local folder based on station_id.
    """
    
    # 3. Mapping: البحث عن الصورة بناء على الـ station_id
    # بنجرب اكتر من صيغة عشان لو الصورة jpg او png
    image_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        temp_path = os.path.join(IMAGES_DIR, f"{request.station_id}{ext}")
        if os.path.exists(temp_path):
            image_path = temp_path
            break
    
    if not image_path:
        raise HTTPException(status_code=404, detail=f"Image for station ID '{request.station_id}' not found in '{IMAGES_DIR}' folder.")

    # قراءة الصورة من المسار المحلي
    img = cv2.imread(image_path)

    if img is None:
        raise HTTPException(status_code=500, detail="Could not read the image file. It might be corrupted.")

    # YOLO Inference
    results = model_yolo.predict(img, conf=0.15, verbose=False)[0]
    names = results.names
    data = results.boxes.data.cpu().numpy()
    
    # Calculate Reading passing the max_scale from JSON
    val, status = get_analog_reading(data, names, request.max_scale)

    if val is None:
        return {
            "station_id": request.station_id,
            "status": "Detection Failed",
            "error": status,
            "reading": 0
        }

    return {
        "station_id": request.station_id,
        "max_scale": request.max_scale,
        "reading": val,
        "status": status,
        "image_source": image_path # عشان التأكيد اننا قرأنا الملف الصح
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)