# ------------------------------------------------------------
# FastAPI + Ultralytics YOLOv10-n “Phone Detection” backend
# ------------------------------------------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from pathlib import Path
import cv2, numpy as np, base64, io
from PIL import Image

app = FastAPI()

# --- CORS for browser front-end ------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load weights sitting next to this script ---------------------------------
WEIGHTS_PATH = Path(__file__).with_name("yolov10n.pt")  # backend/yolov10n.pt

try:
    model = YOLO(str(WEIGHTS_PATH))
    print(f"✅ Model loaded: {WEIGHTS_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # /detect will 500 until weights load

# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    """Health-check."""
    return {"message": "Mobile Phone Detection API",
            "model_loaded": model is not None}

@app.post("/detect")
async def detect_phones(file: UploadFile = File(...),
                        confidence: float = 0.5):
    """Return phone bounding boxes + annotated JPEG (base64)."""
    if model is None:
        raise HTTPException(500, "Model not loaded")

    try:
        # --- read frame ----------
        img = Image.open(io.BytesIO(await file.read()))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # --- inference -----------
        results = model(frame, conf=confidence)
        phones, annotated = [], frame.copy()

        for r in results:
            for box in (r.boxes or []):
                if int(box.cls[0]) == 67:             # COCO “cell phone”
                    conf, (x1, y1, x2, y2) = float(box.conf[0]), box.xyxy[0]
                    phones.append({"confidence": conf,
                                   "bbox": [x1, y1, x2, y2]})
                    cv2.rectangle(annotated,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.putText(annotated,
                                f"Phone {conf:.2f}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)

        # --- encode back ----------
        _, buf = cv2.imencode(".jpg", annotated)
        img_b64 = base64.b64encode(buf).decode()

        return {"phones": phones,
                "count": len(phones),
                "annotated_image": img_b64}

    except Exception as e:
        raise HTTPException(500, str(e))

# --- local dev ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Mobile Phone Detection API at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
