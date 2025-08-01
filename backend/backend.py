from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv10 model (downloads automatically if not present)
try:
    model = YOLO('yolov10n.pt')  # ~6MB download on first run
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.get("/")
async def root():
    return {"message": "Mobile Phone Detection API", "model_loaded": model is not None}

@app.post("/detect")
async def detect_phones(file: UploadFile = File(...), confidence: float = 0.5):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run detection
        results = model(cv_image, conf=confidence)
        
        # Filter for cell phone class (class 67 in COCO dataset)
        phones = []
        annotated_image = cv_image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    # Cell phone class ID is 67
                    if class_id == 67:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        phones.append({
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_image, f'Phone {conf:.2f}', 
                                  (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "phones": phones,
            "count": len(phones),
            "annotated_image": img_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Mobile Phone Detection API...")
    print("📱 Open your browser and load the HTML file")
    print("🔗 API will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)