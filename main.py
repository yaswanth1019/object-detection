from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2
from pycocotools.coco import COCO
import os
import io
from typing import Dict, Any

app = FastAPI(
    title="Object Detection API",
    version="1.0.0",
    description="Mobile Object Detection API deployed on Railway"
)

# Add CORS middleware for mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Your existing model definition (ImprovedCNNDetector class)
class ImprovedCNNDetector(nn.Module):
    def __init__(self, num_classes: int):
        super(ImprovedCNNDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(32,448),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(448, num_classes)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(32,448),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(448, 4)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        cls_logits = self.classifier(x)
        bbox_preds = self.bbox_regressor(x)
        return cls_logits, bbox_preds

# Configuration - Use environment variables for Railway
MODEL_PATH = os.getenv("MODEL_PATH", "best_cnn_detector.pth")
COCO_JSON_PATH = os.getenv("COCO_JSON_PATH", "tirupati_data_robo/test/_annotations.coco.json")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Railway Deployment - Using device: {device}")

# Load COCO categories and create a mapping
try:
    coco = COCO(COCO_JSON_PATH)
    cats = coco.loadCats(coco.getCatIds())

    cat_id_to_contiguous_id = {cat['id']: idx for idx, cat in enumerate(cats)}
    contiguous_id_to_cat_id = {idx: cat['id'] for idx, cat in enumerate(cats)}
    cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

    num_classes = len(cats)
    print(f"✅ Loaded {num_classes} classes from COCO dataset.")

except Exception as e:
    raise RuntimeError(f"❌ Failed to load COCO annotations: {e}")

# Initialize and load the model
try:
    model = ImprovedCNNDetector(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Transform
transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

@app.get("/")
async def root():
    return {
        "message": "🚀 Object Detection API is running on Railway!",
        "status": "healthy",
        "classes": num_classes,
        "device": str(device),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
        "classes": num_classes
    }

@app.get("/classes")
async def get_classes():
    """Get the list of classes in the COCO dataset."""
    class_names = {}
    for contiguous_id, cat_id in contiguous_id_to_cat_id.items():
        class_names[contiguous_id] = cat_id_to_name[cat_id]
    return {"classes": class_names}

@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    """Predict object class and bounding box from uploaded image."""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    # Check file size (limit to 50MB for mobile)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Store original image dimensions for bbox scaling
        orig_width, orig_height = image.size
        print(f"📱 Processing image: {orig_width}x{orig_height} from mobile device")

        # Apply transforms
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            pred_cls, pred_box = model(image_tensor)

            # Get predicted class
            class_probs = torch.softmax(pred_cls, dim=1)
            confidence, class_idx = torch.max(class_probs, dim=1)
            confidence = confidence.item()
            class_idx = class_idx.item()

            # Map contiguous class index back to COCO category
            coco_cat_id = contiguous_id_to_cat_id.get(class_idx)
            class_name = cat_id_to_name.get(coco_cat_id, "Unknown") if coco_cat_id else "Unknown"

            # Get bounding box
            bbox = pred_box.squeeze(0).cpu().tolist()

            # Scale bounding box to original image size
            scaled_bbox = [
                bbox[0] * orig_width,   # x1
                bbox[1] * orig_height,  # y1
                bbox[2] * orig_width,   # x2
                bbox[3] * orig_height   # y2
            ]

        result = {
            "filename": file.filename,
            "image_size": {"width": orig_width, "height": orig_height},
            "prediction": {
                "class_id": class_idx,
                "coco_category_id": coco_cat_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": {
                    "raw": bbox,
                    "scaled": scaled_bbox,
                    "format": "x1_y1_x2_y2"
                }
            },
            "processing_info": {
                "device": str(device),
                "deployment": "Railway"
            }
        }
        
        print(f"✅ Detection complete: {class_name} ({confidence*100:.1f}%)")
        return result
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)