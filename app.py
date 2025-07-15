from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import shutil
import time
import glob

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()
labels = [
   "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")

# Initialize SQLite
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)
        
        # Create the objects table to store individual detected objects in a given image
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)
        
        # Create index for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")

init_db()

@app.get("/labels")
def get_lables():
    """
    Get labels of objects detected in the last week
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT DISTINCT label FROM detection_objects join prediction_sessions WHERE timestamp >= DATETIME('now', '-7 days')").fetchall()
        print(rows)
    return [row["label"] for row in rows]

def save_prediction_session(uid, original_image, predicted_image):
    """
    Save prediction session to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (uid, original_image, predicted_image))

def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

@app.get("/stats")
async def get_stats():
    """
    Get statistics for the last week:
    - Total number of prediction sessions
    - Average confidence score from detection_objects
    - Most common detected labels (top 3)
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Total predictions in last 7 days
        total_predictions_row = conn.execute("""
            SELECT COUNT(*) as total FROM prediction_sessions
            WHERE timestamp >= DATETIME('now', '-7 days')
        """).fetchone()
        total_predictions = total_predictions_row["total"]

        # Average score for detections only from the last 7 days
        avg_score_row = conn.execute("""
            SELECT AVG(do.score) as avg_score
            FROM detection_objects do
            JOIN prediction_sessions ps ON do.prediction_uid = ps.uid
            WHERE ps.timestamp >= DATETIME('now', '-7 days')
        """).fetchone()
        average_confidence_score = avg_score_row["avg_score"] or 0.0

        # Top 3 most common labels in last 7 days
        common_labels_rows = conn.execute("""
            SELECT do.label, COUNT(*) as count
            FROM detection_objects do
            JOIN prediction_sessions ps ON do.prediction_uid = ps.uid
            WHERE ps.timestamp >= DATETIME('now', '-7 days')
            GROUP BY do.label
            ORDER BY count DESC
            LIMIT 3
        """).fetchall()
        most_common_labels = {row["label"]: row["count"] for row in common_labels_rows}

    return {
        "total_predictions": total_predictions,
        "average_confidence_score": round(average_confidence_score, 3),
        "most_common_labels": most_common_labels
    }


    
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Predict objects in an image
    todo in the ui make a commnet that the model acceptes only images jpeg/jpg and png 
    """
    start_time = time.time()

    ext = os.path.splitext(file.filename)[1]
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Invalid image format")
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        results = model(original_path, device="cpu")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    annotated_frame = results[0].plot()  # NumPy image with boxes
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    save_prediction_session(uid, original_path, predicted_path)
    
    detected_labels = []
    if results[0].boxes is None or results[0].boxes == []:
        return {
            "prediction_uid": uid, 
            "detection_count": 0,
            "labels": detected_labels,
            "time_took": time.time() - start_time
        }
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(uid, label, score, bbox)
        detected_labels.append(label)
    time_taken = time.time() - start_time
    return {
        "prediction_uid": uid, 
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": time_taken
    }

@app.get("/prediction/count/")
def get_prediction_count():
    """
    Get total number of prediction sessions
    """
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT count(*) FROM prediction_sessions WHERE timestamp >= DATETIME('now', '-7 days')").fetchall()
    return {"count": count[0][0]}
    
@app.delete("/prediction/{uid}")
def delete_prediction(uid: str):
    """
    Delete prediction session by uid
    """
    original_files = glob.glob(os.path.join(UPLOAD_DIR, uid + ".*"))
    predicted_files = glob.glob(os.path.join(PREDICTED_DIR, uid + ".*"))

    if not original_files or not predicted_files:
        raise HTTPException(status_code=404, detail="Prediction not found")

    try:
        for f in original_files + predicted_files:
            os.remove(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete files: {str(e)}")

    with sqlite3.connect(DB_PATH) as conn:
        try:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))
            conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (uid,))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database deletion failed: {str(e)}")

    return {"message": "Prediction session deleted"}

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    """
    Get prediction session by uid with all detected objects
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Get prediction session
        session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
        # Get all detection objects for this prediction
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", 
            (uid,)
        ).fetchall()
        
        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"]
                } for obj in objects
            ]
        }

@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    """
    Get prediction sessions containing objects with specified label
    """
    if label not in labels :
       raise HTTPException(status_code=400, detail="Invalid image type")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.label = ?
        """, (label,)).fetchall()
        
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float):
    """
    Get prediction sessions containing objects with score >= min_score
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.score >= ?
        """, (min_score,)).fetchall()
        
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str):
    """
    Get image by type and filename
    """
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    path = os.path.join("uploads", type, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request):
    """
    Get prediction image by uid
    """
    accept = request.headers.get("accept", "")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")
        image_path = row[0]

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Predicted image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        # If the client doesn't accept image, respond with 406 Not Acceptable
        raise HTTPException(status_code=406, detail="Client does not accept an image format")
    

@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080,reload=True)
