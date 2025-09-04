import base64
import json
from typing import Annotated
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, Request
from fastapi.params import Query
from fastapi.responses import FileResponse, Response
import sqlalchemy
from sqlalchemy.orm import Session
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import shutil
import time
import glob
import hashlib
from db import get_db,init_db
from models import User
from repository import authenticate_user, create_detection_object, create_prediction, create_user, delete_prediction_detection_session, query_average_score_last_7_days, query_detection_objects_by_prediction_last_week, query_detection_objects_by_prediction_uid, query_most_common_labels_last_7_days, query_prediction_by_label, query_prediction_by_score, query_prediction_by_uid, query_prediction_count, query_prediction_count_last_7_days, query_prediction_image_by_uid
from pydantic import BaseModel
import boto3
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
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")
S3_BUCKET = os.getenv("S3_BUCKET", "hamad-yolo-images")


s3_client = boto3.client("s3", region_name=AWS_REGION)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")

# Initialize SQLite

@app.on_event("startup")
def on_startup():
    init_db() # pragma: no cover

    


# @app.post("/signup")
# async def signup(request: Request):
#     data = await request.json()
#     username = data.get("username")
#     password = data.get("password")
#     with sqlite3.connect(DB_PATH) as conn:
#         conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
#     return {"message": "User created successfully"}

import base64
from fastapi import HTTPException
class UserCreate(BaseModel):
    username: str
    password: str
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Security

security = HTTPBasic()

@app.post("/login")
def login(credentials: HTTPBasicCredentials = Security(security), db: Session = Depends(get_db)):
    username = credentials.username
    password = credentials.password
    authenticate_user(db, username, password)
    return {"message": f"User '{username}' logged in successfully.", "Authorization": f"Basic {base64.b64encode(f'{username}:{password}'.encode()).decode()}"}





@app.get("/labels")
def get_labels(request: Request,db:Session=Depends(get_db)):
    """
    Get labels of objects detected in the last week
    """
    #require_auth(request,db)
    # with sqlite3.connect(DB_PATH) as conn:
    #     conn.row_factory = sqlite3.Row
    #     rows = conn.execute("""
    #         SELECT DISTINCT do.label
    #         FROM detection_objects do
    #         JOIN prediction_sessions ps ON do.prediction_uid = ps.uid
    #         WHERE ps.timestamp >= DATETIME('now', '-7 days')
    #     """).fetchall()
    rows = query_detection_objects_by_prediction_last_week(db)

    return [row.label for row in rows]

def save_prediction_session(db : Session,uid, original_image, predicted_image, user_id=None):
    # with sqlite3.connect(DB_PATH) as conn:
    #     conn.execute("""
    #         INSERT INTO prediction_sessions (uid, original_image, predicted_image, user_id)
    #         VALUES (?, ?, ?, ?)
    #     """, (uid, original_image, predicted_image, user_id))
    try:
        create_prediction(db, uid, original_image, predicted_image,user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save prediction session: {str(e)}")




def save_detection_object(db: Session,prediction_uid, label, score, box):
    # """
    # Save detection object to database
    # """
    # with sqlite3.connect(DB_PATH) as conn:
    #     conn.execute("""
    #         INSERT INTO detection_objects (prediction_uid, label, score, box)
    #         VALUES (?, ?, ?, ?)
    #     """, (prediction_uid, label, score, str(box)))
    try:
        box_str = json.dumps(box)
        create_detection_object(db, prediction_uid, label, score, box_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save detection object: {str(e)}")
@app.get("/list-s3")
def list_s3():
    objects = s3_client.list_objects_v2(Bucket=S3_BUCKET)
    keys = [obj["Key"] for obj in objects.get("Contents", [])]
    return {"bucket": S3_BUCKET, "objects": keys}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    s3_key = f"{file.filename}"
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=file.file)
    return {"message": "File uploaded successfully"}

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """
    Get statistics for the last week:
    - Total number of prediction sessions
    - Average confidence score from detection_objects
    - Most common detected labels (top 3)
    """
    # with sqlite3.connect(DB_PATH) as conn:
    #     conn.row_factory = sqlite3.Row

    #     # Total predictions in last 7 days
    #     total_predictions_row = conn.execute("""
    #         SELECT COUNT(*) as total FROM prediction_sessions
    #         WHERE timestamp >= DATETIME('now', '-7 days')
    #     """).fetchone()
    #     total_predictions = total_predictions_row.total
    try:
        total_predictions = query_prediction_count_last_7_days(db)
    except Exception as e:    
        raise HTTPException(status_code=500, detail=f"Failed to get prediction count: {str(e)}")

        # Average score for detections only from the last 7 days
        # avg_score_row = conn.execute("""
        #     SELECT AVG(do.score) as avg_score
        #     FROM detection_objects do
        #     JOIN prediction_sessions ps ON do.prediction_uid = ps.uid
        #     WHERE ps.timestamp >= DATETIME('now', '-7 days')
        # """).fetchone()
        # average_confidence_score = avg_score_row.avg_score or 0.0
    try:
        average_confidence_score = query_average_score_last_7_days(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get average score: {str(e)}")

        # Most common labels in last 7 days
        # Top 3 most common labels in last 7 days
        # common_labels_rows = conn.execute("""
        #     SELECT do.label, COUNT(*) as count
        #     FROM detection_objects do
        #     JOIN prediction_sessions ps ON do.prediction_uid = ps.uid
        #     WHERE ps.timestamp >= DATETIME('now', '-7 days')
        #     GROUP BY do.label
        #     ORDER BY count DESC
        #     LIMIT 3
        # """).fetchall()
        # most_common_labels = {row.label: row.count for row in common_labels_rows}
    try:
        most_common_labels = query_most_common_labels_last_7_days(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get most common labels: {str(e)}")

    return {
        "total_predictions": total_predictions,
        "average_confidence_score": round(average_confidence_score, 3),
        "most_common_labels": most_common_labels
    
    }

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()
def get_user_id(auth_header: str | None, db: Session) -> int | None:
    if auth_header and auth_header.startswith("Basic "):
        import secrets
        try:
            decoded = base64.b64decode(auth_header.split(" ")[1]).decode("utf-8")
            username, password = decoded.split(":", 1)
            hashed_pw = hash_password(password)
            #need to get the user from fucntion in repository file
            user = authenticate_user(db, username, password)
            if user and secrets.compare_digest(user.password, hashed_pw):
                return user.id
            return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to authenticate user: {str(e)}")
    else:
        return None
def require_auth(request: Request, db: Session) -> int:
    user_id = get_user_id(request.headers.get("Authorization"), db)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user_id

@app.post("/signup")
def add_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        create_user(db, user.username, user.password)
    except sqlalchemy.exc.IntegrityError:
        db.rollback()
        return {"message": f"User '{user.username}' already exists."}
    
    token = base64.b64encode(f"{user.username}:{user.password}".encode()).decode()
    return {
        "message": f"User '{user.username}' created successfully.",
        "Authorization": f"Basic {token}"
    }
@app.post("/predict")
def predict(
    request: Request,
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
    chat_id: str = Query(None, description="Chat ID for S3 storage"),
    img: str = Query(None, description="S3 key")
    
):
    """
    Predict objects in an image.
    Supports:
    - Upload via `file`
    - Download via `img` query param (from S3)
    Stores results in S3 under <bucket>/<chat_id>/original/ and /predicted/.
    """

    user_id = get_user_id(request.headers.get("Authorization"), db)
    start_time = time.time()
    if chat_id:
        s3_chat_id = chat_id
    elif user_id:
        s3_chat_id = str(user_id)
    else:
        s3_chat_id = "anonymous"
    uid = str(uuid.uuid4())
    # chat_id = str(user_id) if user_id else "anonymous"

    # --- Step 1: Retrieve Image ---
    if img:  # Download from S3
        ext = os.path.splitext(img)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(status_code=400, detail="Invalid image format from S3")
        original_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
        try:
            s3_client.download_file(S3_BUCKET, img, original_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not download {img} from S3: {e}")

    elif file:  # Uploaded directly
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")
        original_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
        with open(original_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    else:
        raise HTTPException(status_code=400, detail="Either file upload or img query parameter required")

    predicted_path = os.path.join(PREDICTED_DIR, f"pred_{uid}{ext}")

    # --- Step 2: Run YOLO Detection ---
    try:
        results = model(original_path, device="cpu")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    # --- Step 3: Save to DB ---
    save_prediction_session(db, uid, original_path, predicted_path, user_id)

    detected_labels = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            label_idx = int(box.cls[0].item())
            label = model.names[label_idx]
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            save_detection_object(db, uid, label, score, bbox)
            detected_labels.append(label)

    # --- Step 4: Upload images to S3 ---
    try:
        s3_client.upload_file(original_path, S3_BUCKET, f"{s3_chat_id}/original/{uid}{ext}")
        s3_client.upload_file(predicted_path, S3_BUCKET, f"{s3_chat_id}/predicted/{uid}{ext}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {e}")

    # --- Step 5: Cleanup local files ---
    if os.path.exists(original_path):
        os.remove(original_path)
    if os.path.exists(predicted_path):
        os.remove(predicted_path)

    # --- Step 6: Return response ---
    return {
        "prediction_uid": uid,
        "detection_count": len(detected_labels),
        "labels": detected_labels,
        "time_took": time.time() - start_time,
        "user_id": user_id,
        "s3_original": f"s3://{S3_BUCKET}/{chat_id}/original/{uid}{ext}",
        "s3_predicted": f"s3://{S3_BUCKET}/{chat_id}/predicted/{uid}{ext}"
    }


@app.get("/prediction/count")
def get_prediction_count(request: Request,db: Session = Depends(get_db)):
    """
    Get total number of prediction sessions
    """
    print("get_prediction_count")
    require_auth(request,db)
    # with sqlite3.connect(DB_PATH) as conn:
    #     count = conn.execute("SELECT count(*) FROM prediction_sessions WHERE timestamp >= DATETIME('now', '-7 days')").fetchall()
    # return {"count": count[0][0]}
    try:
        count = query_prediction_count(db)
        print(count,"count")
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction count: {str(e)}")
    
@app.delete("/prediction/{uid}")
def delete_prediction(request: Request,uid: str,db: Session = Depends(get_db)):
    """
    Delete prediction session by uid
    """
    require_auth(request,db)
    original_files = glob.glob(os.path.join(UPLOAD_DIR, uid + ".*"))
    predicted_files = glob.glob(os.path.join(PREDICTED_DIR, uid + ".*"))

    if not original_files or not predicted_files:
        raise HTTPException(status_code=404, detail="Prediction not found")

    try:
        for f in original_files + predicted_files:
            print(f,"removed")
            os.remove(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete files: {str(e)}")

    try:
        delete_prediction_detection_session(db,uid)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete prediction session: {str(e)}")
    # with sqlite3.connect(DB_PATH) as conn:
    #     try:
    #         conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))
    #         conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (uid,))
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=f"Database deletion failed: {str(e)}")

    return {"message": "Prediction session deleted"}

@app.get("/prediction/{uid}")
def get_prediction_by_uid(request: Request,uid: str,db: Session = Depends(get_db)):
    """
    Get prediction session by uid with all detected objects
    """
    require_auth(request,db)
    session = query_prediction_by_uid(db,uid)
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")
    # Get all detection objects for this prediction
    objects = query_detection_objects_by_prediction_uid(db,uid)
    if not objects:
        raise HTTPException(status_code=404, detail="Prediction not found in detection objects")
    
    print(session,"session")
    print(objects,"objects")
    return {
            "prediction_uid": session.prediction_uid,
            "timestamp": session.timestamp,
            "original_image": session.original_image,
            "predicted_image": session.predicted_image,
            "detection_objects": [
                {
                    "id": obj.id,
                    "label": obj.label,
                    "score": obj.score,
                    "box": obj.box
                } for obj in objects
            ]
        }

@app.get("/predictions/label/{label}")
def get_predictions_by_label(request: Request,label: str,db:Session=Depends(get_db)):
    """
    Get prediction sessions containing objects with specified label
    """
    require_auth(request,db)
    if label not in labels :
       raise HTTPException(status_code=400, detail="Invalid image type")
    # with sqlite3.connect(DB_PATH) as conn:
    #     conn.row_factory = sqlite3.Row
    #     rows = conn.execute("""
    #         SELECT DISTINCT ps.uid, ps.timestamp
    #         FROM prediction_sessions ps
    #         JOIN detection_objects do ON ps.uid = do.prediction_uid
    #         WHERE do.label = ?
    #     """, (label,)).fetchall()
    rows = query_prediction_by_label(db,label)
    if not rows:
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    return [{"prediction_uid": row.prediction_uid, "timestamp": row.timestamp} for row in rows]

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(request: Request,min_score: float,db: Session = Depends(get_db)):
    """
    Get prediction sessions containing objects with score >= min_score
    """
    require_auth(request,db)
    # with sqlite3.connect(DB_PATH) as conn:
    #     conn.row_factory = sqlite3.Row
    #     rows = conn.execute("""
    #         SELECT DISTINCT ps.uid, ps.timestamp
    #         FROM prediction_sessions ps
    #         JOIN detection_objects do ON ps.uid = do.prediction_uid
    #         WHERE do.score >= ?
    #     """, (min_score,)).fetchall()
        
    #     return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]
    try:
        rows = query_prediction_by_score(db,min_score)
        return [{"prediction_uid": row.prediction_uid, "timestamp": row.timestamp} for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictions by score: {str(e)}")
    

@app.get("/image/{type}/{filename}")
def get_image(request: Request,type: str, filename: str,db: Session = Depends(get_db)):
    """
    Get image by type and filename
    """
    require_auth(request,db)
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    path = os.path.join("uploads", type, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request,db: Session = Depends(get_db)):
    """
    Get prediction image by uid
    """
    require_auth(request,db)
    accept = request.headers.get("accept", "")
    # with sqlite3.connect(DB_PATH) as conn:
    #     row = conn.execute("SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
    #     if not row:
    #         raise HTTPException(status_code=404, detail="Prediction not found")
    #     image_path = row[0]
    print("query pred image uid ")
    image_path = query_prediction_image_by_uid(db,uid)
    if not image_path:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
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
    return {"status": "ok"} # pragma: no cover

if __name__ == "__main__": # pragma: no cover
    import uvicorn # pragma: no cover
    uvicorn.run("app:app", host="0.0.0.0", port=8080,reload=True) # pragma: no cover
