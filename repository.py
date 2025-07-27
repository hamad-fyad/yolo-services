import hashlib
import os
from fastapi import HTTPException
from sqlalchemy.orm import Session

from models import PredictionSession, DetectionObject, User
from datetime import datetime


def query_prediction_by_uid(db: Session,prediction_uid: str):
    return db.query(PredictionSession).filter_by(prediction_uid=prediction_uid).first()

def query_detection_objects_by_prediction_uid(db: Session,prediction_uid: str):
    return db.query(DetectionObject).filter_by(prediction_uid=prediction_uid).all()

def query_all_predictions(db: Session):
    return db.query(PredictionSession).all()

def query_all_detection_objects(db: Session):
    return db.query(DetectionObject).all()

def create_prediction(db: Session,prediction_uid: str, original_image: str, predicted_image: str):
    prediction = PredictionSession(prediction_uid=prediction_uid, original_image=original_image, predicted_image=predicted_image)
    db.add(prediction)
    db.commit()
    return prediction

def create_detection_object(db: Session,detection_object: DetectionObject):
    db.add(detection_object)
    db.commit()
    return detection_object

def update_prediction(db: Session,prediction_uid: str, original_image: str, predicted_image: str):
    prediction = db.query(PredictionSession).filter_by(prediction_uid=prediction_uid).first()
    prediction.original_image = original_image
    prediction.predicted_image = predicted_image
    db.commit()
    return prediction

def delete_prediction(db: Session,prediction_uid: str):
    prediction = db.query(PredictionSession).filter_by(prediction_uid=prediction_uid).first()
    db.delete(prediction)
    db.commit()
    return prediction

def delete_detection_object(db: Session,detection_object: DetectionObject):
    db.delete(detection_object)
    db.commit()
    return detection_object

def delete_all_predictions(db:Session):
    predictions = db.query(PredictionSession).all()
    for prediction in predictions:
        db.delete(prediction)
    db.commit()
    return predictions

def delete_all_detection_objects(db:Session):
    detection_objects = db.query(DetectionObject).all()
    for detection_object in detection_objects:
        db.delete(detection_object)
    db.commit()
    return detection_objects



def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(db:Session, username: str, password: str):
    hashed_password = hash_password(password)
    user = User(username=username, password=hashed_password)
    db.add(user)
    db.commit()
    return user
def authenticate_user(db:Session, username: str, password: str):
    hashed_password = hash_password(password)
    user = db.query(User).filter_by(username=username).first()
    if user and user.password == hashed_password:
        return user
    return None
def create_prediction(db:Session,prediction_uid: str, original_image: str, predicted_image: str, user_id: int=None):
    prediction = PredictionSession(prediction_uid=prediction_uid, original_image=original_image, predicted_image=predicted_image, user_id=user_id)
    db.add(prediction)
    db.commit()
    return prediction


def create_detection_object(db:Session,prediction_uid: str, label: str, score: float, box: str):
    detection = DetectionObject(prediction_uid=prediction_uid, label=label, score=score, box=box)
    db.add(detection)
    db.commit()
    return detection
    
def delete_prediction_detection_session(db:Session,prediction_uid: str):
    try:
        db.query(DetectionObject).filter_by(prediction_uid=prediction_uid).delete()
        db.query(PredictionSession).filter_by(prediction_uid=prediction_uid).delete()
        db.commit()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete detection objects: {str(e)}")
   
def query_prediction_image_by_uid(db:Session,prediction_uid: str):
    return db.query(PredictionSession).filter_by(prediction_uid=prediction_uid).first()
    

