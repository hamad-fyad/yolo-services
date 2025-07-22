from sqlalchemy.orm import Session
from models import PredictionSession, DetectionObject, User
from db import get_db

db = get_db()

def query_prediction_by_uid(db: Session,uid: str):
    return db.query(PredictionSession).filter_by(uid=uid).first()

def query_detection_objects_by_prediction_uid(db: Session,uid: str):
    return db.query(DetectionObject).filter_by(prediction_uid=uid).all()

def query_all_predictions(db: Session):
    return db.query(PredictionSession).all()

def query_all_detection_objects(db: Session):
    return db.query(DetectionObject).all()

def create_prediction(db: Session,uid: str, original_image: str, predicted_image: str):
    prediction = PredictionSession(uid=uid, original_image=original_image, predicted_image=predicted_image)
    db.add(prediction)
    db.commit()
    return prediction

def create_detection_object(db: Session,detection_object: DetectionObject):
    db.add(detection_object)
    db.commit()
    return detection_object

def update_prediction(db: Session,uid: str, original_image: str, predicted_image: str):
    prediction = db.query(PredictionSession).filter_by(uid=uid).first()
    prediction.original_image = original_image
    prediction.predicted_image = predicted_image
    db.commit()
    return prediction

def delete_prediction(db: Session,uid: str):
    prediction = db.query(PredictionSession).filter_by(uid=uid).first()
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

