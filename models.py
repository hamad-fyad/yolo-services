from sqlalchemy import Column, ForeignKey, String, DateTime, Integer, Float, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from db import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer,autoincrement=True, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

    predictions = relationship("PredictionSession", back_populates="user", cascade="all, delete-orphan")


class PredictionSession(Base):
    __tablename__ = 'prediction_sessions'
    prediction_uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_image = Column(String)
    predicted_image = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))

    user = relationship("User", back_populates="predictions")
    detection_objects = relationship("DetectionObject", back_populates="prediction_session", cascade="all, delete-orphan")


class DetectionObject(Base):
    __tablename__ = 'detection_objects'

    id = Column(Integer, primary_key=True)
    prediction_uid = Column(String, ForeignKey('prediction_sessions.prediction_uid'))
    label = Column(String)
    score = Column(Float)
    box = Column(String)

    prediction_session = relationship("PredictionSession", back_populates="detection_objects")


# Index definitions
Index("idx_prediction_uid", DetectionObject.prediction_uid)
Index("idx_label", DetectionObject.label)
Index("idx_score", DetectionObject.score)
