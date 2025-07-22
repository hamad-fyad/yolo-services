# models.py

from sqlalchemy import Column, String, DateTime, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# All models inherit from this base class
Base = declarative_base()


class PredictionSession(Base):
    """
    Model for prediction_sessions table
    
    This replaces: CREATE TABLE prediction_sessions (...)
    """
    __tablename__ = 'prediction_sessions'
    
    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_image = Column(String)
    predicted_image = Column(String)

class DetectionObject(Base):
    """
    Model for detection_objects table
    
    This replaces: CREATE TABLE detection_objects (...)
    """
    __tablename__ = 'detection_objects'
    
    id = Column(Integer, primary_key=True)
    prediction_uid = Column(String)
    label = Column(String)
    score = Column(Float)
    box = Column(String)

class User(Base):
    """
    Model for users table
    
    This replaces: CREATE TABLE users (...)
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

