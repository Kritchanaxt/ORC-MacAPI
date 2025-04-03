# File: app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)

class ProcessingResult(Base):
    __tablename__ = "processing_results"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    processing_type = Column(String)  # 'ocr', 'face_quality', or 'card_detection'
    result = Column(String)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=func.now())