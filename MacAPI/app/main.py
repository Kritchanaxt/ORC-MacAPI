# File: app/main.py
import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import io
from PIL import Image

from app.database import get_db, engine, Base
from app import models
from app.services.ocr import recognize_text
from app.services.face_quality import detect_face_quality
from app.services.card_detect import detect_card
from app.utils.image_utils import load_image, save_image

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Document Processing API", 
              description="API for OCR, face quality, and card detection using Vision framework")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Extract text from image using OCR
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    image = load_image(await file.read())
    
    # Process image
    result = recognize_text(image)
    processing_time = time.time() - start_time
    
    # Save result to database
    db_result = models.ProcessingResult(
        filename=file.filename,
        processing_type="ocr",
        result=result,
        processing_time=processing_time
    )
    db.add(db_result)
    db.commit()
    
    return {
        "text": result, 
        "processing_time": processing_time,
        "result_id": db_result.id
    }

@app.post("/face_quality")
async def face_quality(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Detect face quality in image
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    image = load_image(await file.read())
    
    # Process image
    quality_score = detect_face_quality(image)
    processing_time = time.time() - start_time

    
    # Save result to database
    db_result = models.ProcessingResult(
        filename=file.filename,
        processing_type="face_quality",
        result=str(quality_score),
        processing_time=processing_time
    )
    db.add(db_result)
    db.commit()
    
    return {
        "quality_score": quality_score, 
        "processing_time": processing_time,
        "result_id": db_result.id
    }

@app.post("/card_detection")
async def card_detection(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Detect card in image and correct perspective
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    image = load_image(await file.read())
    
    # Process image
    processed_image = detect_card(image)
    processing_time = time.time() - start_time
    
    # Save processed image
    output_path = f"output/{int(time.time())}_card.jpg"
    save_image(processed_image, output_path)
    
    # Save result to database
    db_result = models.ProcessingResult(
        filename=file.filename,
        processing_type="card_detection",
        result=output_path,
        processing_time=processing_time
    )
    db.add(db_result)
    db.commit()
    
    return {
        "message": "Card detected and corrected", 
        "processing_time": processing_time,
        "result_id": db_result.id,
        "output_path": output_path
    }

@app.get("/processing_speed_comparison")
async def processing_speed_comparison(db: Session = Depends(get_db)):
    """
    Compare processing speed of different services
    """
    ocr_results = db.query(models.ProcessingResult).filter(
        models.ProcessingResult.processing_type == "ocr"
    ).all()
    
    face_quality_results = db.query(models.ProcessingResult).filter(
        models.ProcessingResult.processing_type == "face_quality"
    ).all()
    
    card_detection_results = db.query(models.ProcessingResult).filter(
        models.ProcessingResult.processing_type == "card_detection"
    ).all()
    
    def calculate_stats(results):
        if not results:
            return {"count": 0, "avg_time": 0, "min_time": 0, "max_time": 0}
            
        times = [result.processing_time for result in results]
        return {
            "count": len(results),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times)
        }
    
    return {
        "ocr": calculate_stats(ocr_results),
        "face_quality": calculate_stats(face_quality_results),
        "card_detection": calculate_stats(card_detection_results)
    }