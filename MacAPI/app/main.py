# File: app/main.py
import time
import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import io
from PIL import Image

from app.database import get_db, engine, Base
from app import models
from app.services.ocr import recognize_text, get_supported_languages
from app.services.face_quality import detect_face_quality
from app.services.card_detect import detect_card
from app.utils.image_utils import load_image, save_image

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Document Processing API", 
              description="API for OCR, face quality, and card detection using Vision framework. Supports multiple languages including Thai (th) and English (en).")

@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...), 
    languages: str = None,
    db: Session = Depends(get_db)
):
    """
    Extract text from image using OCR
    
    - **file**: Image file to process
    - **languages**: Comma-separated list of language codes (e.g. 'en,th,ja')
                    Leave empty for automatic language detection (defaults to Thai and English)
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        start_time = time.time()
        image = load_image(await file.read())
        
        # Parse languages parameter
        language_list = None
        if languages:
            language_list = [lang.strip() for lang in languages.split(',')]
        
        # Process image
        result = recognize_text(image, language_list)
        processing_time = time.time() - start_time
        
        # Save original image
        output_path = f"output/{int(time.time())}_ocr.jpg"
        save_image(image, output_path)
        
        # Save result to database
        db_result = models.ProcessingResult(
            filename=file.filename,
            processing_type="ocr",
            result=result["text"],
            processing_time=processing_time
        )
        db.add(db_result)
        db.commit()
        
        # Format response
        response = {
            "text": result["text"],
            "dimensions": result["dimensions"],
            "text_regions": result["text_observations"],
            "processing_time": processing_time,
            "result_id": db_result.id,
            "output_path": output_path
        }
        
        # Add languages used if specified or detected
        if language_list:
            response["languages_used"] = language_list
        if "detected_languages" in result and result["detected_languages"]:
            response["detected_languages"] = result["detected_languages"]
        
        return response
    except Exception as e:
        # Log the error for debugging
        print(f"Error in OCR endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/supported_languages")
async def supported_languages():
    """
    Get list of supported languages for OCR
    """
    return get_supported_languages()


@app.post("/face_quality")
async def face_quality(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Detect face quality in image
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        start_time = time.time()
        image = load_image(await file.read())
        
        # Process image
        quality_score = detect_face_quality(image)
        processing_time = time.time() - start_time

        # Save original image (no processed image for face quality)
        output_path = f"output/{int(time.time())}_face.jpg"
        save_image(image, output_path)
        
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
            "result_id": db_result.id,
            "output_path": output_path
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Error in face quality endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/card_detection")
async def card_detection(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Detect card in image and correct perspective
    """
    try:
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
    except Exception as e:
        # Log the error for debugging
        print(f"Error in card detection endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/processing_speed_comparison")
async def processing_speed_comparison(db: Session = Depends(get_db)):
    """
    Compare processing speed of different services
    """
    try:
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
    except Exception as e:
        # Log the error for debugging
        print(f"Error in processing speed comparison endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting processing stats: {str(e)}")


@app.get("/output/{filename}")
async def get_output_file(filename: str):
    """
    Get processed output file
    """
    file_path = f"output/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)