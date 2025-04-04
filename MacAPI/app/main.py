import time
import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import io
from PIL import Image
from datetime import datetime

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

# ค่าคงที่สำหรับการคำนวณ Rack Cooling Rate (ปรับได้ตามต้องการ)
COOLING_FACTOR = 0.5  # ปัจจัยสมมติสำหรับปรับสเกล Rack Cooling Rate

@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...), 
    languages: str = None,
    db: Session = Depends(get_db)
):
    """
    Extract text from image using OCR and calculate processing rates
    
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
        
        # คำนวณ Fast Rate (operations per second)
        fast_rate = 1.0 / processing_time if processing_time > 0 else 0.0
        
        # คำนวณ Rack Cooling Rate (สมมติเป็นหน่วยสมมติ เช่น efficiency units per second)
        cooling_rate = 1.0 / (processing_time * COOLING_FACTOR) if processing_time > 0 else 0.0
            
        # Save processed image
        timestamp = int(time.time())
        processed_output_path = f"output/{timestamp}_ocr_processed.jpg"
        save_image(image, processed_output_path)  
        
        # Save result to database
        db_result = models.ProcessingResult(
            filename=file.filename,
            processing_type="ocr",
            result=result["text"],
            processing_time=processing_time
        )
        db.add(db_result)
        db.commit()
        
        # Format response with rates, dimensions, and created time
        response = {
            "text": result["text"],
            "dimensions": {
                "width": result["dimensions"][0],
                "height": result["dimensions"][1]
            },
            "processing_time": round(processing_time, 4),
            "fast_rate": round(fast_rate, 4),
            "rack_cooling_rate": round(cooling_rate, 4),
            "processed_output_path": processed_output_path,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add languages used if specified or detected
        if language_list:
            response["languages_used"] = language_list
        if "detected_languages" in result and result["detected_languages"]:
            response["detected_languages"] = result["detected_languages"]
        
        return response
    except Exception as e:
        print(f"Error in OCR endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/face_quality")
async def face_quality(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Detect face quality in image and calculate processing rates
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        start_time = time.time()
        image = load_image(await file.read())
        
        # Process image
        quality_score = detect_face_quality(image)
        processing_time = time.time() - start_time

        # คำนวณ Fast Rate (operations per second)
        fast_rate = 1.0 / processing_time if processing_time > 0 else 0.0
        
        # คำนวณ Rack Cooling Rate
        cooling_rate = 1.0 / (processing_time * COOLING_FACTOR) if processing_time > 0 else 0.0

        # Save processed image
        timestamp = int(time.time())
        processed_output_path = f"output/{timestamp}_face_quality_processed.jpg"
        save_image(image, processed_output_path) 
        
        # Get image dimensions
        height, width = image.shape[:2]
        
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
            "dimensions": {
                "width": width,
                "height": height
            },
            "processing_time": round(processing_time, 4),
            "fast_rate": round(fast_rate, 4),
            "rack_cooling_rate": round(cooling_rate, 4),
            "processed_output_path": processed_output_path,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"Error in face quality endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/card_detection")
async def card_detection(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Detect card in image, correct perspective, and calculate processing rates
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        start_time = time.time()
        image = load_image(await file.read())
        
        # Process image
        processed_image = detect_card(image)
        processing_time = time.time() - start_time
        
        # คำนวณ Fast Rate (operations per second)
        fast_rate = 1.0 / processing_time if processing_time > 0 else 0.0
        
        # คำนวณ Rack Cooling Rate
        cooling_rate = 1.0 / (processing_time * COOLING_FACTOR) if processing_time > 0 else 0.0

         # Save processed image
        timestamp = int(time.time())
        processed_output_path = f"output/{timestamp}_card_detection_processed.jpg"
        save_image(image, processed_output_path)  
        
        # Get original image dimensions
        height, width = image.shape[:2]
        
        # Save result to database
        db_result = models.ProcessingResult(
            filename=file.filename,
            processing_type="card_detection",
            result=processed_output_path,
            processing_time=processing_time
        )
        db.add(db_result)
        db.commit()
        
        return {
            "message": "Card detected and corrected",
            "dimensions": {
                "width": width,
                "height": height
            },
            "processing_time": round(processing_time, 4),
            "fast_rate": round(fast_rate, 4),
            "rack_cooling_rate": round(cooling_rate, 4),
            "processed_output_path": processed_output_path,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
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