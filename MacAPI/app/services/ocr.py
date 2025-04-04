# File: app/services/ocr.py
import numpy as np
import cv2
from PIL import Image
import io
import objc
from Foundation import NSData, NSImage
from Quartz import CIImage
from Vision import VNRecognizeTextRequest, VNImageRequestHandler, VNRecognizeTextRequestRevision3

def recognize_text(image, languages=None):
    """
    Recognize text in image with text regions information
    
    Args:
        image (numpy.ndarray): Image as numpy array
        languages (list): List of language codes to recognize (e.g. ['en', 'th', 'ja'])
                          None for automatic language detection
    
    Returns:
        dict: Dictionary containing:
            - text: Extracted full text from image
            - dimensions: Image dimensions (width, height)
            - text_observations: List of detected text regions with text and bounding boxes
            - detected_languages: List of detected languages in the image
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Convert numpy array to NSData for Vision framework
    success, encoded_image = cv2.imencode('.png', image)
    image_data = NSData.dataWithBytes_length_(encoded_image.tobytes(), len(encoded_image.tobytes()))
    
    # Create CIImage from image data for Vision processing
    ci_image = CIImage.imageWithData_(image_data)
    
    # Create request and handler
    request = VNRecognizeTextRequest.alloc().init()
    
    # Set recognition level to accurate
    request.setRecognitionLevel_(1)  # 0 = fast, 1 = accurate
    
    # Use latest text recognition revision
    request.setRevision_(VNRecognizeTextRequestRevision3)
    
    # Set languages if specified, with thai/english defaults if not specified
    if languages:
        request.setRecognitionLanguages_(languages)
    else:
        # Default to recognizing both Thai and English if no languages are specified
        request.setRecognitionLanguages_(["th", "en"])
    
    # Create handler
    handler = VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    
    # Perform request
    success = handler.performRequests_error_([request], None)
    
    if not success:
        return {"text": "", "dimensions": (width, height), "text_observations": [], "detected_languages": []}
    
    results = request.results()
    if not results:
        return {"text": "", "dimensions": (width, height), "text_observations": [], "detected_languages": []}
    
    # Extract text and bounding boxes
    text_observations = []
    full_text = []
    detected_languages = set()
    
    for result in results:
        text = result.text()
        full_text.append(text)
        
        # Get language identification if available
        if hasattr(result, 'language') and result.language():
            detected_languages.add(result.language())
        
        # Get normalized bounding box
        boundingBox = result.boundingBox()
        
        # Convert normalized coordinates to pixel coordinates
        x = int(boundingBox.origin.x * width)
        y = int(boundingBox.origin.y * height)
        w = int(boundingBox.size.width * width)
        h = int(boundingBox.size.height * height)
        
        # Convert to top-left coordinates (Vision uses bottom-left origin)
        y = height - y - h
        
        # text_observations.append({
        #     "text": text,
        #     "confidence": result.confidence(),
        #     "bounding_box": {
        #         "x": x,
        #         "y": y,
        #         "width": w,
        #         "height": h
        #     }
        # })
    
    return {
        "text": " ".join(full_text),
        "dimensions": (width, height),
        "text_observations": text_observations,
        "detected_languages": list(detected_languages)
    }

def get_supported_languages():
    """
    Get list of supported languages for OCR
    
    Returns:
        dict: Dictionary of language codes and their names
    """
    # List of common supported languages in Vision framework
    return {
        "th": "Thai",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "zh-Hans": "Simplified Chinese",
        "zh-Hant": "Traditional Chinese",
        "fr": "French",
        "it": "Italian",
        "de": "German",
        "es": "Spanish",
        "pt": "Portuguese",
        "ru": "Russian"
    }