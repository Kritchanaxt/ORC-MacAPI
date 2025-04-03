# File: app/services/ocr.py
import numpy as np
import cv2
from PIL import Image
import io
import objc
from Foundation import NSData, NSImage
from Quartz import CIImage
from Vision import VNRecognizeTextRequest, VNImageRequestHandler

def recognize_text(image):
    """
    Recognize text in image
    
    Args:
        image (numpy.ndarray): Image as numpy array
        
    Returns:
        str: Extracted text from image
    """
    # Convert numpy array to NSImage for Vision framework
    success, encoded_image = cv2.imencode('.png', image)
    image_data = NSData.dataWithBytes_length_(encoded_image.tobytes(), len(encoded_image.tobytes()))
    
    # Create CIImage from image data for Vision processing
    ci_image = CIImage.imageWithData_(image_data)
    
    # Create request and handler
    request = VNRecognizeTextRequest.alloc().init()
    handler = VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    
    # Perform request
    success = handler.performRequests_error_([request], None)
    
    if not success:
        return ""
    
    results = request.results()
    if not results:
        return ""
    
    # Extract and return text
    extracted_text = " ".join([res.text() for res in results])
    return extracted_text