import numpy as np
import cv2
from PIL import Image
import io
import objc
from Foundation import NSData, NSImage
from Quartz import CIImage
from Vision import VNDetectFaceCaptureQualityRequest, VNImageRequestHandler

def detect_face_quality(image):
    """
    Detect face quality in image
    
    Args:
        image (numpy.ndarray): Image as numpy array
        
    Returns:
        float: Face quality score between 0.0 and 1.0
    """
    # Convert numpy array to NSImage for Vision framework
    success, encoded_image = cv2.imencode('.png', image)
    image_data = NSData.dataWithBytes_length_(encoded_image.tobytes(), len(encoded_image.tobytes()))
    
    # Create CIImage from image data for Vision processing
    ci_image = CIImage.imageWithData_(image_data)
    
    # Create request and handler
    request = VNDetectFaceCaptureQualityRequest.alloc().init()
    handler = VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    
    # Perform request
    success = handler.performRequests_error_([request], None)
    
    if not success:
        return 0.0
    
    results = request.results()
    if not results:
        return 0.0
    
    # Return face quality score
    return results[0].faceCaptureQuality()