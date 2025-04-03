# File: app/services/card_detect.py
import numpy as np
import cv2
from PIL import Image
import io
import objc
from Foundation import NSData, NSImage
from Quartz import CIImage
from Vision import VNDetectRectanglesRequest, VNImageRequestHandler

def detect_card(image):
    """
    Detect card in image and correct perspective
    
    Args:
        image (numpy.ndarray): Image as numpy array
        
    Returns:
        numpy.ndarray: Perspective corrected image of the card
    """
    # Convert numpy array to NSImage for Vision framework
    success, encoded_image = cv2.imencode('.png', image)
    image_data = NSData.dataWithBytes_length_(encoded_image.tobytes(), len(encoded_image.tobytes()))
    ns_image = NSImage.alloc().initWithData_(image_data)
    
    # Use Vision framework to detect rectangles
    request = VNDetectRectanglesRequest.alloc().init()
    request.setMinimumAspectRatio_(0.5)
    request.setMaximumAspectRatio_(2.0)
    request.setQuadratureTolerance_(10.0)  # Allow for some perspective distortion
    
    # Create CIImage from NSImage for Vision processing
    ci_image = CIImage.imageWithData_(image_data)
    
    # Create handler and perform request
    handler = VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
    success = handler.performRequests_error_([request], None)
    
    if not success:
        return image  # Return original if detection failed
    
    results = request.results()
    if not results:
        return image  # Return original if no rectangles detected
    
    # Get the first (hopefully largest/most prominent) rectangle
    observation = results[0]
    
    # Get the corners of the detected rectangle
    # Vision uses normalized coordinates (0-1)
    h, w = image.shape[:2]
    
    # Get rectangle coordinates
    box_points = observation.boundingBox()
    top_left = (int(box_points.origin.x * w), int(box_points.origin.y * h))
    top_right = (int((box_points.origin.x + box_points.size.width) * w), int(box_points.origin.y * h))
    bottom_right = (int((box_points.origin.x + box_points.size.width) * w), 
                   int((box_points.origin.y + box_points.size.height) * h))
    bottom_left = (int(box_points.origin.x * w), int((box_points.origin.y + box_points.size.height) * h))
    
    src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    # Define the size of the output image (standardized card size)
    card_w, card_h = 640, 400  # Standard card aspect ratio
    dst_pts = np.array([[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]], dtype=np.float32)
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply perspective transformation
    warped = cv2.warpPerspective(image, M, (card_w, card_h))
    
    return warped