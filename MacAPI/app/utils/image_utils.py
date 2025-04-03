import numpy as np
import cv2
from PIL import Image
import io
import os

def load_image(image_bytes):
    """
    Load image from bytes
    
    Args:
        image_bytes (bytes): Raw image bytes
        
    Returns:
        numpy.ndarray: Image as numpy array
    """
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)

def save_image(image, path):
    """
    Save image to file
    
    Args:
        image (numpy.ndarray): Image as numpy array
        path (str): Path to save image
        
    Returns:
        bool: True if successful
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save image
    return cv2.imwrite(path, image)