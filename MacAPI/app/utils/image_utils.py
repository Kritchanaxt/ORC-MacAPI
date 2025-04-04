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

def draw_text_with_thai_support(image, text, position, font_scale=1.0, color=(0, 0, 255), thickness=2):
    """
    Draw text on image with proper Thai language support
    
    Args:
        image (numpy.ndarray): Image to draw on
        text (str): Text to draw (supports Thai and English)
        position (tuple): Position (x, y) to draw text
        font_scale (float): Font scale
        color (tuple): BGR color tuple
        thickness (int): Line thickness
        
    Returns:
        numpy.ndarray: Image with text
    """
    # Convert to PIL Image for better text support
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Use PIL's ImageDraw to support Thai text
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil_img)
    
    # Try to use a font that supports Thai, falling back to default
    try:
        # Try to find a system font that supports Thai
        font = ImageFont.truetype("Arial Unicode MS", int(12 * font_scale))
    except IOError:
        try:
            font = ImageFont.truetype("Tahoma", int(12 * font_scale))
        except IOError:
            # Fall back to default
            font = ImageFont.load_default()
    
    # Draw text with PIL (which supports Thai)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    
    # Convert back to OpenCV format
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result