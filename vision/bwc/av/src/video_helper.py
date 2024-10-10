import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def preprocess_image(image):
    """Apply advanced preprocessing techniques to enhance overall image quality without emphasizing saturated colors or dark regions"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if image is in RGBA mode
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    
    # Convert to LAB color space for more accurate color processing
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel to improve overall contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge back and convert to RGB
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Apply bilateral filter to smooth while preserving edges
    smooth = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(smooth)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(processed_image)
    contrast_enhanced = enhancer.enhance(1.2)
    
    # Enhance sharpness one more time
    sharpness_enhancer = ImageEnhance.Sharpness(contrast_enhanced)
    final_image = sharpness_enhancer.enhance(1.3)
    
    return final_image

def encode_frame(frame):
    """Encode a video frame to base64"""
    # Convert frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Preprocess the image
    processed_image = preprocess_image(pil_image)
    
    # Convert back to OpenCV format
    processed_frame = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode(".jpg", processed_frame)
    return base64.b64encode(buffer).decode("utf-8")
