import os
import json
from io import BytesIO
import base64
import logging
import numpy as np
import cv2
from dotenv import find_dotenv, load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from langchain_anthropic import ChatAnthropic
from langchain.schema.messages import HumanMessage

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TARGET_IMAGE_SIZE = (1024, 1024)  
MAX_FILE_SIZE = 2 * 1024 * 1024  

INPUT_DIR = "../data/input"
OUTPUT_DIR = "../data/output"

def process_pdf(pdf_path):
    """Process a PDF file page by page and get a description of each page"""
    output_data = {"messages": []}

    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        page_count = len(reader.pages)
        for page_number in range(page_count):
            try:
                images = convert_from_path(
                    pdf_path, first_page=page_number + 1, last_page=page_number + 1
                )
                image = images[0]
                
                # Preprocess the image
                preprocessed_image = preprocess_image(image)
                
                # Resize the preprocessed image
                resized_image = resize_image(preprocessed_image)
                
                base64_image = encode_image(resized_image)
                
                # Check if the encoded image exceeds the max file size
                if len(base64_image) > MAX_FILE_SIZE:
                    logging.warning(f"Encoded image for page {page_number+1} exceeds max file size. Reducing quality.")
                    base64_image = reduce_image_quality(resized_image)
                
                description = get_page_description(base64_image)

                output_data["messages"].append({
                    "page_number": page_number + 1,
                    "description": description
                })
                logging.info(
                    f"Processed page {page_number+1} of {pdf_path}"
                )
            except Exception as e:
                logging.warning(
                    f"Error processing page {page_number+1} of {pdf_path}: {str(e)}"
                )
                output_data["messages"].append({
                    "page_number": page_number + 1,
                    "description": f"Error processing page: {str(e)}"
                })

    return output_data

def preprocess_image(image):
    """Apply advanced preprocessing techniques to improve image quality and enhance colors"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if image is in RGBA mode
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Apply CLAHE to L channel in LAB color space
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Enhance saturation, especially for red tones
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase overall saturation
    s = cv2.add(s, 30)
    
    # Enhance red tones (Hue values around 0 and 180)
    red_mask = cv2.inRange(h, 0, 10) | cv2.inRange(h, 170, 180)
    s[red_mask] = cv2.add(s[red_mask], 50)
    
    hsv_enhanced = cv2.merge([h, s, v])
    color_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(color_enhanced)
    
    # Further enhance contrast and color
    enhancer = ImageEnhance.Contrast(processed_image)
    contrast_enhanced = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Color(contrast_enhanced)
    final_image = enhancer.enhance(1.4)
    
    return final_image


def resize_image(image):
    """Resize the image to fit within the target size while maintaining aspect ratio"""
    image.thumbnail(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
    return image

def encode_image(image):
    """Encode an image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def reduce_image_quality(image, quality=85):
    """Reduce the quality of the image to decrease file size"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_page_description(base64_image):
    """Get a description of the page using Claude Haiku"""
    chat = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=1024)

    # chat = ChatAnthropic(model="claude-3-5-sonnet-20240620", max_tokens=1024)

 
    prompt = """
    Is this person bleeding? 
 
    """

    try:
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ]
                )
            ]
        )
        return msg.content.strip()
    except Exception as e:
        logging.error(f"Error getting page description: {str(e)}")
        return f"Error occurred while processing the image: {str(e)}"

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process all PDF files in the input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(INPUT_DIR, filename)
            output_data = process_pdf(pdf_path)

            # Create output JSON file
            output_filename = os.path.splitext(filename)[0] + '_output.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, 'w') as outfile:
                json.dump(output_data, outfile, indent=2)

            logging.info(f"Processed {filename} and saved results to {output_filename}")

if __name__ == "__main__":
    main()