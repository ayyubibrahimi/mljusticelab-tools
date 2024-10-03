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
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


load_dotenv(find_dotenv())

# chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

chat = ChatOpenAI(model="gpt-4o-mini")
# chat = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=1024)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TARGET_IMAGE_SIZE = (800, 800)  # Reduced from 1024x1024
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
                
                # Resize first to reduce processing time
                resized_image = resize_image(image)
                
                # Preprocess the resized image
                preprocessed_image = preprocess_image(resized_image)
                
                # Encode with size control
                base64_image = encode_image_with_size_control(preprocessed_image, MAX_FILE_SIZE)
                
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
    """Apply advanced preprocessing techniques to enhance overall image quality"""
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
    
    # Adjust gamma to enhance details in darker regions
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(smooth, table)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(gamma_corrected)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(processed_image)
    contrast_enhanced = enhancer.enhance(1.2)
    
    # Enhance color
    color_enhancer = ImageEnhance.Color(contrast_enhanced)
    color_enhanced = color_enhancer.enhance(1.1)
    
    # Enhance sharpness one more time
    sharpness_enhancer = ImageEnhance.Sharpness(color_enhanced)
    final_image = sharpness_enhancer.enhance(1.3)
    
    return final_image

def resize_image(image):
    """Resize the image to fit within the target size while maintaining aspect ratio"""
    image.thumbnail(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
    return image

def encode_image_with_size_control(image, max_size, initial_quality=85):
    """Encode an image to base64 with size control"""
    quality = initial_quality
    while quality > 20:  # Set a lower bound for quality
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=quality, optimize=True)
        img_str = base64.b64encode(buffered.getvalue())
        if len(img_str) <= max_size:
            return img_str.decode('utf-8')
        quality -= 5
    raise ValueError("Unable to compress image to required size while maintaining acceptable quality")

def get_page_description(base64_image):
    """Get a description of the page using Claude Haiku"""
    
    prompt = """
    Provide a technical description of this image. 

    Write your description inside <image_analysis> tags. Begin your description with "The image depicts" or a similar phrase.
    """

    try:
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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