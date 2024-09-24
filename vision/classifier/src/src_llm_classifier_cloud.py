import os
import json
from io import BytesIO
import base64
import logging
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from langchain.schema.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from concurrent.futures import ProcessPoolExecutor, as_completed
import fitz
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

LARGE_IMAGE_THRESHOLD = 3.7 * 1024 * 1024  # 3.7 megapixels
TARGET_IMAGE_SIZE = (800, 800)  # Reduced from 1024x1024
MAX_FILE_SIZE = 2 * 1024 * 1024



chat = ChatAnthropic(model_name="claude-3-haiku-20240307",  temperature=0)


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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge back and convert to RGB
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Sharpen the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Apply bilateral filter to smooth while preserving edges
    smooth = cv2.bilateralFilter(sharpened, 9, 75, 75)

    # Adjust gamma to enhance details in darker regions
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
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
            return img_str.decode("utf-8")
        quality -= 5
    raise ValueError(
        "Unable to compress image to required size while maintaining acceptable quality"
    )


def get_page_description(base64_image):
    """Get a description of the page using Claude Haiku"""

    prompt = """
    <task_description>
    Analyze the input image and determine whether it is a photograph or not. This is a binary classification task. Return True if the input is a photograph, and False otherwise. Do not provide any additional explanation or commentary.
    </task_description>

    <context>
    This classification will be used to quickly sort inputs into photographs and non-photographs. The majority of inputs may be documents or other non-photographic images, but the focus is on accurately identifying true photographs when they occur.
    </context>

    <classification_categories>
    1. Photograph: True
    2. Non-Photograph: False
    </classification_categories>

    <thinking_process>
    1. Observe the visual characteristics of the input.
    2. Identify key features that suggest whether it's a photograph or not.
    3. Consider any ambiguities or edge cases.
    4. Make a final determination based on the overall assessment.
    5. Return only True or False based on this determination.
    </thinking_process>

    <classification_guidelines>

    <photograph_indicators>
    - Realistic representation of real-world scenes or objects
    - Natural lighting, shadows, and textures
    - Depth of field and focus typical of camera lenses
    - Presence of photographic artifacts (e.g., lens flare, motion blur)
    </photograph_indicators>

    <non_photograph_indicators>
    - Stylized or abstract representations
    - Predominantly text-based content (e.g., documents, screenshots of text)
    - Graphical elements like charts, diagrams, or user interface components
    - Hand-drawn or digitally created illustrations
    - Computer-generated imagery or renderings
    </non_photograph_indicators>

    </classification_guidelines>

    <additional_instructions>
    - If the input is ambiguous, classify it based on the predominant characteristics.
    - Do not provide any explanation or reasoning in the output.
    - If no image is provided or there are technical issues preventing analysis, return False.
    </additional_instructions>

    <output_format>
    [True/False]
    </output_format>
    """

    try:
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ]
                )
            ]
        )
        return msg.content.strip()
    except Exception as e:
        logging.error(f"Error getting page description: {str(e)}")
        return f"Error occurred while processing the image: {str(e)}"


def process_page(pdf_path, page_number):
    """Process a single page of a PDF"""
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Get the specified page
        page = pdf_document[page_number]

        # Render page to an image
        pix = page.get_pixmap()

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Check image size
        if img.width * img.height > LARGE_IMAGE_THRESHOLD:
            logging.info(f"Large image detected on page {page_number+1} of {pdf_path}")
            pdf_document.close()
            return page_number + 1, True  # Classify as True (1)

        # Resize first to reduce processing time
        resized_image = resize_image(img)

        # Preprocess the resized image
        preprocessed_image = preprocess_image(resized_image)

        # Encode with size control
        base64_image = encode_image_with_size_control(preprocessed_image, MAX_FILE_SIZE)

        is_photo = get_page_description(base64_image)

        logging.info(f"Processed page {page_number+1} of {pdf_path}")
        pdf_document.close()
        return page_number + 1, is_photo
    except Exception as e:
        logging.warning(
            f"Error processing page {page_number+1} of {pdf_path}: {str(e)}"
        )
        return page_number + 1, False


def process_pdf(pdf_path):
    """Process a PDF file page by page in parallel"""
    results = {}

    pdf_document = fitz.open(pdf_path)
    page_count = len(pdf_document)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_page, pdf_path, page_number) for page_number in range(page_count)]
        for future in as_completed(futures):
            page_number, is_photo = future.result()
            results[page_number] = is_photo

    pdf_document.close()
    return results

def save_summaries_to_json(model_predictions, pdf_path):
    results = []
    for page_number, prediction in model_predictions.items():
        prediction_str = str(prediction).lower()
        is_photo = "true" in prediction_str

        result = {
            "sentence": "photo" if is_photo else "non-photo",
            "filename": os.path.basename(pdf_path),
            "start_page": page_number,
            "end_page": page_number,
        }
        results.append(result)

    return {"files": results}


if __name__ == "__main__":

    input_directory = "../data/input"
    output_path = "../data/output"
    output_data = []

    custom_template = ""

    try:
        for entry in os.listdir(input_directory):
            entry_path = os.path.join(input_directory, entry)

            if os.path.isfile(entry_path) and entry.endswith(".pdf"):
                model_predictions =  process_pdf(entry_path)
                output_data.append(save_summaries_to_json(model_predictions, entry))

            elif os.path.isdir(entry_path):
                # Process directory containing JSON files
                for filename in os.listdir(entry_path):
                    if filename.endswith(".pdf"):
                        input_file_path = os.path.join(entry_path, filename)

                        model_predictions = process_pdf(input_file_path)
                        output_data.append(
                            save_summaries_to_json(model_predictions, filename)
                        )

        # Convert the output data to JSON string
        with open(output_path, "w") as output_file:
            json.dump(output_data, output_file, indent=4)

    except Exception as e:
        logger.error(f"Error processing JSON: {str(e)}")
        print(json.dumps({"success": False, "message": "Failed to process JSON"}))
        sys.exit(1)
