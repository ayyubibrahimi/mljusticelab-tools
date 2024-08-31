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
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

LARGE_IMAGE_THRESHOLD = 3.7 * 1024 * 1024  # 5 megapixels

TARGET_IMAGE_SIZE = (800, 800)  # Reduced from 1024x1024
MAX_FILE_SIZE = 2 * 1024 * 1024

INPUT_DIR = "../data/input/archive"
OUTPUT_DIR = "../data/output"


# chat = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash", google_api_key="AIzaSyBo4QCK-ReBOeIh3OZ0JkDtYRyN_313ly4"
# )

chat = ChatOpenAI(
    model="gpt-4o-mini",
    api_key="sk-wZIQ599Pbniu8a-_FCGwpuQ-z42darwuTEmsx_OsMiT3BlbkFJroHul_OnZR5OHUBAJASPM8xC56Or3G_Spbk9Ba-VUA",
    temperature=0
)

# chat = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            images = convert_from_path(
                pdf_path, first_page=page_number + 1, last_page=page_number + 1
            )
            
            # Check if DecompressionBombWarning was raised
            if any(issubclass(warn.category, Image.DecompressionBombWarning) for warn in w):
                logging.warning(f"DecompressionBombWarning for page {page_number+1} of {pdf_path}")
                return page_number + 1, True  # Classify as True (1)
        
        image = images[0]
        
        # Check image size
        if image.width * image.height > LARGE_IMAGE_THRESHOLD:
            logging.info(f"Large image detected on page {page_number+1} of {pdf_path}")
            return page_number + 1, True  # Classify as True (1)

        # Resize first to reduce processing time
        resized_image = resize_image(image)

        # Preprocess the resized image
        preprocessed_image = preprocess_image(resized_image)

        # Encode with size control
        base64_image = encode_image_with_size_control(
            preprocessed_image, MAX_FILE_SIZE
        )

        is_photo = get_page_description(base64_image)

        logging.info(f"Processed page {page_number+1} of {pdf_path}")
        return page_number + 1, is_photo
    except Exception as e:
        logging.warning(
            f"Error processing page {page_number+1} of {pdf_path}: {str(e)}"
        )
        return page_number + 1, False

def process_pdf(pdf_path):
    """Process a PDF file page by page in parallel"""
    results = {}

    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        page_count = len(reader.pages)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_page, pdf_path, page_number) for page_number in range(page_count)]
        for future in as_completed(futures):
            page_number, is_photo = future.result()
            results[page_number] = is_photo

    return results

def process_dataset(pdf_path, csv_path, output_path):
    """Process a single dataset and save results"""
    # Read original CSV
    df = pd.read_csv(csv_path)

    # Process PDF
    model_predictions = process_pdf(pdf_path)

    # Add model predictions to dataframe
    df["prediction"] = df["page_no"].map(model_predictions)

    # Post-processing: Convert predictions to binary values
    df["prediction"] = df["prediction"].apply(
        lambda x: 1 if isinstance(x, bool) and x or (isinstance(x, str) and "true" in x.lower()) else 0
    )

    # Sort the dataframe by page_no
    df = df.sort_values("page_no")

    # Save results
    df.to_csv(output_path, index=False)


def calculate_metrics(output_dir):
    """Calculate comprehensive metrics for each processed CSV and overall metrics"""
    results = []
    overall_metrics = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for filename in os.listdir(output_dir):
        if filename.endswith("_output.csv"):
            filepath = os.path.join(output_dir, filename)
            df = pd.read_csv(filepath)

            if "label" not in df.columns or "prediction" not in df.columns:
                logging.error(
                    f"Required columns not found in {filename}. Skipping this file."
                )
                continue

            TP = ((df["label"] == 1) & (df["prediction"] == 1)).sum()
            FP = ((df["label"] == 0) & (df["prediction"] == 1)).sum()
            TN = ((df["label"] == 0) & (df["prediction"] == 0)).sum()
            FN = ((df["label"] == 1) & (df["prediction"] == 0)).sum()

            total = TP + FP + TN + FN
            accuracy = (TP + TN) / total if total > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            results.append(
                {
                    "dataset": filename,
                    "TP": TP,
                    "FP": FP,
                    "TN": TN,
                    "FN": FN,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                }
            )

            # Accumulate overall metrics
            for key in ["TP", "FP", "TN", "FN"]:
                overall_metrics[key] += results[-1][key]

            # Log individual dataset metrics for debugging
            logging.info(f"Metrics for {filename}:")
            logging.info(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
            logging.info(
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}"
            )

    # Calculate overall metrics
    total_samples = sum(overall_metrics.values())
    overall_accuracy = (
        (overall_metrics["TP"] + overall_metrics["TN"]) / total_samples
        if total_samples > 0
        else 0
    )
    overall_precision = (
        overall_metrics["TP"] / (overall_metrics["TP"] + overall_metrics["FP"])
        if (overall_metrics["TP"] + overall_metrics["FP"]) > 0
        else 0
    )
    overall_recall = (
        overall_metrics["TP"] / (overall_metrics["TP"] + overall_metrics["FN"])
        if (overall_metrics["TP"] + overall_metrics["FN"]) > 0
        else 0
    )
    overall_f1 = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )

    # Add overall metrics to results
    results.append(
        {
            "dataset": "Overall",
            "TP": overall_metrics["TP"],
            "FP": overall_metrics["FP"],
            "TN": overall_metrics["TN"],
            "FN": overall_metrics["FN"],
            "accuracy": overall_accuracy,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
        }
    )

    # Create and save results.csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    return results_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)

        csv_file = os.path.splitext(pdf_file)[0] + ".csv"
        csv_path = os.path.join(INPUT_DIR, csv_file)

        if not os.path.exists(csv_path):
            logging.warning(f"Missing CSV file for {pdf_file}")
            continue

        output_filename = f"{os.path.splitext(pdf_file)[0]}_output.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Check if output file already exists
        if os.path.exists(output_path):
            logging.info(
                f"Output file {output_filename} already exists. Skipping processing."
            )
            continue

        process_dataset(pdf_path, csv_path, output_path)

        logging.info(f"Processed {pdf_file} and saved results to {output_filename}")

    # Calculate comprehensive metrics after processing all datasets
    results_df = calculate_metrics(OUTPUT_DIR)

    # Log overall metrics
    overall_metrics = results_df.iloc[-1]
    logging.info(f"Overall Metrics:")
    logging.info(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    logging.info(f"Precision: {overall_metrics['precision']:.4f}")
    logging.info(f"Recall: {overall_metrics['recall']:.4f}")
    logging.info(f"F1 Score: {overall_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
