import os
import json
import fitz  # PyMuPDF
import azure
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import time
import logging
import sys
import easyocr
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from PIL import Image


def getcreds():
    user = os.getenv("CREDS_USER")
    password = os.getenv("CREDS_PASSWORD")

    if not user or not password:
        raise ValueError("Credentials not set in environment variables")

    return user, password


class DocClient:
    def __init__(self, endpoint, key, header_ratio=None, footer_ratio=None):
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
        self.easyocr_reader = easyocr.Reader(
            ["en"], gpu=False, detect_network="craft", recog_network="english_g2"
        )
        self.header_ratio = header_ratio
        self.footer_ratio = footer_ratio
        self.max_image_size = (
            1000,
            1000,
        )  # Maximum width and height for resized images

    def close(self):
        self.client.close()

    def extract_content_azure(self, result):
        contents = {}
        for read_result in result.analyze_result.read_results:
            lines = read_result.lines
            lines.sort(key=lambda line: line.bounding_box[1])

            page_height = max(line.bounding_box[7] for line in lines)
            header_height = (
                page_height * self.header_ratio if self.header_ratio is not None else 0
            )
            footer_height = (
                page_height * self.footer_ratio if self.footer_ratio is not None else 0
            )

            page_content = []
            for line in lines:
                if (
                    header_height
                    <= line.bounding_box[1]
                    <= (page_height - footer_height)
                ):
                    page_content.append(" ".join([word.text for word in line.words]))
            contents[f"page_{read_result.page}"] = "\n".join(page_content)
        return contents

    def advanced_preprocess_image(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        binary = cv2.adaptiveThreshold(
            contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)

        eroded = cv2.erode(dilated, kernel, iterations=1)

        return eroded

    def process_with_easyocr(self, image_data):
        result = self.easyocr_reader.readtext(image_data)
        sorted_result = sorted(
            result, key=lambda x: x[0][0][1]
        )  # Sort by top-left y-coordinate

        if not sorted_result:
            return ""

        page_height = max(max(box[0][1], box[2][1]) for box, _, _ in sorted_result)

        if self.header_ratio is not None or self.footer_ratio is not None:
            header_height = (
                page_height * self.header_ratio if self.header_ratio is not None else 0
            )
            footer_start = (
                page_height * (1 - self.footer_ratio)
                if self.footer_ratio is not None
                else page_height
            )

            filtered_result = []
            for box, text, conf in sorted_result:
                box_top = min(box[0][1], box[1][1])
                box_bottom = max(box[2][1], box[3][1])

                # Check if the box is at least partially within the valid area
                if box_bottom > header_height and box_top < footer_start:
                    filtered_result.append((box, text, conf))
        else:
            filtered_result = sorted_result

        return "\n".join([text for _, text, _ in filtered_result])

    def reduce_pdf_size(self, page):
        """
        Reduce the size of a PDF page by converting it to an image and resizing it.
        """
        pix = page.get_pixmap(dpi=150)  # Lower DPI for smaller file size
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.thumbnail(self.max_image_size, Image.LANCZOS)

        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG", optimize=True, quality=85)
        img_byte_arr.seek(0)

        return img_byte_arr

    def process_page(self, page, page_num):
        try:
            # Reduce PDF size before processing
            img_byte_arr = self.reduce_pdf_size(page)

            # Try Azure OCR first
            try:
                ocr_result = self.client.read_in_stream(img_byte_arr, raw=True)
                operation_id = ocr_result.headers["Operation-Location"].split("/")[-1]
                while True:
                    result = self.client.get_read_result(operation_id)
                    if result.status.lower() not in ["notstarted", "running"]:
                        break
                    time.sleep(1)
                if result.status.lower() == "failed":
                    raise azure.core.exceptions.HttpResponseError("Azure OCR failed")
                page_results = self.extract_content_azure(result)
            except Exception as e:
                logging.warning(
                    f"Azure OCR failed for page {page_num}, falling back to EasyOCR: {str(e)}"
                )
                # Fallback to EasyOCR
                img = np.frombuffer(img_byte_arr.getvalue(), dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                processed_img = self.advanced_preprocess_image(img)
                page_content = self.process_with_easyocr(processed_img)
                page_results = {f"page_{page_num}": page_content}

            return page_results
        except Exception as e:
            logging.error(f"Error processing page {page_num}: {str(e)}")
            return {f"page_{page_num}": ""}

    def pdf2df(self, pdf_path):
        doc = fitz.open(pdf_path)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_page = {
                executor.submit(self.process_page, page, i + 1): i + 1
                for i, page in enumerate(doc)
            }
            all_pages_content = []
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_results = future.result()
                    all_pages_content.append(page_results)
                except Exception as e:
                    logging.error(
                        f"Error processing page {page_num} of file {pdf_path}: {str(e)}"
                    )

        all_pages_content.sort(key=lambda x: int(list(x.keys())[0].split("_")[1]))
        return all_pages_content

    def image2df(self, image_path):
        all_pages_content = []
        try:
            with open(image_path, "rb") as img_file:
                img_byte_arr = img_file.read()

                # Try Azure OCR first
                try:
                    ocr_result = self.client.read_in_stream(
                        BytesIO(img_byte_arr), raw=True
                    )
                    operation_id = ocr_result.headers["Operation-Location"].split("/")[
                        -1
                    ]
                    while True:
                        result = self.client.get_read_result(operation_id)
                        if result.status.lower() not in ["notstarted", "running"]:
                            break
                        time.sleep(1)
                    if result.status.lower() == "failed":
                        raise azure.core.exceptions.HttpResponseError(
                            "Azure OCR failed"
                        )
                    page_results = self.extract_content_azure(result)
                except Exception as e:
                    logging.warning(
                        f"Azure OCR failed for image, falling back to EasyOCR: {str(e)}"
                    )
                    # Fallback to EasyOCR
                    img = cv2.imdecode(
                        np.frombuffer(img_byte_arr, np.uint8), cv2.IMREAD_COLOR
                    )
                    processed_img = self.advanced_preprocess_image(img)
                    page_content = self.process_with_easyocr(processed_img)
                    page_results = {"page_1": page_content}

                all_pages_content.append(page_results)
        except Exception as e:
            logging.error(f"Error processing image file {image_path}: {str(e)}")
        return all_pages_content

    def process(self, file_path):
        if file_path.lower().endswith(".pdf"):
            return self.pdf2df(file_path)
        elif file_path.lower().endswith((".jpeg", ".jpg", ".png")):
            return self.image2df(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")


def reformat_json_structure(data, filename):
    new_messages = []
    for page_data in data:
        for key, content in page_data.items():
            new_messages.append(
                {
                    "page_content": content,
                    "page_number": 0,  # Placeholder value, will be updated later
                    "filename": filename,
                }
            )
    return {"files": new_messages}


def update_page_numbers(data):
    messages = data["files"]
    for i, message in enumerate(messages, start=1):
        message["page_number"] = i
    return data


if __name__ == "__main__":
    start_time = time.time()

    logger = logging.getLogger()
    azurelogger = logging.getLogger("azure")
    logger.setLevel(logging.INFO)
    azurelogger.setLevel(logging.ERROR)

    file_path = "../data/input/spinoza/2014-10-13 Unlawful Arrest SFPD246.pdf"
    output_dir = "../data/output/spinoza"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename for the output JSON file
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + "_output.json"
    output_path = os.path.join(output_dir, output_filename)

    header_ratio = 0.01
    footer_ratio = 0.01

    if not file_path.lower().endswith((".pdf", ".jpeg", ".jpg", ".png")):
        logger.error(f"File {file_path} is not a supported format")
        sys.exit(1)

    filename = os.path.basename(file_path)

    endpoint, key = getcreds()
    client = DocClient(endpoint, key, header_ratio, footer_ratio)

    results = client.process(file_path)
    formatted_results = reformat_json_structure(results, filename)
    updated_results = update_page_numbers(formatted_results)

    with open(output_path, "w") as output_file:
        json.dump(updated_results, output_file, indent=4)

    client.close()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
