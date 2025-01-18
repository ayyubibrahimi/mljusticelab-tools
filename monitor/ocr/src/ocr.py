import os
import json
import pdf2image
import azure
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import time
import logging
from PIL import Image
import PIL
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import easyocr
import numpy as np
import cv2
from multiprocessing import cpu_count

def getcreds():
    with open("../creds/creds_cv.txt", "r") as c:
        creds = c.readlines()
    return creds[0].strip(), creds[1].strip()

class DocClient:
    def __init__(self, endpoint, key):
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False, detect_network='craft', recog_network='english_g2')
        
    def close(self):
        self.client.close()

    def extract_content(self, result):
        contents = {}
        for read_result in result.analyze_result.read_results:
            lines = read_result.lines
            lines.sort(key=lambda line: line.bounding_box[1])

            page_height = max(line.bounding_box[7] for line in lines)
            header_height = footer_height = page_height * 0.01

            page_content = []
            for line in lines:
                if header_height < line.bounding_box[1] < (page_height - footer_height):
                    page_content.append(" ".join([word.text for word in line.words]))

            contents[f"page_{read_result.page}"] = "\n".join(page_content)

        return contents

    def process_with_easyocr(self, image_data):
        # Convert image data to format suitable for EasyOCR
        img = np.array(Image.open(image_data))
        
        # Preprocess the image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Perform OCR using EasyOCR
        result = self.easyocr_reader.readtext(binary)
        
        # Sort results by vertical position
        sorted_result = sorted(result, key=lambda x: x[0][0][1])
        
        # Extract text
        return "\n".join([text for _, text, _ in sorted_result])

    def process_page(self, pdf_data, page_num):
        try:
            image = pdf2image.convert_from_bytes(
                pdf_data, dpi=500, first_page=page_num, last_page=page_num
            )[0]

            # Check image dimensions
            width, height = image.size
            total_pixels = width * height
            MAX_PIXELS = 25000000  # 25 megapixels threshold
            
            if total_pixels > MAX_PIXELS:
                logging.warning(f"Image size ({width}x{height}={total_pixels} pixels) exceeds threshold of {MAX_PIXELS} pixels for page {page_num}")
                return {
                    "page_content": "This input is an image, no content to parse",
                    "page_number": page_num
                }

            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            try:
                # Try Azure OCR first
                ocr_result = self.client.read_in_stream(img_byte_arr, raw=True)
                operation_id = ocr_result.headers["Operation-Location"].split("/")[-1]

                while True:
                    result = self.client.get_read_result(operation_id)
                    if result.status.lower() not in ["notstarted", "running"]:
                        break
                    time.sleep(1)

                if result.status.lower() == "failed":
                    raise azure.core.exceptions.HttpResponseError("Azure OCR failed")

                page_content = self.extract_content(result)
                
            except Exception as e:
                logging.warning(f"Azure OCR failed for page {page_num}, falling back to EasyOCR: {str(e)}")
                # Fallback to EasyOCR
                page_content = {f"page_1": self.process_with_easyocr(img_byte_arr)}

            return {
                "page_content": page_content[f"page_1"],
                "page_number": page_num
            }

        except PIL.Image.DecompressionBombError:
            logging.warning(f"Image size exceeds PIL limit for page {page_num}")
            return {
                "page_content": "This input is an image, no content to parse",
                "page_number": page_num
            }

        except Exception as e:
            logging.error(f"Error processing page {page_num}: {str(e)}")
            return None

    def pdf2df(self, pdf_path):
        with open(pdf_path, "rb") as file:
            pdf_data = file.read()

        num_pages = pdf2image.pdfinfo_from_bytes(pdf_data)["Pages"]
        
        # Use all available CPUs for processing
        max_workers = cpu_count()
        
        # Calculate chunk size based on number of pages and CPUs
        chunk_size = max(1, num_pages // max_workers)
        
        def process_chunk(start_page, end_page):
            chunk_results = []
            for page_num in range(start_page, min(end_page, num_pages + 1)):
                result = self.process_page(pdf_data, page_num)
                if result:
                    chunk_results.append(result)
            return chunk_results

        with ThreadPoolExecutor() as executor:
            future_to_chunk = {
                executor.submit(process_chunk, i, i + chunk_size): i
                for i in range(1, num_pages + 1, chunk_size)
            }
            
            all_results = []
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as exc:
                    logging.error(f'Chunk processing generated an exception: {exc}')

        all_results.sort(key=lambda x: x["page_number"])
        return all_results

    def process(self, pdf_path, output_dir):
        outname = os.path.basename(pdf_path).replace(".pdf", "")
        outstring = os.path.join(output_dir, "{}.json".format(outname))
        outpath = os.path.abspath(outstring)

        if os.path.exists(outpath):
            logging.info(f"skipping {outpath}, file already exists")
            return outpath

        logging.info(f"sending document {outname}")

        page_results = self.pdf2df(pdf_path)

        with open(outpath, "w") as f:
            json.dump({"messages": page_results}, f, indent=4)

        logging.info(f"finished writing to {outpath}")
        return outpath

def update_page_keys_in_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    corrected_messages = []
    for page in data["messages"]:
        page_number = page["page_number"]
        corrected_messages.append({
            "page_content": page["page_content"],
            "page_number": page_number
        })

    with open(json_file, "w") as f:
        json.dump({"messages": corrected_messages}, f, indent=4)

def reformat_json_structure(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    new_messages = []
    for page in data["messages"]:
        new_messages.append({
            "page_content": page["page_content"],
            "page_number": page["page_number"]
        })

    new_data = {"messages": new_messages}

    with open(json_file, "w") as f:
        json.dump(new_data, f, indent=4)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    azurelogger = logging.getLogger("azure")
    azurelogger.setLevel(logging.ERROR)

    # Read the processed_index.csv
    csv_path = '../../download/data/output/processed_index.csv'
    df = pd.read_csv(csv_path)
    
    endpoint, key = getcreds()
    client = DocClient(endpoint, key)

    start_time = time.time()
    total_files = len(df)
    processed_files = 0

    df.loc[:, "local_pdf_path"] = df.local_pdf_path.str.replace(r"^../(.+)", r"../../download/\1", regex=True)

    for idx, row in df.iterrows():
        pdf_path = row['local_pdf_path']
        
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF file not found at {pdf_path}")
            continue
            
        file_start_time = time.time()
        
        # Get the directory of the PDF file and use it as output directory
        output_dir = os.path.dirname(pdf_path)
        
        # Process the PDF
        json_file_path = client.process(pdf_path, output_dir)
        if json_file_path:
            update_page_keys_in_json(json_file_path)
            reformat_json_structure(json_file_path)
            processed_files += 1

            file_end_time = time.time()
            file_duration = file_end_time - file_start_time
            logger.info(f"Processed {os.path.basename(pdf_path)} in {file_duration:.2f} seconds")
            
        # Log progress periodically
        if (idx + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / (idx + 1)
            remaining_files = total_files - (idx + 1)
            estimated_remaining_time = remaining_files * avg_time
            logger.info(f"Progress: {idx + 1}/{total_files} files processed")
            logger.info(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes")

    client.close()

    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info(f"Total processing time: {total_duration:.2f} seconds")
    logger.info(f"Total files processed: {processed_files} out of {total_files}")
    if processed_files > 0:
        average_time = total_duration / processed_files
        logger.info(f"Average time per file: {average_time:.2f} seconds")

if __name__ == "__main__":
    main()