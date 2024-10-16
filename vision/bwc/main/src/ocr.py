import os
import json
import pdf2image
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import time
import logging
from PIL import Image
import PIL
from concurrent.futures import ThreadPoolExecutor, as_completed

def getcreds():
    with open("../creds/creds_cv.txt", "r") as c:
        creds = c.readlines()
    return creds[0].strip(), creds[1].strip()

def create_client(endpoint, key):
    return ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

def extract_content(result):
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

def process_page(client, pdf_data, page_num):
    try:
        image = pdf2image.convert_from_bytes(
            pdf_data, dpi=500, first_page=page_num, last_page=page_num
        )[0]

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")

        img_byte_arr.seek(0)
        ocr_result = client.read_in_stream(img_byte_arr, raw=True)
        operation_id = ocr_result.headers["Operation-Location"].split("/")[-1]

        while True:
            result = client.get_read_result(operation_id)
            if result.status.lower() not in ["notstarted", "running"]:
                break
            time.sleep(1)

        if result.status.lower() == "failed":
            logging.error(f"OCR failed for page {page_num}")
            return None

        page_content = extract_content(result)
        return {
            "page_content": page_content[f"page_1"],
            "page_number": page_num
        }

    except PIL.Image.DecompressionBombError:
        logging.warning(f"Image size exceeds limit for page {page_num}. Returning blank page content.")
        return None

    except Exception as e:
        logging.error(f"Error processing page {page_num}: {e}")
        return None

def pdf2df(client, pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_data = file.read()

    num_pages = pdf2image.pdfinfo_from_bytes(pdf_data)["Pages"]
    
    with ThreadPoolExecutor() as executor:
        future_to_page = {executor.submit(process_page, client, pdf_data, i + 1): i + 1 for i in range(num_pages)}
        page_results = []
        
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                if result:
                    page_results.append(result)
            except Exception as exc:
                logging.error(f'Page {page_num} generated an exception: {exc}')

    page_results.sort(key=lambda x: x["page_number"])
    return page_results

def process_pdf(client, pdf_path, output_dir):
    outname = os.path.basename(pdf_path).replace(".pdf", "")
    outstring = os.path.join(output_dir, "{}.json".format(outname))
    outpath = os.path.abspath(outstring)

    if os.path.exists(outpath):
        logging.info(f"skipping {outpath}, file already exists")
        return outpath

    logging.info(f"sending document {outname}")

    page_results = pdf2df(client, pdf_path)

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