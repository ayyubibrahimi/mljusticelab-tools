import os
import json
import pdf2image
import azure
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import time
import logging
import sys

def getcreds():
    with open("pages/api/creds_cv.txt", "r") as c:
        creds = c.readlines()
    return creds[0].strip(), creds[1].strip()

class DocClient:
    def __init__(self, endpoint, key):
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    def close(self):
        self.client.close()

    def extract_content(self, result):
        contents = {}
        for read_result in result.analyze_result.read_results:
            lines = read_result.lines
            lines.sort(key=lambda line: line.bounding_box[1])
            page_content = []
            for line in lines:
                page_content.append(" ".join([word.text for word in line.words]))
            contents[f"page_{read_result.page}"] = "\n".join(page_content)
        return contents

    def pdf2df(self, pdf_path):
        all_pages_content = []
        with open(pdf_path, "rb") as file:
            pdf_data = file.read()
            num_pages = pdf2image.pdfinfo_from_bytes(pdf_data)["Pages"]
            for i in range(num_pages):
                try:
                    image = pdf2image.convert_from_bytes(
                        pdf_data, dpi=500, first_page=i + 1, last_page=i + 1
                    )[0]
                    img_byte_arr = BytesIO()
                    image.save(img_byte_arr, format="PNG")
                    img_byte_arr.seek(0)
                    ocr_result = self.client.read_in_stream(img_byte_arr, raw=True)
                    operation_id = ocr_result.headers["Operation-Location"].split("/")[-1]
                    while True:
                        result = self.client.get_read_result(operation_id)
                        if result.status.lower() not in ["notstarted", "running"]:
                            break
                        time.sleep(1)
                    if result.status.lower() == "failed":
                        logging.error(f"OCR failed for page {i+1} of file {pdf_path}")
                        continue
                    page_results = self.extract_content(result)
                    all_pages_content.append(page_results)
                except azure.core.exceptions.HttpResponseError as e:
                    logging.error(f"Error processing page {i+1} of file {pdf_path}: {e}")
                    continue
        return all_pages_content

    def process(self, pdf_path):
        return self.pdf2df(pdf_path)

def reformat_json_structure(data):
    new_messages = []
    for page_data in data:
        for key, content in page_data.items():
            page_num = int(key.split('_')[1])
            new_messages.append({
                "page_content": content,
                "page_number": page_num
            })
    return {"messages": new_messages}

if __name__ == "__main__":
    logger = logging.getLogger()
    azurelogger = logging.getLogger("azure")
    logger.setLevel(logging.INFO)
    azurelogger.setLevel(logging.ERROR)

    if len(sys.argv) != 3:
        logger.error("Usage: python ocr.py <path_to_pdf_file> <path_to_output_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2]

    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"File {pdf_path} is not a PDF")
        sys.exit(1)

    endpoint, key = getcreds()
    client = DocClient(endpoint, key)
    print(client)

    results = client.process(pdf_path)
    formatted_results = reformat_json_structure(results)

    with open(output_path, "w") as output_file:
        json.dump(formatted_results, output_file, indent=4)

    client.close()