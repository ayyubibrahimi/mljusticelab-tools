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

def getcreds():
    with open("../creds/creds_cv.txt", "r") as c:
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
        with open(pdf_path, "rb") as file:
            pdf_data = file.read()

            num_pages = pdf2image.pdfinfo_from_bytes(pdf_data)["Pages"]
            page_results = []

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

                    page_content = self.extract_content(result)
                    page_results.append({
                        "page_content": page_content[f"page_1"],
                        "page_number": i + 1
                    })

                except PIL.Image.DecompressionBombError:
                    logging.warning(
                        f"Image size exceeds limit for page {i+1} of file {pdf_path}. Returning blank page content."
                    )
                    continue

                except azure.core.exceptions.HttpResponseError as e:
                    logging.error(
                        f"Error processing page {i+1} of file {pdf_path}: {e}"
                    )
                    continue

            return page_results

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
        corrected_key = f"page_{page_number}"
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

if __name__ == "__main__":
    logger = logging.getLogger()
    azurelogger = logging.getLogger("azure")
    logger.setLevel(logging.INFO)
    azurelogger.setLevel(logging.ERROR)

    input_directory = "data/input/pdfs"
    output_directory = "../data/output/ocr"
    endpoint, key = getcreds()
    client = DocClient(endpoint, key)

    for root, dirs, files in os.walk(input_directory):
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        if pdf_files:
            logging.info(f"Processing {len(pdf_files)} files in directory: {root}")

            for file in pdf_files:
                file_path = os.path.join(root, file)

                # Create corresponding output directory if it doesn't exist
                relative_path = os.path.relpath(root, input_directory)
                output_subdir = os.path.join(output_directory, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                json_file_path = client.process(file_path, output_subdir)
                update_page_keys_in_json(json_file_path)
                reformat_json_structure(json_file_path)

    client.close()