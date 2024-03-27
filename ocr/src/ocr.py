import os
import json
import pdf2image
import azure
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import time
import logging



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

    def pdf2df(self, pdf_path, json_file):
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
                    operation_id = ocr_result.headers["Operation-Location"].split("/")[
                        -1
                    ]

                    while True:
                        result = self.client.get_read_result(operation_id)
                        if result.status.lower() not in ["notstarted", "running"]:
                            break
                        time.sleep(1)

                    if result.status.lower() == "failed":
                        logging.error(f"OCR failed for page {i+1} of file {pdf_path}")
                        continue

                    page_results = self.extract_content(result)

                    with open(json_file, "a") as f:
                        json.dump(page_results, f)
                        f.write("\n")

                except azure.core.exceptions.HttpResponseError as e:
                    logging.error(
                        f"Error processing page {i+1} of file {pdf_path}: {e}"
                    )
                    continue

    def process(self, pdf_path):
        outname = os.path.basename(pdf_path).replace(".pdf", "")
        outstring = os.path.join(
            "../data/output", "{}.json".format(outname)
        )
        outpath = os.path.abspath(outstring)

        if os.path.exists(outpath):
            logging.info(f"skipping {outpath}, file already exists")
            return outpath

        logging.info(f"sending document {outname}")

        with open(outpath, "w") as f:
            f.write('{ "messages": {\n')

        self.pdf2df(pdf_path, outpath)

        with open(outpath, "a") as f:
            f.write("\n}}")

        logging.info(f"finished writing to {outpath}")
        return outpath


def update_page_keys_in_json(json_file):
    corrected_messages = {}

    with open(json_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines, start=0):
            if i == 0:
                continue
            try:
                content = json.loads(line.strip())
                corrected_key = f"page_{i}"
                corrected_messages[corrected_key] = content[f"page_1"]
            except json.JSONDecodeError:
                continue

    with open(json_file, "w") as f:
        json.dump({"messages": corrected_messages}, f, indent=4)


def reformat_json_structure(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    new_messages = []
    for key, content in data["messages"].items():
        page_num = int(key.split('_')[1])  
        new_messages.append({
            "page_content": content,
            "page_number": page_num
        })

    new_data = {"messages": new_messages}

    with open(json_file, "w") as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":
    logger = logging.getLogger()
    azurelogger = logging.getLogger("azure")
    logger.setLevel(logging.INFO)
    azurelogger.setLevel(logging.ERROR)

    doc_directory = "../data/input"

    endpoint, key = getcreds()
    client = DocClient(endpoint, key)

    files = [
    f
    for f in os.listdir(doc_directory)
    if os.path.isfile(os.path.join(doc_directory, f)) and f.lower().endswith('.pdf')
    ]
    logging.info(f"starting to process {len(files)} files")
    for file in files:
        json_file_path = client.process(os.path.join(doc_directory, file))
        update_page_keys_in_json(json_file_path)
        reformat_json_structure(json_file_path)

    client.close()