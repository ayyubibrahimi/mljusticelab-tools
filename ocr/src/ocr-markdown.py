import os
import pdf2image
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from io import BytesIO
import logging

def getcreds():
    with open("../creds/creds_cv.txt", "r") as c:
        creds = c.readlines()
    return creds[0].strip(), creds[1].strip()

class DocClient:
    def __init__(self, endpoint, key):
        self.client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    def close(self):
        pass

    def pdf2txt(self, pdf_path, txt_file):
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

                    poller = self.client.begin_analyze_document("prebuilt-read", img_byte_arr)
                    result = poller.result()

                    with open(txt_file, "a") as f:
                        for page in result.pages:
                            for line in page.lines:
                                f.write(f"{line.content}\n")

                except Exception as e:
                    logging.error(f"Error processing page {i+1} of file {pdf_path}: {e}")
                    continue

    def process(self, pdf_path):
        outname = os.path.basename(pdf_path).replace(".pdf", "")
        outstring = os.path.join("../data/output", "{}.txt".format(outname))
        outpath = os.path.abspath(outstring)

        if os.path.exists(outpath):
            logging.info(f"skipping {outpath}, file already exists")
            return outpath

        logging.info(f"sending document {outname}")

        self.pdf2txt(pdf_path, outpath)

        logging.info(f"finished writing to {outpath}")
        return outpath

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
        txt_file_path = client.process(os.path.join(doc_directory, file))

    client.close()