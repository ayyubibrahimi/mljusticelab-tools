from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from pdf2image import convert_from_path
import json
import os
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path):
    """Converts PDF file to images"""
    return convert_from_path(pdf_path)


def save_ocr_to_json(model, pdf_path, ocr_json_path, publish_date):
    """Performs OCR on a PDF and saves the result in a JSON format with page numbers"""
    doc = DocumentFile.from_pdf(pdf_path)
    result = model(doc)
    messages = []
    for page_num, page in enumerate(result.pages):
        page_text = ""
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    page_text += word.value + " "
        page_text = page_text.strip() + "\n"
        record = {
            "page_content": page_text.strip(),
            "metadata": {"page_number": page_num + 1},
        }
        messages.append(record)
    with open(ocr_json_path, "w") as file:
        json.dump({"messages": messages, "publish_date": publish_date}, file, indent=4)


def process_files(model, input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            json_filename = f"{os.path.splitext(filename)[0]}.json"
            ocr_json_path = os.path.join(output_dir, json_filename)
            publish_date = None  # Set the publish date if available, otherwise use None
            save_ocr_to_json(model, pdf_path, ocr_json_path, publish_date)
            logger.info(f"Processed file: {filename}")


if __name__ == "__main__":
    input_directory = "../data/input"
    output_directory = "../data/output"

    # Load the OCR model from DocTR
    model = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="vitstr_small",
        pretrained=True,
        assume_straight_pages=False,
        preserve_aspect_ratio=True,
        export_as_straight_boxes=False,
    )

    # Modify the binarization threshold and the box threshold for better text region detection
    # model.det_predictor.model.postprocessor.bin_thresh = 0.5
    # model.det_predictor.model.postprocessor.box_thresh = 0.2

    # Disable automatic grouping of lines into blocks
    model.resolve_blocks = False

    process_files(model, input_directory, output_directory)
