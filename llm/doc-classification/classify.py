import os
import json
from io import BytesIO
import base64
import logging
from dotenv import find_dotenv, load_dotenv
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path
from PIL import Image
from langchain_anthropic import ChatAnthropic
from langchain.schema.messages import HumanMessage

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_IMAGE_SIZE = 1 * 1024 * 1024  # 1 MB

MAX_IMAGE_PIXELS = 89000000  # 89 million pixels


def process_pdf(pdf_path):
    """Process a PDF file page by page and determine the type of each page"""
    output_data = {}
    with open(pdf_path, "rb") as file:
        reader = PdfFileReader(file)
        page_count = reader.getNumPages()

        for page_number in range(page_count):
            try:
                images = convert_from_path(
                    pdf_path, first_page=page_number + 1, last_page=page_number + 1
                )
                image = images[0]

                if image.width * image.height > MAX_IMAGE_PIXELS:
                    page_type = "picture"
                    logging.info(
                        f"Classified page {page_number+1} of {pdf_path} as 'picture' due to large image size"
                    )
                else:
                    base64_image = encode_image(image)

                    if len(base64_image) > MAX_IMAGE_SIZE:
                        page_type = "picture"
                    else:
                        page_type = determine_page_type(base64_image)

                output_data[f"page_{page_number+1}"] = {
                    "page_number": page_number + 1,
                    "type": page_type,
                }
                logging.info(
                    f"Processed page {page_number+1} of {pdf_path} - Type: {page_type}"
                )
            except Exception as e:
                logging.warning(
                    f"Skipped page {page_number+1} of {pdf_path} due to error: {str(e)}"
                )

    return output_data


def encode_image(image):
    """Encode an image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def determine_page_type(base64_image):
    """Determine the type of a page using Claude Haiku"""
    chat = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=1024)
    prompt = """Please analyze the given image and determine the type of page it represents. The possible types are: "narrative text", "form or table", "chart or graph", "picture", "cover page", "signature page", "appendix", "bibliography", "handwritten", "email", "blank page". If the page doesn't match any of the above types, please classify it as "other". Please provide the page type as a single word without any additional explanation."""
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            )
        ]
    )
    return msg.content.strip().lower()


def process_directory(directory):
    """Process all PDF files in a directory"""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            logging.info(f"Processing {pdf_path}")
            output_data = process_pdf(pdf_path)
            output_filename = f"{os.path.splitext(filename)[0]}.json"
            output_path = os.path.join("data/output", output_filename)
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            json_output = json.dumps(output_data)

            print(json_output, end="")

            logging.info(f"Processed {filename} and saved output to {output_path}")


directory = "../../ocr/data/input"
process_directory(directory)
