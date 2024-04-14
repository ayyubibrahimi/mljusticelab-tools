import os
import json
from pdf2image import convert_from_path
from langchain_anthropic import ChatAnthropic
from langchain.schema.messages import HumanMessage
from dotenv import find_dotenv, load_dotenv
from io import BytesIO
import base64
import logging

load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_pdf_in_batches(pdf_path, batch_size=10):
    """Process a PDF file in batches and determine the type of each page"""
    output_data = {}
    pages = convert_from_path(pdf_path)
    page_count = len(pages)
    for start_page in range(0, page_count, batch_size):
        end_page = min(start_page + batch_size, page_count)
        batch_pages = pages[start_page:end_page]
        for page_number, page in enumerate(batch_pages, start=start_page + 1):
            base64_image = encode_image(page)
            page_type = determine_page_type(base64_image)
            output_data[f"page_{page_number}"] = {
                "page_number": page_number,
                "type": page_type
            }
            logging.info(f"Processed page {page_number} of {pdf_path} - Type: {page_type}")
    return output_data

def encode_image(image):
    """Encode an image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def determine_page_type(base64_image):
    """Determine the type of a page using Claude Haiku"""
    chat = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=1024)
    prompt = """
    Please analyze the given image and determine the type of page it represents.
    The possible types are: "narrative", "form", "chart", "graph", "picture", 
    "table", "cover page", "signature page", "appendix", "index", "bibliography", 
    "redacted", "handwritten notes", "typed notes", "email", "legal document", 
    "map", "diagram", "blank page". If the page doesn't match any of the above 
    types, please classify it as "other". Please provide the page type as a 
    single word without any additional explanation.
    """
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

def process_directory(directory, batch_size=10):
    """Process all PDF files in a directory"""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            logging.info(f"Processing {pdf_path}")
            output_data = process_pdf_in_batches(pdf_path, batch_size)
            output_filename = f"{os.path.splitext(filename)[0]}.json"
            output_path = os.path.join("../data/output", output_filename)
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            logging.info(f"Processed {filename} and saved output to {output_path}")

directory = "../../ocr/data/input"
batch_size = 10  # Adjust the batch size as needed
process_directory(directory, batch_size)