import os
import json
from io import BytesIO
import base64
import logging
from dotenv import find_dotenv, load_dotenv
from PyPDF2 import PdfReader
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
    output_data = {"messages": []}
    page_types = []  # Store the page types

    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        page_count = len(reader.pages)
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

                output_data["messages"].append({
                    "page_number": page_number + 1,
                    "type": page_type,
                })
                page_types.append(page_type)  # Add the page type to the list
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
    prompt = """
    Please analyze the given image and determine if the page type falls under one of the following categories (return 1) or not (return 0). The specified types are:

    1. "narrative": Continuous text, such as a report, article, or book page. Characteristics:
    - Paragraphs of text
    - May include headings, subheadings, or page numbers
    - Structured format with speaker labels (e.g., "Q:", "A:")
    3. "form": Structured documents with fields, checkboxes, or tables for data entry. Characteristics:
    - Presence of input fields, checkboxes, or dropdown menus
    - Labels or instructions for each field
    - Distinctive layout with sections and borders
    4. "email": Electronic mail messages or email printouts. Characteristics:
    - Presence of email headers (e.g., "From:", "To:", "Subject:")
    - Indented or quoted text for replies or forwarded messages
    - Email signatures or disclaimers at the bottom

    Answer with a single digit "1" if the page falls under any of these categories and "0" if it does not.

    After making a decision, return a single digit with no other accompanying text. 
    """

    try:
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
        page_type = msg.content.strip().lower()
        return page_type
    except Exception as e:
        print(f"Error determining page type: {str(e)}")
        return "unknown"