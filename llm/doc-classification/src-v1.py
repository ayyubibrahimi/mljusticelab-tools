import os
import json
from pdf2image import convert_from_path
from langchain_anthropic import ChatAnthropic
from langchain.schema.messages import HumanMessage
from dotenv import find_dotenv, load_dotenv
from io import BytesIO
import base64

load_dotenv(find_dotenv())

def process_pdf(pdf_path):
    """Process a PDF file and determine the type of each page"""
    pages = convert_from_path(pdf_path)
    output_data = {}

    for i, page in enumerate(pages, start=1):
        base64_image = encode_image(page)
        
        page_type = determine_page_type(base64_image)
        
        output_data[f"page_{i}"] = {
            "page_number": i,
            "type": page_type
        }

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
    The possible types are: "narrative", "form", "chart", "graph", "picture", "table", "cover page", "signature page", "appendix", "index", "bibliography", "redacted", "handwritten notes", "typed notes", "email", "legal document", "map", "diagram", "blank page".
    If the page doesn't match any of the above types, please classify it as "other".
    Please provide the page type as a single word without any additional explanation.
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

def process_directory(directory):
    """Process all PDF files in a directory"""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            output_data = process_pdf(pdf_path)
            
            output_filename = f"{os.path.splitext(filename)[0]}.json"
            output_path = os.path.join("../data/output", output_filename)
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Processed {filename} and saved output to {output_path}")

directory = "../../ocr/data/input"
process_directory(directory)