import pytesseract
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
import json
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import uuid
import re
import PyPDF2
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_and_split(json_path):
    """Loads OCR text from JSON and splits it into chunks that approximately span 2 pages"""
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".messages[]",
        content_key="page_content",
    )
    data = loader.load()
    return data


template = """
You are an AI assistant tasked with generating a detailed timeline of events based on the provided police report. Please extract relevant information from the report to create a chronological timeline.

Format the timeline as follows:
- Use bullet points for each event
- Include the date and time (if available) for each event
- Provide a brief description of each event
- If a specific time is not mentioned, use relative terms like "before", "after", "later", etc.
- If the location of an event is mentioned, include it in the description

Previous Page Ending:
{previous_page_ending}

Current Page:
{current_page}

Next Page Beginning:
{next_page_beginning}

Timeline of Events:
"""


def generate_timeline(docs, query, window_size=100):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    prompt_response = ChatPromptTemplate.from_template(template)
    response_chain = prompt_response | llm | StrOutputParser()
    responses = []
    page_numbers = []

    for i in range(len(docs)):
        current_page = docs[i].page_content.replace("\n", " ")

        previous_page_ending = (
            docs[i - 1].page_content.replace("\n", " ")[-window_size:] if i > 0 else ""
        )
        next_page_beginning = (
            docs[i + 1].page_content.replace("\n", " ")[:window_size]
            if i < len(docs) - 1
            else ""
        )
        page_number = docs[i].metadata.get("page_number")

        if current_page:
            response = response_chain.invoke(
                {
                    "question": query,
                    "previous_page_ending": previous_page_ending,
                    "current_page": current_page,
                    "next_page_beginning": next_page_beginning,
                }
            )
            responses.append(response)
        else:
            responses.append("")

        if page_number is not None:
            page_numbers.append(page_number)

    concatenated_responses = "\n\n".join(responses)
    return concatenated_responses, page_numbers


def process_files_and_generate_pdfs(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            docs = load_and_split(json_path)
            query = "Generate a timeline of events based on the police report."
            concatenated_responses, _ = generate_timeline(docs, query)

            output_pdf = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.pdf"
            )
            c = canvas.Canvas(output_pdf, pagesize=letter)
            y = 750

            for line in concatenated_responses.split("\n"):
                c.drawString(50, y, line)
                y -= 20
                if y < 50:
                    c.showPage()
                    y = 750

            c.save()


if __name__ == "__main__":
    input_directory = "../data/output"
    output_directory = "../data/output"
    process_files_and_generate_pdfs(input_directory, output_directory)
