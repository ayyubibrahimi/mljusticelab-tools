# summary 1

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
import os
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from dotenv import find_dotenv, load_dotenv
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json

load_dotenv(find_dotenv())

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# experiment with different spacy models and bert models

nlp = spacy.load("en_core_web_sm")

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)


def augment_named_entities(text):
    doc = nlp(text)
    ner_results = ner_pipeline(text)

    entity_map = {}
    for entity in ner_results:
        start, end, label = entity["start"], entity["end"], entity["entity_group"]
        entity_map[(start, end)] = label

    augmented_text = ""
    prev_end = 0

    for ent in doc.ents:
        if (
            ent.label_ == "DATE"
            or ent.label_ == "PERSON"
            or ent.label_ == "EVENT"
            or ent.label_ == "FAC"
            or ent.label_ == "ORG"
            or ent.label_ == "LAW"
            or ent.label_ == "GPE"
            or ent.label_ == "PRODUCT"
            or ent.label_ == "NORP"
            or ent.label_ == "WORK_OF_ART"
            or ent.label_ == "TIME"
            or ent.label_ == "LOC"
        ):
            augmented_text += text[prev_end : ent.start_char]
            augmented_text += f"{ent.text}: {ent.label_}"
            prev_end = ent.end_char
        elif (ent.start_char, ent.end_char) in entity_map:
            label = entity_map[(ent.start_char, ent.end_char)]
            augmented_text += text[prev_end : ent.start_char]
            augmented_text += f"{ent.text}: {label}"
            prev_end = ent.end_char

    augmented_text += text[prev_end:]
    return augmented_text


def load_and_split(json_path):
    """Loads OCR text from JSON and splits it into chunks that approximately span 2 pages"""
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".messages[]",
        content_key="page_content",
    )
    data = loader.load()

    # Augment named entities in each page's content
    for doc in data:
        doc.page_content = augment_named_entities(doc.page_content)

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


def generate_timeline(docs, query, window_size=500):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    prompt_response = ChatPromptTemplate.from_template(template)
    response_chain = prompt_response | llm | StrOutputParser()
    output = []

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
        page_number = docs[i].metadata.get("seq_num")

        response = {
            "page_content": "",
            "page_number": page_number,
            "page_numbers": [page_number],
        }
        if current_page:
            processed_content = response_chain.invoke(
                {
                    "question": query,
                    "previous_page_ending": previous_page_ending,
                    "current_page": current_page,
                    "next_page_beginning": next_page_beginning,
                }
            )
            response["page_content"] = processed_content
        output.append(response)

    return output


def write_json_output(output_data, output_file_path):
    with open(output_file_path, "w") as file:
        json.dump(output_data, file, indent=4)


def create_pdf(output_data, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    text_object = c.beginText(40, 750)
    text_object.setFont("Helvetica", 12)

    for item in output_data:
        page_number = item.get("page_number")
        page_content = item.get("page_content")

        text_object.textLine(f"Page Number: {page_number}")
        text_object.textLines(page_content)
        text_object.textLine("\n")

        if text_object.getY() < 100:
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText(40, 750)
            text_object.setFont("Helvetica", 12)

    c.drawText(text_object)
    c.save()


combine_template = """
You are an AI assistant tasked with combining two summaries of a police report into a single, coherent summary. The summaries may contain overlapping information, so please consolidate and organize the information chronologically.

Summary 1:
{summary1}

Summary 2:
{summary2}

Combined Summary:
"""


def combine_summaries(summary1, summary2, query):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    prompt_response = ChatPromptTemplate.from_template(combine_template)
    response_chain = prompt_response | llm | StrOutputParser()

    processed_content = response_chain.invoke(
        {"summary1": summary1["page_content"], "summary2": summary2["page_content"]}
    )

    page_numbers = summary1.get("page_numbers", []) + summary2.get("page_numbers", [])
    new_summary = {"page_content": processed_content, "page_numbers": page_numbers}
    return new_summary


def recursive_summarize(summaries, query):
    if len(summaries) == 1:
        return summaries[0]
    else:
        mid = len(summaries) // 2
        left_summaries = summaries[:mid]
        right_summaries = summaries[mid:]

        left_summary = recursive_summarize(left_summaries, query)
        right_summary = recursive_summarize(right_summaries, query)

        combined_summary = combine_summaries(left_summary, right_summary, query)
        return combined_summary


def process_files_and_generate_outputs(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            docs = load_and_split(json_path)
            query = "Generate a timeline of events based on the police report."
            output_data = generate_timeline(docs, query)

            final_summary = recursive_summarize(output_data, query)

            output_json_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}_final_summary.json"
            )
            write_json_output([final_summary], output_json_path)

            output_pdf_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}_final_summary.pdf"
            )
            create_pdf([final_summary], output_pdf_path)


if __name__ == "__main__":
    input_directory = "../data/output-ocr"
    output_directory = "../data/output-llm"
    process_files_and_generate_outputs(input_directory, output_directory)
