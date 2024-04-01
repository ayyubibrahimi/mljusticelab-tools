import os
import logging
from dotenv import find_dotenv, load_dotenv
import json
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from langchain_anthropic import ChatAnthropic
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# nltk.download('stopwords')

load_dotenv(find_dotenv())

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_lg")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="max"
)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")


def preprocess_text(text):
    # text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def augment_named_entities(text, threshold=0.9):
    text = preprocess_text(text)
    doc = nlp(text)
    ner_results = ner_pipeline(text)
    entity_map = {}
    for entity in ner_results:
        start, end, label, score = (
            entity["start"],
            entity["end"],
            entity["entity_group"],
            entity["score"],
        )
        if score >= threshold:
            entity_map[(start, end)] = label

    label_mapping = {
        "DATE": "Date",
        "PERSON": "Person",
        "EVENT": "Event",
        "FAC": "Facility",
        "ORG": "Organization",
        "LAW": "Law",
        "PRODUCT": "Product",
        "TIME": "Time",
        "LOC": "Location",
    }

    augmented_text = ""
    prev_end = 0
    for ent in doc.ents:
        if ent.label_ in label_mapping:
            label = label_mapping[ent.label_]
            augmented_text += text[prev_end : ent.start_char]
            augmented_text += f"({ent.text}: {label})"
            prev_end = ent.end_char
        elif (ent.start_char, ent.end_char) in entity_map:
            label = entity_map[(ent.start_char, ent.end_char)]
            augmented_text += text[prev_end : ent.start_char]
            augmented_text += f"({ent.text}: {label})"
            prev_end = ent.end_char

    augmented_text += text[prev_end:]
    # print(augmented_text)
    return augmented_text


def load_and_split(json_path):
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".messages[]",
        content_key="page_content",
    )
    data = loader.load()

    for doc in data:
        doc.page_content = augment_named_entities(doc.page_content)

    return data


generate_template = """
As an AI assistant, your task is to generate a concise summary of the key events and sections described in the provided police report excerpt. Use your understanding of the context and the following guidelines to create a clear and structured outline:

- Identify the main sections or topics covered in the excerpt, such as the incident description, arrests, witness statements, evidence collection, or any other distinct parts of the report.
- For each section or topic, provide a brief and informative title that captures the essence of the content.
- Underneath each section title, list the key events, actions, or details related to that section in a concise manner.
- Use bullet points or numbered lists to present the information in a clear and organized format.
- Maintain a logical flow and structure based on the order in which the sections and events appear in the report.
- If there are any subsections or subtopics within a main section, indent them appropriately to show the hierarchy.
- Avoid including minor or irrelevant details that do not significantly contribute to the overall understanding of the report.
- If the text is poorly OCR'd or lacks sufficient information to identify a section or event, skip that particular piece of the report.

Given the context from the previous page ending, the current page, and the next page beginning, generate a structured outline of the key sections and events in the police report excerpt.

Previous Page Ending: {previous_page_ending}
Current Page: {current_page}
Next Page Beginning: {next_page_beginning}

Structured Outline:
"""


def generate_timeline(docs, query, window_size=500):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
    prompt_response = ChatPromptTemplate.from_template(generate_template)
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

        response = {"page_content": "", "page_number": page_number}
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
        
    with open("../data/output/general_timeline.json", "w") as file:
        json.dump(output, file, indent=2)


    return output

def generate_pdf(toc_string, output_directory):
    pdf_file_path = os.path.join(output_directory, "table_of_contents.pdf")
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    section_style = styles["Heading2"]

    elements = []

    title = Paragraph("Table of Contents", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))

    toc_lines = toc_string.strip().split("\n\n")
    for line in toc_lines:
        if ":" in line:
            section_title, page_range = line.split(":", 1)
            section_text = f"{section_title.strip()} (Pages: {page_range.strip()})"
            section_paragraph = Paragraph(section_text, section_style)
            elements.append(section_paragraph)
            elements.append(Spacer(1, 12))

    doc.build(elements)


toc_template = """
As an AI assistant, your task is to update the table of contents for the provided police report based on the current page summary and the existing table of contents. Please follow these guidelines:

1. Review the current page summary and identify if it belongs to an existing section in the table of contents or if it represents a new section.
2. If the current page summary belongs to an existing section:
   - Update the page range for that section to include the current page number.
   - If necessary, modify the section description to accommodate the content of the current page.
3. If the current page summary represents a new section:
   - Add a new entry to the table of contents for this section.
   - Provide a brief and informative description that captures the main content or theme of the section.
   - Include the current page number as the starting page for this section.
4. Ensure that the table of contents remains organized in a logical order based on the chronology of events or the flow of information in the report.
5. Use clear and concise language for the section descriptions, making them easily understandable.
6. Preserve the existing structure and formatting of the table of contents.
7. Output the updated table of contents as a JSON object with the following structure:

    "section_title": "Section Title 1",
    "section_description": "Detailed description of section 1...",
    "page_range": "1-5",
    "subsections": 
        "subsection_title": "Subsection Title 1.1",
        "subsection_description": "Detailed description of subsection 1.1...",
        "page_range": "1-3"

        "subsection_title": "Subsection Title 1.2",
        "subsection_description": "Detailed description of subsection 1.2...",
        "page_range": "4-5"

    "section_title": "Section Title 2",
    "section_description": "Detailed description of section 2...",
    "page_range": "6-10"
        "subsection_title": "Subsection Title 2.1",
        "subsection_description": "Detailed description of subsection 2.1...",
        "page_range": "6-8"

        "subsection_title": "Subsection Title 2.2",
        "subsection_description": "Detailed description of subsection 2.2...",
        "page_range": "9-10"

Current Page Summary:
{current_page_summary}

Existing Table of Contents:
{existing_toc}

Updated Table of Contents:
"""

def generate_initial_table_of_contents(summaries, output_directory):
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

    prompt_response = ChatPromptTemplate.from_template(toc_template)
    response_chain = prompt_response | llm | StrOutputParser()

    table_of_contents = ""

    for summary in summaries:
        page_content = summary["page_content"]
        page_number = summary.get("page_number")

        updated_toc = response_chain.invoke({
            "current_page_summary": f"Page {page_number}: {page_content}",
            "existing_toc": table_of_contents
        })

        table_of_contents = updated_toc.strip()

    generate_pdf(table_of_contents, output_directory)

    return table_of_contents

toc_update_template = """
As an AI assistant, your task is to update the table of contents (TOC) based on the provided page summary to ensure it accurately represents the content of the police report. Please follow these guidelines:

1. Carefully review the page summary and compare it against the sections and subsections in the existing TOC.
2. Identify any significant events, topics, or details from the page summary that are missing or underrepresented in the TOC.
3. If the TOC needs to be updated:
   - Add new sections or subsections to cover the missing content.
   - Update the titles and descriptions of existing sections and subsections to better reflect the content.
   - Adjust the page ranges of sections and subsections to include the current page number.
4. Ensure that the updated TOC follows a logical order and structure based on the chronology of events and the flow of information.
5. Preserve the existing structure and formatting of the TOC.
6. Output the updated table of contents as a JSON object with the following structure:

    "section_title": "Section Title 1",
    "section_description": "Detailed description of section 1...",
    "page_range": "1-5",
    "subsections": 
        "subsection_title": "Subsection Title 1.1",
        "subsection_description": "Detailed description of subsection 1.1...",
        "page_range": "1-3"

        "subsection_title": "Subsection Title 1.2",
        "subsection_description": "Detailed description of subsection 1.2...",
        "page_range": "4-5"

    "section_title": "Section Title 2",
    "section_description": "Detailed description of section 2...",
    "page_range": "6-10"
        "subsection_title": "Subsection Title 2.1",
        "subsection_description": "Detailed description of subsection 2.1...",
        "page_range": "6-8"

        "subsection_title": "Subsection Title 2.2",
        "subsection_description": "Detailed description of subsection 2.2...",
        "page_range": "9-10"


Page Summaries:
{page_summaries}

Existing Table of Contents:
{table_of_contents}

Updated Table of Contents (JSON):
"""


def update_table_of_contents_iteratively(page_summaries, initial_toc, output_directory, batch_size=5):
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

    prompt_response = ChatPromptTemplate.from_template(toc_update_template)
    response_chain = prompt_response | llm | StrOutputParser()

    updated_toc = initial_toc

    for i in range(0, len(page_summaries), batch_size):
        batch_summaries = page_summaries[i:i+batch_size]

        batch_summary_string = "\n".join([
            f"Page {summary['page_number']}: {summary['page_content']}"
            for summary in batch_summaries
        ])

        updated_toc_json = response_chain.invoke({
            "page_summaries": batch_summary_string,
            "table_of_contents": updated_toc,
        })

        # updated_toc = json.loads(updated_toc_json)

    # Generate the table of contents PDF
    generate_pdf(updated_toc_json, output_directory)

    return updated_toc


def write_json_output(combined_summary, sentence_to_page, output_file_path):
    output_data = []
    for sentence, page_number in sentence_to_page.items():
        output_data.append({"sentence": sentence, "page_number": page_number})

    with open(output_file_path, "w") as file:
        json.dump(output_data, file, indent=4)

if __name__ == "__main__":
    input_directory = "../../ocr/data/output"
    output_directory = "../data/output"
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            json_path = os.path.join(input_directory, filename)
            docs = load_and_split(json_path)
            query = "Generate a timeline of events based on the police report."
            page_summaries = generate_timeline(docs, query)
            
            initial_toc = generate_initial_table_of_contents(page_summaries, output_directory)
            print(f"Generated initial table of contents for {filename}: {initial_toc}")

            updated_toc = update_table_of_contents_iteratively(page_summaries, initial_toc, output_directory, batch_size=5)
            print(f"Updated table of contents for {filename}: {updated_toc}")