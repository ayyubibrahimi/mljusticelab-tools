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
As an AI assistant, your task is to extract key events, actions, and details from the provided police report excerpt to create a chronological timeline. Use your understanding of the context and the following guidelines to identify and organize the relevant information:

- Identify significant events, actions, or details that contribute to the overall narrative of the police report.
- For each identified event or action, provide a brief and informative description that captures its essence.
- Maintain a chronological order based on the sequence in which the events or actions occurred in the report.
- If there are any related sub-events or additional details for a main event, include them as sub-points under the main event.
- Avoid including minor or irrelevant details that do not significantly contribute to the overall timeline.
- If the text is poorly OCR'd or lacks sufficient information to identify an event or action, skip that particular piece of the report.

Given the context from the previous page ending, the current page, and the next page beginning, extract the key events and actions from the police report excerpt and organize them in a chronological timeline.

Previous Page Ending: {previous_page_ending}
Current Page: {current_page}
Next Page Beginning: {next_page_beginning}

Chronological Timeline:
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

    # with open("../data/output/general_timeline.json", "w") as file:
    #     json.dump(output, file, indent=2)

    return output


def generate_pdf(table_of_contents_text, output_directory):
    pdf_file_path = os.path.join(output_directory, "table_of_contents.pdf")
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    section_style = styles["Heading2"]
    subsection_style = styles["Heading3"]

    elements = []

    title = Paragraph("Table of Contents", title_style)
    elements.append(title)
    elements.append(Spacer(1, 24))

    lines = table_of_contents_text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue

        if line.startswith("-"):
            subsection_text = line[2:].strip()
            subsection_paragraph = Paragraph(subsection_text, subsection_style)
            elements.append(subsection_paragraph)
            elements.append(Spacer(1, 4))
        else:
            section_parts = line.split("(")
            section_title = section_parts[0].strip()
            page_range = section_parts[1].strip(")") if len(section_parts) > 1 else ""

            section_text = f"{section_title}"
            if page_range:
                section_text += f" <font size=8>({page_range})</font>"

            section_paragraph = Paragraph(section_text, section_style)
            elements.append(section_paragraph)

            # Add extra space between sections, except for the last section
            if i < len(lines) - 1 and not lines[i + 1].startswith("-"):
                elements.append(Spacer(1, 12))

    doc.build(elements)


toc_template = """
As an AI assistant, your task is to update the table of contents for the provided police report based on the current page summary and the existing table of contents. Please follow these guidelines:

1. Review the current page summary and identify if it belongs to an existing section in the table of contents or if it represents a new section.
2. If the current page summary belongs to an existing section:
   - Update the page range for that section to include the current page number.
   - If necessary, modify the section title or description to better represent the content of the current page.
3. If the current page summary represents a new section:
   - Add a new entry to the table of contents for this section.
   - Provide a clear and concise title and description that captures the main content or theme of the section.
   - Include the current page number as the starting page for this section.
4. If the current page summary contains information that is relevant to multiple sections:
   - Consider creating a new subsection within an existing section to capture the specific content.
   - Update the page ranges and descriptions of the affected sections and subsections accordingly.
5. Ensure that the table of contents remains organized in a logical order based on the chronology of events or the flow of information in the report.
6. Use clear and concise language for the section titles and descriptions, making them easily understandable.
7. Maintain a consistent formatting style for the table of contents.

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

        updated_toc = response_chain.invoke(
            {
                "current_page_summary": f"Page {page_number}: {page_content}",
                "existing_toc": table_of_contents,
            }
        )

        table_of_contents = updated_toc.strip()

    # generate_pdf(table_of_contents, output_directory)

    return table_of_contents


toc_update_template = """
As an AI assistant, your task is to update the table of contents (TOC) based on the provided page summaries to ensure it accurately represents the content of the police report. Please follow these guidelines:

1. Carefully review each page summary and compare it against the sections and subsections in the existing TOC.
2. Identify any significant events, topics, or details from the page summaries that are missing or underrepresented in the TOC.
3. For each page summary:
   - If the content belongs to an existing section or subsection, update the page range and description accordingly.
   - If the content introduces a new section or subsection, add it to the TOC with a clear title, description, and page range.
   - If the content spans multiple sections or subsections, consider creating new subsections or modifying existing ones to better capture the specific details.
4. Ensure that the updated TOC follows a logical order and structure based on the chronology of events and the flow of information.
5. Maintain a consistent formatting style and use clear, concise language for the titles and descriptions.
6. If any sections or subsections become redundant or irrelevant after the updates, consider merging or removing them to keep the TOC concise and informative.

Page Summaries:
{page_summaries}

Existing Table of Contents:
{table_of_contents}

Updated Table of Contents:
"""


def update_table_of_contents_iteratively(
    page_summaries, initial_toc, output_directory, batch_size=5
):
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

    prompt_response = ChatPromptTemplate.from_template(toc_update_template)
    response_chain = prompt_response | llm | StrOutputParser()

    updated_toc = initial_toc

    for i in range(0, len(page_summaries), batch_size):
        batch_summaries = page_summaries[i : i + batch_size]

        batch_summary_string = "\n".join(
            [
                f"Page {summary['page_number']}: {summary['page_content']}"
                for summary in batch_summaries
            ]
        )

        updated_toc_json = response_chain.invoke(
            {
                "page_summaries": batch_summary_string,
                "table_of_contents": updated_toc,
            }
        )

        # updated_toc = json.loads(updated_toc_json)

    print(updated_toc_json)

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

            initial_toc = generate_initial_table_of_contents(
                page_summaries, output_directory
            )
            print(f"Generated initial table of contents for {filename}: {initial_toc}")

            updated_toc = update_table_of_contents_iteratively(
                page_summaries, initial_toc, output_directory, batch_size=5
            )
            print(f"Updated table of contents for {filename}: {updated_toc}")
