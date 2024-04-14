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
import sys

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
As an AI assistant, your task is to generate a concise and chronological summary of the events described in the provided police report excerpt. Use your understanding of the context and the following guidelines to create an accurate timeline:

- Identify and extract key events, such as incidents, arrests, witness statements, and evidence collection. 
- Determine the sequence of events based on the information provided, paying attention to temporal indicators like dates, times, and phrases such as "before", "after", and "during".
- Focus on the most critical actions and developments that contribute to the overall narrative.
- Use clear and concise language to describe each event in the timeline.
- Begin the summary by setting the scene, introducing the people, property, and other relevant information before describing the actions.
- Organize the events in true chronological order, based on when they actually occurred, rather than from the perspective of the writer or any individual involved.
- After narrating the main events, include additional facts such as evidence collected, pictures taken, witness statements, recovered property, and any other necessary details.
- Do not infer any details that are not explicitly stated. If the text is too poorly OCR'd to derive an event, ignore this piece of the report. 


Documents: {current_page}

Chronological Event Summary:
"""


def generate_timeline(docs, query, selected_model, window_size=500):
    if selected_model == "gpt-4-0125-preview":
        llm = ChatOpenAI(model_name="gpt-4-0125-preview")
    elif selected_model == "gpt-3.5-0125":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    elif selected_model == "claude-3-haiku-20240307":
        llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
    elif selected_model == "claude-3-sonnet-20240229":
        llm = ChatAnthropic(model_name="claude-3-sonnet-20240229")
    else:
        llm = ChatAnthropic(model_name="claude-3-opus-20240229")

    prompt_response = ChatPromptTemplate.from_template(generate_template)
    response_chain = prompt_response | llm | StrOutputParser()

    processed_content = response_chain.invoke(
                {
                    "question": query,
                    "current_page": docs,
                }
            )
    return processed_content


def write_json_output(output_data, summary, total_pages):
    try:
        # Append the current summary and page count to the output_data
        output_data["summaries"].append({
            "filename": filename,
            "summary": summary,
            "total_pages": total_pages
        })
    except Exception as e:
        logger.error(f"An error occurred while writing JSON output: {e}")
        error_message = {
            "success": False,
            "message": str(e)
        }
        print(json.dumps(error_message), end='')
        sys.exit(1)

if __name__ == "__main__":
    input_directory = sys.argv[1]
    selected_model = sys.argv[2]

    try:
        output_data = {"summaries": []}
        for filename in os.listdir(input_directory):
            if filename.endswith(".json"):
                json_path = os.path.join(input_directory, filename)
                with open(json_path, 'r') as f:
                    input_file = json.load(f)
                docs = load_and_split(json_path)
                query = "Generate a timeline of events based on the police report."
                page_summaries = generate_timeline(docs, query, selected_model)
                # Pass the basename of the file to the output function
                basename = os.path.basename(json_path)
                page_numbers = [message.get("page_number") for message in input_file.get("messages", [])]
                page_numbers = [num for num in page_numbers if num is not None]
                total_pages = max(page_numbers) if page_numbers else 0
                write_json_output(output_data, page_summaries, total_pages)

        # Print the output_data as a single JSON object
        print(json.dumps(output_data, ensure_ascii=False), end='')
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        print(json.dumps({"success": False, "message": str(e)}))
        sys.exit(1)