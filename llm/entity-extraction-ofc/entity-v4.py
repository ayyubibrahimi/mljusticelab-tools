import os
import logging
import pandas as pd
from langchain_community.document_loaders.json_loader import JSONLoader
from helper import PROMPT_TEMPLATE_HYDE
import spacy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import tiktoken
from multiprocessing import Pool, cpu_count
from langchain_together import ChatTogether
from langchain_mistralai import ChatMistralAI
import re
from collections import namedtuple
import json

Doc = namedtuple("Doc", ["page_content", "metadata"])

load_dotenv(find_dotenv())

nlp = spacy.load("en_core_web_lg")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration parameters
TEMPERATURE = 0.7
ITERATION_TIMES = 1
MAX_RETRIES = 10

REQUIRED_COLUMNS = [
    "Officer Name", "Officer Role", "Officer Context", "page_number", "fn", "Query", 
    "Prompt Template for Hyde", "Prompt Template for Model", 
    "Temperature", "token_count", "file_type", "model"
]

DEFAULT_VALUES = {
    "Officer Name": "",
    "Officer Role": "",
    "Officer Context": "",
    "page_number": [],
    "fn": "",
    "Query": "",
    "Prompt Template for Hyde": "",
    "Prompt Template for Model": "",
    "Temperature": 0.0,
    "token_count": 0,
    "file_type": "",
    "model": ""
}


template = """
    <task_description>
    As an AI assistant, meticulously analyze criminal justice documents and extract information about law enforcement personnel.
    </task_description>

    <input>
    <query>{question}</queryn>
    <documents>{docs}</documents>
    </input>

    <thinking_process>
    <step1>
        <instruction>Identify law enforcement personnel in the documents.</instruction>
        <thinking>
        Search for name prefixes such as ofcs., officers, sergeants, sgts., lieutenants, lts., captains, cpts., commanders, sheriffs, deputies, dtys., detectives, dets., inspectors, technicians, analysts, coroners.
        Are there any names associated with these prefixes?
        Are there any other contextual clues that suggest an individual is law enforcement personnel?
        </thinking>
    </step1>

    <step2>
        <instruction>Analyze the context of each identified law enforcement personnel.</instruction>
        <thinking>
        What is the surrounding information about this individual?
        Are there any details about their actions, responsibilities, or involvement in the case?
        Is there any ambiguity regarding their employment in law enforcement?
        </thinking>
    </step2>

    <step3>
        <instruction>Determine the role of each identified law enforcement personnel.</instruction>
        <thinking>
        Based on the context, what specific role does this individual seem to have?
        Are they described as a Lead Detective, Supervising Officer, Detective, Officer on Scene, Arresting Officer, Crime Lab Analyst, etc.?
        If the role is not explicitly stated, can it be inferred from their actions or responsibilities?
        </thinking>
    </step3>
    </thinking_processt>

    <verification_instructions>
    <instruction>Verify the certainty of law enforcement identification</instruction>
    <steps>
        1. For each potential law enforcement individual identified, assess the level of certainty:
        - Is there explicit mention of their law enforcement role?
        - Do they have a law enforcment title associated with their name, such as officer X or detective Y?
        
        2. Only include individuals in the output if there is high certainty they are law enforcement personnel.
        3. If no individuals can be confidently identified as law enforcement, return an empty array for the "officers" key.
    </steps>
    </verification_instructions>


    <confidence_threshold>
    <instruction>Implement a confidence scoring system</instruction>
    <details>
        Assign a confidence score (0-100) to each identified individual based on:
        - Explicitness of law enforcement role mention: +50 points
        - Clear description of law enforcement duties: +30 points
        - Consistent context supporting law enforcement role: +20 points
        - Ambiguous or conflicting information: -30 points

        Only include individuals with a confidence score of 80 or higher in the final output. 
        If there are no individuals with a confidence score of 90 or higher, return None. 
    </details>
    </confidence_threshold>

    <output_format>
    Provide a JSON object with an "officers" key containing an array of officer objects. Each officer object should have "name", "context", and "role" properties.
    </output_format>


    """


QUERY = [
    "In the transcript, identify individuals by their names along with their specific law enforcement titles, such as officer, sergeant, lieutenant, captain, commander, sheriff, deputy, detective, inspector, technician, analyst, det., sgt., lt., cpt., p.o., ofc., and coroner. Alongside each name and title, note the context of their mention. This includes the roles they played in key events, decisions they made, actions they took, their interactions with others, responsibilities in the case, and any significant outcomes or incidents they were involved in."
]


def clean_name(officer_name):
    return re.sub(
        r"(Detective|Officer|[Dd]et\.|[Ss]gt\.|[Ll]t\.|[Cc]pt\.|[Oo]fc\.|Deputy|Captain|[CcPpLl]|Sergeant|Lieutenant|Techn?i?c?i?a?n?|Investigator|^-|\d{1}\)|\w{1}\.)\.?\s+",
        "",
        officer_name,
    )


def extract_officer_data(text):
    officer_data = []

    normalized_text = re.sub(r"\s*-\s*", "", text)

    officer_sections = re.split(r"\n(?=Officer Name:)", normalized_text)

    for section in officer_sections:
        if not section.strip():
            continue

        officer_dict = {}

        name_match = re.search(
            r"Officer Name:\s*(.*?)\s*Officer Context:", section, re.DOTALL
        )
        context_match = re.search(
            r"Officer Context:\s*(.*?)\s*Officer Role:", section, re.DOTALL
        )
        role_match = re.search(r"Officer Role:\s*(.*)", section, re.DOTALL)

        if name_match and name_match.group(1):
            officer_dict["Officer Name"] = clean_name(name_match.group(1).strip())
        if context_match and context_match.group(1):
            officer_dict["Officer Context"] = context_match.group(1).strip()
        if role_match and role_match.group(1):
            officer_dict["Officer Role"] = role_match.group(1).strip()

        if officer_dict:
            officer_data.append(officer_dict)

    return officer_data


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["page_number"] = record.get("page_number")
    return metadata


def format_content(content):
    # Remove extra whitespace and empty lines
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = re.sub(r' +', ' ', content)
    
    # Split content into lines
    lines = content.split('\n')
    
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Check if the line is a header (e.g., all caps, or ends with a colon)
            if line.isupper() or line.endswith(':'):
                formatted_lines.append(f"\n{line}\n")
            else:
                formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def word_count(text):
    return len(text.split())
def preprocess_document(file_path):
    logger.info(f"Processing document: {file_path}")

    with open(file_path, "r") as file:
        file_content = file.read()
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            data = {}

    logger.info(f"Keys in parsed JSON data: {data.keys()}")

    if "messages" in data and data["messages"]:
        docs = []
        original_page_number = 1
        for message in data["messages"]:
            page_content = message.get("page_content", "")
            if word_count(page_content) >= 50:
                formatted_content = format_content(page_content)
                doc = Doc(
                    page_content=formatted_content,
                    metadata={"page_number": original_page_number},
                )
            else:
                doc = Doc(
                    page_content="No data to be processed on this page",
                    metadata={"page_number": original_page_number},
                )
            docs.append(doc)
            original_page_number += 1

        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_count = sum(len(enc.encode(doc.page_content)) for doc in docs)

        logger.info(f"Data loaded and formatted from document: {file_path}")
        return docs, token_count
    else:
        logger.warning(f"No valid data found in document: {file_path}")
        return [], 0
    

json_schema = {
    "title": "OfficerInformation",
    "description": "Information about law enforcement personnel extracted from documents.",
    "type": "object",
    "properties": {
        "officers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the law enforcement officer"
                    },
                    "context": {
                        "type": "string",
                        "description": "The context in which the officer is mentioned"
                    },
                    "role": {
                        "type": "string",
                        "description": "The role or title of the officer"
                    }
                },
                "required": ["name", "context", "role"]
            }
        }
    },
    "required": ["officers"]
}
import logging

logger = logging.getLogger(__name__)

def get_response_from_query(db, query, temperature, model):
    logger.info("Performing query...")
    if db is None:
        logger.warning("Database is None. Returning empty results.")
        return [], [], model

    docs = db

    try:
        if model == "claude-3-haiku-20240307":
            llm = ChatAnthropic(model_name=model, temperature=temperature)
        elif model == "claude-3-5-sonnet-20240620":
            llm = ChatAnthropic(model_name=model, temperature=temperature)
        elif model == "mistralai/Mixtral-8x22B-Instruct-v0.1":
            llm = ChatTogether(model_name=model, temperature=temperature)
        elif model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            llm = ChatTogether(model_name=model, temperature=temperature)
        elif model == "meta-llama/Llama-3-8b-chat-hf":
            llm = ChatTogether(model_name=model, temperature=temperature)
        elif model == "meta-llama/Llama-3-70b-chat-hf":
            llm = ChatTogether(model_name=model, temperature=temperature)
        elif model == "open-mistral-nemo":
            llm = ChatMistralAI(model_name=model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported model: {model}")

        structured_llm = llm.with_structured_output(json_schema)
        prompt_response = ChatPromptTemplate.from_template(template)
        response_chain = prompt_response | structured_llm

        responses = []
        page_numbers = []
        for doc in docs:
            try:
                page_content = doc.page_content.replace("\n", " ")
                page_number = doc.metadata.get("page_number")
                if page_content:
                    response = response_chain.invoke({"question": query, "docs": page_content})
                    logger.info(f"Structured output: {response}")
                    if response is not None:
                        responses.append(response)
                    else:
                        logger.warning("Received None response from model. Skipping.")
                else:
                    responses.append({"officers": []})
                if page_number is not None:
                    page_numbers.append(page_number)
            except Exception as e:
                logger.error(f"Error processing document: {e}")

        all_officers = []
        for response in responses:
            if response is not None:
                officers = response.get("officers", [])
                if officers is not None:
                    all_officers.extend(officers)
                else:
                    logger.warning("Received None for 'officers' in response. Skipping.")
            else:
                logger.warning("Encountered None response. Skipping.")

        return all_officers, page_numbers, model

    except Exception as e:
        logger.error(f"An error occurred in get_response_from_query: {e}")
        return [], [], model
    
def process_file(args):
    file_name, input_path, output_path, file_type, model = args
    csv_output_path = os.path.join(output_path, f"{file_name}.csv")
    if os.path.exists(csv_output_path):
        logger.info(f"CSV output for {file_name} already exists. Skipping...")
        return

    file_path = os.path.join(input_path, file_name)
    output_data = []

    db, token_count = preprocess_document(file_path)
    officer_data, page_numbers, model = get_response_from_query(
        db, QUERY, TEMPERATURE, model
    )

    if not officer_data:
        # If no officers found, create a row with default values
        default_row = {column: DEFAULT_VALUES[column] for column in REQUIRED_COLUMNS}
        default_row["page_number"] = page_numbers
        default_row["fn"] = file_name
        default_row["Query"] = QUERY
        default_row["Prompt Template for Hyde"] = PROMPT_TEMPLATE_HYDE
        default_row["Prompt Template for Model"] = template
        default_row["Temperature"] = TEMPERATURE
        default_row["token_count"] = token_count
        default_row["file_type"] = file_type
        default_row["model"] = model
        output_data.append(default_row)
    else:
        for officer in officer_data:
            item = {
                "Officer Name": clean_name(officer["name"]),
                "Officer Context": officer["context"],
                "Officer Role": officer["role"],
                "page_number": page_numbers,
                "fn": file_name,
                "Query": QUERY,
                "Prompt Template for Hyde": PROMPT_TEMPLATE_HYDE,
                "Prompt Template for Model": template,
                "Temperature": TEMPERATURE,
                "token_count": token_count,
                "file_type": file_type,
                "model": model
            }
            output_data.append(item)

    output_df = pd.DataFrame(output_data, columns=REQUIRED_COLUMNS)
    output_df.to_csv(csv_output_path, index=False)

def process_files(input_path, output_path, file_type, model):
    file_list = [f for f in os.listdir(input_path) if f.endswith(".json")]
    
    # Filter out files that have already been processed
    unprocessed_files = []
    for file_name in file_list:
        csv_output_path = os.path.join(output_path, f"{file_name}.csv")
        if not os.path.exists(csv_output_path):
            unprocessed_files.append(file_name)
    
    if not unprocessed_files:
        logger.info(f"All files in {input_path} have already been processed.")
        return

    args_list = [(file_name, input_path, output_path, file_type, model) for file_name in unprocessed_files]
    
    # Use half of the available CPU cores, but at least 1
    num_processes = max(1, cpu_count() // 10)
    
    logger.info(f"Processing {len(unprocessed_files)} files using {num_processes} processes.")
    
    with Pool(processes=num_processes) as pool:
        pool.map(process_file, args_list)

def process_query(input_path_transcripts, input_path_reports, output_path, model):
    try:
        logger.info("Processing transcripts...")
        process_files(input_path_transcripts, output_path, "transcript", model)
        
        logger.info("Processing reports...")
        process_files(input_path_reports, output_path, "report", model)
        
        logger.info("Processing completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    input_path_transcripts = "data/input/transcripts"
    input_path_reports = "data/input/reports"  # Note: This is the same as transcripts, is this intentional?
    output_path = "data/output"
    model = "open-mistral-nemo"

    logger.info("Starting processing...")
    process_query(input_path_transcripts, input_path_reports, output_path, model)
    logger.info("Processing completed.")