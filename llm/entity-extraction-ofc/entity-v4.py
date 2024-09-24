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
    "Officer Name", "Officer Role", "Officer Context", "page_number", "fn", 
    "Prompt Template for Hyde", "Prompt Template for Model", 
    "Temperature", "token_count", "file_type", "model"
]

DEFAULT_VALUES = {
    "Officer Name": "",
    "Officer Role": "",
    "Officer Context": "",
    "page_number": [],
    "fn": "",
    "Prompt Template for Hyde": "",
    "Prompt Template for Model": "",
    "Temperature": 0.0,
    "token_count": 0,
    "file_type": "",
    "model": ""
}


template = """
        <task_description>
        As an AI assistant, meticulously analyze criminal justice documents and extract information about law enforcement personnel, being careful to distinguish between officers and other individuals mentioned in the context.
        </task_description>

        <input>
        <documents>{docs}</documents>
        </input>

        <thinking_process>
        <officer_name>Identify potential law enforcement personnel that are referenced in the input documents. The documents are your only source of information. Do not return any additional information that is not contained within the input documents.</instruction>
        1. Search for name prefixes that strongly indicate law enforcement roles:
        - Officer prefixes: Ofc., Officer, Sgt., Sergeant, Lt., Lieutenant, Cpt., Captain, Cmdr., Commander, Sheriff, Dep., Deputy, Det., Detective, Insp., Inspector
        2. Look for these prefixes directly associated with names (e.g., "Sgt. X, "Officer Y", "Lt. Z")
        3. If a name doesn't have one of these prefixes, be very cautious about labeling them as law enforcement
        4. For individuals without these prefixes, look for explicit statements of their law enforcement role (e.g., "Officer X, a police officer with the department") * 
        5. Be aware that mentions of law enforcement actions don't necessarily mean the subject is an officer (e.g., "Person X was arrested by the police" does not mean Person X is an officer) 
        </officer_name

        <officer_context>
        Analyze the context of each potential law enforcement personnel.</instruction>
        1. What specific actions or responsibilities are attributed to this individual?
        2. Are these actions consistent with law enforcement duties?
        3. Is there any information that contradicts their potential status as law enforcement?
        4. For individuals without clear law enforcement prefixes, is there strong contextual evidence of their role?
        5. Be cautious of assuming someone is an officer just because they're mentioned in proximity to law enforcement activities
        </officer_context>


        <officer_role>
        Categorize individuals based on their role in the document
        1. For confirmed law enforcement:
        - Determine their specific role (e.g., Lead Detective, Patrol Officer, Crime Scene Technician)
        - If the role isn't explicitly stated, use context to make a best guess, but indicate uncertainty
        2. For other individuals:
        - Categorize as appropriate: Suspect, Witness, Victim, Civilian, etc.
        - Do not label these individuals as officers unless there's overwhelming evidence to support it
        3. If unsure about an individual's category, label them as "Unspecified Role" rather than assuming they're an officer
        </officer_role>
        </thinking_process>

    <verification_instructions>
        1. For each potential law enforcement entitiy, assess the level of certainty:
        - High Certainty: Clear law enforcement name, context, and role
        - Other: Ambiguous context or conflicting information
        
        2. Only include individuals as officers in the output if they have High Certainty
        3. All other individuals should be categorized based on their apparent role (e.g., witness, suspect, unknown)
    </verification_instructions>


    <output_format>
    Provide a JSON object with an "officers" key containing an array of officer objects. Each officer object should have "name", "context", and "role" properties.
    </output_format>
    """


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

def get_response_from_query(db, temperature, model, pages_to_concatenate=4):
    logger.info("Performing extraction..")
    if db is None:
        logger.warning("Database is None. Returning empty results.")
        return [], [], model

    docs = db

    try:
        if model == "mistralai/Mixtral-8x22B-Instruct-v0.1":
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
        
        for i in range(0, len(docs), pages_to_concatenate):
            concatenated_pages = ""
            current_page_numbers = []
            
            for j in range(pages_to_concatenate):
                if i + j < len(docs):
                    doc = docs[i + j]
                    concatenated_pages += doc.page_content + "\n\n"
                    page_number = doc.metadata.get("page_number")
                    if page_number is not None:
                        current_page_numbers.append(page_number)
            
            if concatenated_pages:
                try:
                    response = response_chain.invoke({"docs": concatenated_pages})
                    logger.info(f"Structured output: {response}")
                    if response is not None:
                        responses.append(response)
                    else:
                        logger.warning("Received None response from model. Skipping.")
                except Exception as e:
                    logger.error(f"Error processing concatenated pages: {e}")
            else:
                responses.append({"officers": []})
            
            page_numbers.extend(current_page_numbers)

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
        db, TEMPERATURE, model, pages_to_concatenate=2
    )

    if not officer_data:
        # If no officers found, create a row with default values
        default_row = {column: DEFAULT_VALUES[column] for column in REQUIRED_COLUMNS}
        default_row["page_number"] = page_numbers
        default_row["fn"] = file_name
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
                "Officer Name": (officer["name"]),
                "Officer Context": officer["context"],
                "Officer Role": officer["role"],
                "page_number": page_numbers,
                "fn": file_name,
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
    num_processes = 1
    
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
    model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

    logger.info("Starting processing...")
    process_query(input_path_transcripts, input_path_reports, output_path, model)
    logger.info("Processing completed.")