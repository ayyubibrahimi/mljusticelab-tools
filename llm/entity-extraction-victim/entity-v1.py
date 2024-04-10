import os
import logging
import pandas as pd
from langchain.chains import LLMChain
from langchain_community.document_loaders import JSONLoader
from langchain.vectorstores.faiss import FAISS

from helper import generate_hypothetical_embeddings, PROMPT_TEMPLATE_HYDE, sort_retrived_documents

import spacy

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np


import numpy as np
from langchain_anthropic import ChatAnthropic

import sys
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

nlp = spacy.load("en_core_web_lg")


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TEMPERATURE = 1
K = 20

template_1 ="""
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about victims.
  
    Query: {question}

    Documents: {docs}

    The response will contain:

    1) The name of the victim. Please prefix the name with "Victim's Name: ". 
       For example, "Victim's Name: John Smith".

    The full response should follow the format below, with no additional information or prefixes:

    Victim's Name: John Smith

    - Do not include any prefixes or additional information
    - Only derive responses from factual information found within the police reports.
    - If no victim is identified in the given context, respond with "Victim's Name: Not Specified".
    - If the age or race of the victim is not specified, omit those fields from the response.
    """

template_2 ="""
   As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about victims.
  
    Query: {question}

    Documents: {docs}

    The response will contain:

    1) The name of the victim. Please prefix the name with "Victim's Name: ". 
       For example, "Victim's Name: John Smith".

    2) The age of the victim, if available. Please prefix the age with "Victim Age: ".
       For example, "Victim Age: 25".

    3) The race of the victim, if available. Please prefix the race with "Victim Race: ".
       For example, "Victim Race: Caucasian".

    The full response should follow the format below, with no additional information or prefixes:

    Victim's Name: John Smith
    Victim Age: 25
    Victim Race: Caucasian

    - Do not include any prefixes or additional information
    - Only derive responses from factual information found within the police reports.
    - If no victim is identified in the given context, respond with "Victim's Name: Not Specified".
    - If the age or race of the victim is not specified, omit those fields from the response.
    """


QUERIES = [
    "In the transcript, identify individuals by their names along with their specific law enforcement titles, such as officer, sergeant, lieutenant, captain, commander, sheriff, deputy, detective, inspector, technician, analyst, det., sgt., lt., cpt., p.o., ofc., and coroner. Alongside each name and title, note the context of their mention. This includes the roles they played in key events, decisions they made, actions they took, their interactions with others, responsibilities in the case, and any significant outcomes or incidents they were involved in."
]


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["page_number"] = record.get("page_number")
    return metadata


def preprocess_document(file_path, embeddings):
    MAX_PAGES = 20
    logger.info(f"Processing document: {file_path}")
    loader = JSONLoader(file_path, jq_schema=".messages[]", content_key="page_content", metadata_func=metadata_func)
    text = loader.load()
    logger.info(f"Text loaded from document: {file_path}")

    page_entities = []
    for page in text:
        doc = nlp(page.page_content)
        entities = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities.append(ent.text)
        page_entities.append((page, len(entities)))

    page_entities.sort(key=lambda x: x[1], reverse=True)

    selected_pages = [page for page, _ in page_entities[:MAX_PAGES]]

    if selected_pages:
        db = FAISS.from_documents(selected_pages, embeddings)
    else:
        db = None

    return db

def get_response_from_query(db, query, temperature, k, template):
    logger.info("Performing query...")
    if db is None:
        return []
    doc_list = db.similarity_search_with_score(query, k=k)
    if not doc_list:
        return []
    docs = sort_retrived_documents(doc_list)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

    prompt_response = ChatPromptTemplate.from_template(template)
    response_chain = prompt_response | llm | StrOutputParser()
    officer_data = []
    for doc in docs:
        page_content = doc[0].page_content.replace('\n', ' ')
        page_number = doc[0].metadata.get('page_number')
        if page_content:
            response = response_chain.invoke({"question": query, "docs": page_content})
            officer_data.append({"response": response, "page_number": page_number})
            print(response)
        else:
            officer_data.append({"response": "", "page_number": page_number})
    return officer_data

def process_file(input_file_path, output_file_path, embeddings, query, template):
    if input_file_path.endswith(".json"):
        db = preprocess_document(input_file_path, embeddings)

        officer_data = get_response_from_query(db, query, TEMPERATURE, K, template)

        for item in officer_data:
            item["fn"] = os.path.basename(input_file_path)

        output_df = pd.DataFrame(officer_data)
        output_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    input_directory = "../../ocr/data/output"
    output_directory = "../data/output"

    embeddings = generate_hypothetical_embeddings()

    templates = [template_1, template_2,]

    for query in QUERIES:
        for template_index, template in enumerate(templates, start=1):
            for file_name in os.listdir(input_directory):
                if file_name.endswith(".json"):
                    input_file_path = os.path.join(input_directory, file_name)
                    output_file_name = f"{os.path.splitext(file_name)[0]}_template_{template_index}.csv"
                    output_file_path = os.path.join(output_directory, output_file_name)
                    process_file(input_file_path, output_file_path, embeddings, query, template)