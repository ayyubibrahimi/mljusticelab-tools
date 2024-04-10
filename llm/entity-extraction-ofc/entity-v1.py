import os
import logging
import pandas as pd
from langchain.chains import LLMChain
from langchain_community.document_loaders import JSONLoader
from langchain.vectorstores.faiss import FAISS

from helper import generate_hypothetical_embeddings, PROMPT_TEMPLATE_HYDE, sort_retrived_documents, extract_officer_data, clean_name

import spacy

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np


import numpy as np
from langchain_anthropic import ChatAnthropic

import sys


nlp = spacy.load("en_core_web_lg")


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TEMPERATURE = 1
K = 20

template="""
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel.
  
    Query: {question}

    Documents: {docs}

    The response will contain:

    1) The name of a law enforcement personnel. Law enforcement personnel can be identified by searching for these name prefixes: ofcs., officers, sergeants, sgts., lieutenants, lts., captains, cpts., commanders, sheriffs, deputies, dtys., detectives, dets., inspectors, technicians, analysts, coroners.
       Please prefix the name with "Officer Name: ". 
       For example, "Officer Name: John Smith".

    2) If available, provide an in-depth description of the context of their mention. 
       If the context induces ambiguity regarding the individual's employment in law enforcement, please make this clear in your response.
       Please prefix this information with "Officer Context: ". 

    3) Review the context to discern the role of the officer. For example, Lead Detective (Homicide Division), Supervising Officer (Crime Lab), Detective, Officer on Scene, Arresting Officer, Crime Lab Analyst
       Please prefix this information with "Officer Role: "
       For example, "Officer Role: Lead Detective"

    
    The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: John Smith 
    Officer Context: Mentioned as someone who was present during a search, along with other detectives from different units.
    Officer Role: Patrol Officer

    Officer Name: 
    Officer Context:
    Officer Role: 

    - Do not include any prefixes
    - Only derive responses from factual information found within the police reports.
    - If the context of an identified person's mention is not clear in the report, provide their name and note that the context is not specified.
    - Do not extract information about victims and witnesses
    """


QUERIES = [
    "In the transcript, identify individuals by their names along with their specific law enforcement titles, such as officer, sergeant, lieutenant, captain, commander, sheriff, deputy, detective, inspector, technician, analyst, det., sgt., lt., cpt., p.o., ofc., and coroner. Alongside each name and title, note the context of their mention. This includes the roles they played in key events, decisions they made, actions they took, their interactions with others, responsibilities in the case, and any significant outcomes or incidents they were involved in."
]


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["page_number"] = record.get("page_number")
    return metadata


law_enforcement_titles = [
    "officer", "sergeant", "lieutenant", "captain", "commander", "sheriff",
    "deputy", "detective", "inspector", "technician", "analyst", "coroner",
    "chief", "marshal", "agent", "superintendent", "commissioner", "trooper",
    "constable", "special agent", "patrol officer", "field agent", "investigator",
    "forensic specialist", "crime scene investigator", "public safety officer",
    "security officer", "patrolman", "watch commander", "undercover officer",
    "intelligence officer", "tactical officer", "bomb technician", "K9 handler",
    "SWAT team member", "emergency dispatcher", "corrections officer",
    "probation officer", "parole officer", "bailiff", "court officer",
    "wildlife officer", "park ranger", "border patrol agent", "immigration officer",
    "customs officer", "air marshal", "naval investigator", "military police",
    "forensic scientist", 
    "forensic analyst", 
    "crime lab technician", 
    "forensic technician", 
    "laboratory analyst", 
    "DNA analyst", 
    "toxicologist", 
    "serologist", 
    "ballistics expert", 
    "fingerprint analyst", 
    "forensic chemist", 
    "forensic biologist", 
    "trace evidence analyst", 
    "forensic pathologist", 
    "forensic odontologist", 
    "forensic entomologist", 
    "forensic anthropologist", 
    "digital forensic analyst", 
    "forensic engineer", 
    "crime scene examiner", 
    "evidence technician", 
    "latent print examiner", 
    "forensic psychologist", 
    "forensic document examiner",
    "forensic photography specialist"
    "det.",
    "sgt.",
    "cpt.",
    "lt.",
    "p.o.",
    "dpty.",
    "ofc.",
]


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
                for title in law_enforcement_titles:
                    if title in page.page_content[max(0, ent.start_char - 100):ent.end_char + 100].lower():
                        entities.append(ent.text)
                        break
        page_entities.append((page, len(entities)))

    page_entities.sort(key=lambda x: x[1], reverse=True)

    # print(page_entities)

    selected_pages = [page for page, _ in page_entities[:MAX_PAGES]]

    if selected_pages:
        db = FAISS.from_documents(selected_pages, embeddings)
    else:
        db = None

    return db


def get_response_from_query(db, query, temperature, k):
    logger.info("Performing query...")
    if db is None:
        return "", []
    doc_list = db.similarity_search_with_score(query, k=k)
    if not doc_list:
        return "", []
    docs = sort_retrived_documents(doc_list)

    print(docs)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    llm = ChatAnthropic(model_name="claude-3-sonnet-20240229",)

    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307")

    prompt_response = ChatPromptTemplate.from_template(template)
    response_chain = prompt_response | llm | StrOutputParser()
    responses = []
    page_numbers = []
    for doc in docs:
        page_content = doc[0].page_content.replace('\n', ' ')
        page_number = doc[0].metadata.get('page_number')
        if page_content:
            response = response_chain.invoke({"question": query, "docs": page_content})
            responses.append(response)
        else:
            responses.append("")
        if page_number is not None:
            page_numbers.append(page_number)
    concatenated_responses = "\n\n".join(responses)
    # print(concatenated_responses)
    return concatenated_responses, page_numbers


def process_file(input_file_path, output_file_path, embeddings, query):
    if input_file_path.endswith(".json"):
        output_data = []

        db = preprocess_document(input_file_path, embeddings)

        officer_data_string, page_numbers = get_response_from_query(db, query, TEMPERATURE, K)

        officer_data = extract_officer_data(officer_data_string)

        for item in officer_data:
            item["page_number"] = page_numbers
            item["fn"] = os.path.basename(input_file_path)
            item["Prompt Template for Hyde"] = PROMPT_TEMPLATE_HYDE
            item["Prompt Template for Model"] = template
            item["Chunk Size"] = CHUNK_SIZE
            item["Chunk Overlap"] = CHUNK_OVERLAP
            item["Temperature"] = TEMPERATURE
            item["k"] = K
            item["hyde"] = "1"
            item["model"] = "gpt-3.5-turbo-finetuned"
        output_data.extend(officer_data)

        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python entity.py <input_json_file> <output_csv_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    embeddings = generate_hypothetical_embeddings()

    for query in QUERIES:
        process_file(input_file_path, output_file_path, embeddings, query)