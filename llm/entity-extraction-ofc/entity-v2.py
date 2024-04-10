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
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel.

    Query: {question}

    Documents: {docs}

    The response will contain:

    1. The name of a law enforcement personnel. Law enforcement personnel can be identified by searching for these name prefixes: ofcs., officers, sergeants, sgts., lieutenants, lts., captains, cpts., commanders, sheriffs, deputies, dtys., detectives, dets., inspectors, technicians, analysts, coroners. Please prefix the name with "Officer Name: ". For example, "Officer Name: John Smith".
    
    2. Review the context to discern the role of the officer. For example, Lead Detective (Homicide Division), Supervising Officer (Crime Lab), Detective, Officer on Scene, Arresting Officer, Crime Lab Analyst. Please prefix this information with "Officer Role: ". For example, "Officer Role: Lead Detective".
    
    3. If available, provide the department or unit the officer belongs to. This could include Homicide Division, Narcotics Unit, Crime Scene Investigation Unit, etc. Please prefix this information with "Department/Unit: ". For example, "Department/Unit: Homicide Division".
    
    The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: John Smith
    Officer Role: Patrol Officer
    Department/Unit: Central Precinct

    Officer Name: 
    Officer Role: 
    Department/Unit: 

    ### Additional Instructions
    Only derive responses from factual information found within the police reports.
    If the context does not provide clear information about an officer's role or department/unit, leave those fields blank.
    If multiple officers are mentioned, provide a separate entry for each officer, following the specified format.
    Do not include any additional prefixes, numbering, or formatting beyond what is specified in the template.
    """

template_2 ="""
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel and their spatiotemporal features.

    Query: {question}

    Documents: {docs}

    The response will contain:

    1. The name of a law enforcement personnel. Law enforcement personnel can be identified by searching for these name prefixes: ofcs., officers, sergeants, sgts., lieutenants, lts., captains, cpts., commanders, sheriffs, deputies, dtys., detectives, dets., inspectors, technicians, analysts, coroners. Please prefix the name with "Officer Name: ". For example, "Officer Name: John Smith".
    2. The location where the officer was mentioned or involved in an event. This could include crime scene addresses, precinct locations, or any other relevant locations. Please prefix this information with "Location: ". For example, "Location: 123 Main Street, Anytown, USA".
    3. The date and time associated with the officer's involvement in an event or case. If an exact date and time are not available, provide the most specific information available (e.g., month and year, or approximate time of day). Please prefix this information with "Date/Time: ". For example, "Date/Time: January 15, 2023, 14:30".
    
    The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: John Smith
    Location: 123 Main Street, Anytown, USA
    Date/Time: January 15, 2023, 14:30

    Officer Name: 
    Location: 
    Date/Time: 

    Additional Instructions
    Only derive responses from factual information found within the police reports.
    If the context does not provide clear information about an officer's location or date/time of involvement, leave those fields blank.
    If multiple officers are mentioned, provide a separate entry for each officer, following the specified format.
    Do not include any additional prefixes, numbering, or formatting beyond what is specified in the template. 
    """

template_3 ="""
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel and their case-related features.

    Query: {question}

    Documents: {docs}

    The response will contain:

    1. The name of a law enforcement personnel. Law enforcement personnel can be identified by searching for these name prefixes: ofcs., officers, sergeants, sgts., lieutenants, lts., captains, cpts., commanders, sheriffs, deputies, dtys., detectives, dets., inspectors, technicians, analysts, coroners. Please prefix the name with "Officer Name: ". For example, "Officer Name: John Smith".
    2. The case number or identifier related to the officer's involvement. This could be a specific case number, file number, or any other unique identifier associated with the case. Please prefix this information with "Case Number: ". For example, "Case Number: CR-2023-1234".
    3. The type of event the officer was involved in. This could include crime scene investigation, arrest, interrogation, witness interview, evidence collection, or any other relevant event type. Please prefix this information with "Event Type: ". For example, "Event Type: Crime Scene Investigation".
    
    The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: John Smith
    Case Number: CR-2023-1234
    Event Type: Crime Scene Investigation

    Officer Name: 
    Case Number: 
    Event Type: 

    ### Additional Instructions
    Only derive responses from factual information found within the police reports.
    If the context does not provide clear information about an officer's associated case number or event type, leave those fields blank.
    If multiple officers are mentioned, provide a separate entry for each officer, following the specified format.
    Do not include any additional prefixes, numbering, or formatting beyond what is specified in the template.
    """


template_4 ="""
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel and their collaboration features.

    Query: {question}

    Documents: {docs}

    The response will contain:

    1. The name of a law enforcement personnel. Law enforcement personnel can be identified by searching for these name prefixes: ofcs., officers, sergeants, sgts., lieutenants, lts., captains, cpts., commanders, sheriffs, deputies, dtys., detectives, dets., inspectors, technicians, analysts, coroners. Please prefix the name with "Officer Name: ". For example, "Officer Name: John Smith".
    2. Other officers or personnel the officer collaborated with or worked alongside. This could include other officers, detectives, specialists, or any other law enforcement personnel mentioned in the context of working together. Please prefix this information with "Collaborators: ". For example, "Collaborators: Detective Sarah Johnson, Officer Michael Lee".
    3. The name and role of the officer's supervisor or superior, if mentioned in the context. This could be a sergeant, lieutenant, captain, or any other higher-ranking officer. Please prefix this information with "Supervisor: ". For example, "Supervisor: Captain David Brown".
    
    The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: John Smith
    Collaborators: Detective Sarah Johnson, Officer Michael Lee
    Supervisor: Sergeant Emily Davis

    Officer Name: 
    Collaborators: 
    Supervisor:

    ### Additional Instructions
    Only derive responses from factual information found within the police reports.
    If the context does not provide clear information about an officer's collaborators or supervisor, leave those fields blank.
    If multiple officers are mentioned, provide a separate entry for each officer, following the specified format.
    Do not include any additional prefixes, numbering, or formatting beyond what is specified in the template. 
    """


template_5 ="""
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel, their evidence handling, actions, interactions, and case outcomes.

    Query: {question}

    Documents: {docs}

    The response will contain:

    1. The name of a law enforcement personnel. Law enforcement personnel can be identified by searching for these name prefixes: ofcs., officers, sergeants, sgts., lieutenants, lts., captains, cpts., commanders, sheriffs, deputies, dtys., detectives, dets., inspectors, technicians, analysts, coroners. Please prefix the name with "Officer Name: ". For example, "Officer Name: John Smith".
    2. Information about the specific evidence the officer collected, processed, or handled, key actions or decisions made by the officer during the event or investigation, and notable interactions the officer had with other officers, suspects, witnesses, or victims. Please prefix this information with "Evidence, Actions, and Interactions: ". For example, "Evidence, Actions, and Interactions: Collected fingerprints from the crime scene, interviewed the suspect, and coordinated with the forensic team."
    3. The outcome or result of the officer's involvement in the case. This could include arrests made, charges filed, case resolution, or any other significant outcomes. Please prefix this information with "Outcome: ". For example, "Outcome: Suspect arrested and charged with murder."
    
    The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: John Smith
    Evidence, Actions, and Interactions: Collected DNA samples from the victim's clothing, interviewed witnesses at the scene, and coordinated with the homicide detective.
    Outcome: Evidence contributed to the identification and arrest of the suspect.

    Officer Name: Sarah Johnson
    Evidence, Actions, and Interactions: 
    Outcome: 

    ### Additional Instructions
    Only derive responses from factual information found within the police reports.
    If the context does not provide clear information about an officer's evidence handling, actions, interactions, or case outcome, leave those fields blank.
    If multiple officers are mentioned, provide a separate entry for each officer, following the specified format.
    Do not include any additional prefixes, numbering, or formatting beyond what is specified in the template.
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
    MAX_PAGES = 10
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


def get_response_from_query(db, query, temperature, k, template):
    logger.info("Performing query...")
    if db is None:
        return "", []
    doc_list = db.similarity_search_with_score(query, k=k)
    if not doc_list:
        return "", []
    docs = sort_retrived_documents(doc_list)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

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
            print(response)
        else:
            responses.append("")
        if page_number is not None:
            page_numbers.append(page_number)
    concatenated_responses = "\n\n".join(responses)
    return concatenated_responses, page_numbers


def process_file(input_file_path, output_file_path, embeddings, query, template):
    if input_file_path.endswith(".json"):
        output_data = []

        db = preprocess_document(input_file_path, embeddings)

        officer_data_string, page_numbers = get_response_from_query(db, query, TEMPERATURE, K, template)

        officer_data = extract_officer_data(officer_data_string)

        for item in officer_data:
            item["page_number"] = page_numbers
            item["fn"] = os.path.basename(input_file_path)
        output_data.extend(officer_data)

        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    input_directory = "../../ocr/data/output"
    output_directory = "../data/output"

    embeddings = generate_hypothetical_embeddings()

    templates = [template_1, template_2, template_3, template_4, template_5]

    for query in QUERIES:
        for template_index, template in enumerate(templates, start=1):
            for file_name in os.listdir(input_directory):
                if file_name.endswith(".json"):
                    input_file_path = os.path.join(input_directory, file_name)
                    output_file_name = f"{os.path.splitext(file_name)[0]}_template_{template_index}.csv"
                    output_file_path = os.path.join(output_directory, output_file_name)
                    process_file(input_file_path, output_file_path, embeddings, query, template)