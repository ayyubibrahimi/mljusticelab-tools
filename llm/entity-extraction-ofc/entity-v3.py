import os
import logging
import pandas as pd
from langchain.chains import LLMChain
from langchain_community.document_loaders import JSONLoader
from langchain.vectorstores.faiss import FAISS

from helper import generate_hypothetical_embeddings, PROMPT_TEMPLATE_HYDE, preprocess_document

import spacy

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np


import numpy as np
from langchain_anthropic import ChatAnthropic

import sys
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import re
from fuzzywuzzy import fuzz
import tempfile
import shutil



load_dotenv(find_dotenv())

nlp = spacy.load("en_core_web_lg")


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
TEMPERATURE = 0
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

def get_response_from_query(db, query, temperature, k, template):
    logger.info("Performing query...")
    if db is None:
        return "", []
    doc_list = db.similarity_search_with_score(query, k=k)
    if not doc_list:
        return "", []
    

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

    prompt_response = ChatPromptTemplate.from_template(template)
    response_chain = prompt_response | llm | StrOutputParser()
    responses = []
    page_numbers = []
    for doc in doc_list:
        page_content = doc[0].page_content.replace('\n', ' ')
        page_number = doc[0].metadata.get('page_number')
        if page_content:
            response = response_chain.invoke({"question": query, "docs": page_content})
            responses.append((response, page_number))  # Store response and page number as a tuple
        else:
            responses.append(("", None))  # Store empty response and None for page number
    return responses


def deduplicate_officers(officer_data, threshold=90):
    # List of prefixes to be stripped from the names
    prefixes = ["P.O.", "Officer", "Lieutenant", "Captain", "Sergeant", "Deputy", "Sheriff", 
                "ofc", "lt", "sgt", "cpt", "dep", "shf"]
    
    deduplicated_data = {}
    for officer in officer_data:
        original_name = str(officer.get("Officer Name", ""))
        # Initialize name prefix field

        if not original_name or original_name.lower() == 'nan':
            continue

        name_prefix = ""
        
        # Check and strip prefix from the officer name
        for prefix in prefixes:
            if original_name.startswith(prefix):
                name_prefix = prefix
                # Remove prefix and strip leading/trailing spaces
                name = original_name[len(prefix):].strip()
                break
        else:
            name = original_name
        
        # Normalize name to lowercase for case-insensitive comparison
        name = name.lower()
        
        matched = False
        for deduplicated_name, deduplicated_officer in deduplicated_data.items():
            if fuzz.ratio(name, deduplicated_name) >= threshold:
                page_number = str(officer.get("page_number", "")).split(",")
                fn = str(officer.get("fn", "")).split(",")

                deduplicated_officer["page_numbers"] = list(set(deduplicated_officer["page_numbers"] + page_number))
                deduplicated_officer["fns"] = list(set(deduplicated_officer["fns"] + fn))

                if name_prefix:
                    deduplicated_officer["Name Prefix"] = name_prefix
                
                deduplicated_officer.update(
                    {k: v for k, v in officer.items() if k not in ["page_number", "fn", "Officer Name"]}
                )
                matched = True
                break
        
        if not matched:
            officer["Officer Name"] = name
            officer["page_numbers"] = list(set(str(officer.get("page_number", "")).split(",")))
            officer["fns"] = list(set(str(officer.get("fn", "")).split(",")))
            if name_prefix:
                officer["Name Prefix"] = name_prefix

            deduplicated_data[name] = officer

    return list(deduplicated_data.values())



def clean_name(officer_name):
    return re.sub(
        r"(Detective|Officer|[Dd]et\.|[Ss]gt\.|[Ll]t\.|[Cc]pt\.|[Oo]fc\.|Deputy|Captain|[CcPpLl]|Sergeant|Lieutenant|Techn?i?c?i?a?n?|Investigator|^-|\d{1}\)|\w{1}\.)\.?\s+",
        "",
        officer_name,
    )


def extract_officer_data(text, template):
    officer_data = []

    normalized_text = re.sub(r"\s*-\s*", "", text)

    officer_sections = re.split(r"\n(?=Officer Name:)", normalized_text)

    for section in officer_sections:
        if not section.strip():
            continue

        officer_dict = {}

        name_match = re.search(r"Officer Name:\s*(.*?)\s*(?:Officer|Location|Case Number|Collaborators|Evidence)", section, re.DOTALL)
        if name_match and name_match.group(1):
            officer_dict["Officer Name"] = name_match.group(1).strip()

        if "Officer Role:" in template:
            role_match = re.search(r"Officer Role:\s*(.*?)\s*(?:Department/Unit|Location|Case Number|Collaborators|Evidence)", section, re.DOTALL)
            if role_match and role_match.group(1):
                officer_dict["Officer Role"] = role_match.group(1).strip()

        if "Department/Unit:" in template:
            department_match = re.search(r"Department/Unit:\s*(.*)", section, re.DOTALL)
            if department_match and department_match.group(1):
                officer_dict["Department/Unit"] = department_match.group(1).strip()

        if "Location:" in template:
            location_match = re.search(r"Location:\s*(.*?)\s*(?:Date/Time|Case Number|Collaborators|Evidence)", section, re.DOTALL)
            if location_match and location_match.group(1):
                officer_dict["Location"] = location_match.group(1).strip()

        if "Date/Time:" in template:
            datetime_match = re.search(r"Date/Time:\s*(.*)", section, re.DOTALL)
            if datetime_match and datetime_match.group(1):
                officer_dict["Date/Time"] = datetime_match.group(1).strip()

        if "Case Number:" in template:
            case_number_match = re.search(r"Case Number:\s*(.*?)\s*(?:Event Type|Collaborators|Evidence)", section, re.DOTALL)
            if case_number_match and case_number_match.group(1):
                officer_dict["Case Number"] = case_number_match.group(1).strip()

        if "Event Type:" in template:
            event_type_match = re.search(r"Event Type:\s*(.*)", section, re.DOTALL)
            if event_type_match and event_type_match.group(1):
                officer_dict["Event Type"] = event_type_match.group(1).strip()

        if "Collaborators:" in template:
            collaborators_match = re.search(r"Collaborators:\s*(.*?)\s*(?:Supervisor|Evidence)", section, re.DOTALL)
            if collaborators_match and collaborators_match.group(1):
                officer_dict["Collaborators"] = collaborators_match.group(1).strip()

        if "Supervisor:" in template:
            supervisor_match = re.search(r"Supervisor:\s*(.*)", section, re.DOTALL)
            if supervisor_match and supervisor_match.group(1):
                officer_dict["Supervisor"] = supervisor_match.group(1).strip()

        if "Evidence, Actions, and Interactions:" in template:
            evidence_match = re.search(r"Evidence, Actions, and Interactions:\s*(.*?)\s*(?:Outcome)", section, re.DOTALL)
            if evidence_match and evidence_match.group(1):
                officer_dict["Evidence, Actions, and Interactions"] = evidence_match.group(1).strip()

        if "Outcome:" in template:
            outcome_match = re.search(r"Outcome:\s*(.*)", section, re.DOTALL)
            if outcome_match and outcome_match.group(1):
                officer_dict["Outcome"] = outcome_match.group(1).strip()

        if officer_dict:
            officer_data.append(officer_dict)

    return officer_data






def process_file(input_file_path, output_file_path, embeddings, query, template):
    if input_file_path.endswith(".json"):
        output_data = []

        db = preprocess_document(input_file_path, embeddings)

        responses = get_response_from_query(db, query, TEMPERATURE, K, template)

        for response, page_number in responses:
            officer_data = extract_officer_data(response, template)
            for item in officer_data:
                item["page_number"] = str(page_number)  # Convert to a string
                item["fn"] = os.path.basename(input_file_path)  # Store as a string
            output_data.extend(officer_data)

        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    input_directory = "../../ocr/data/output"
    output_directory = "../data/output"

    embeddings = generate_hypothetical_embeddings()

    templates = [template_1, template_2, template_3, template_4, template_5]

    with tempfile.TemporaryDirectory() as temp_dir:
        for query in QUERIES:
            for template_index, template in enumerate(templates, start=1):
                for file_name in os.listdir(input_directory):
                    if file_name.endswith(".json"):
                        input_file_path = os.path.join(input_directory, file_name)
                        temp_file_path = os.path.join(temp_dir, f"{os.path.splitext(file_name)[0]}_template_{template_index}.csv")
                        process_file(input_file_path, temp_file_path, embeddings, query, template)

        all_officer_data = []
        for file_name in os.listdir(temp_dir):
            if file_name.endswith(".csv"):
                file_path = os.path.join(temp_dir, file_name)
                df = pd.read_csv(file_path)
                all_officer_data.extend(df.to_dict("records"))

        deduplicated_officer_data = deduplicate_officers(all_officer_data)

    output_file_path = os.path.join(output_directory, "deduplicated_officers.csv")
    output_df = pd.DataFrame(deduplicated_officer_data)
    output_df.to_csv(output_file_path, index=False)
