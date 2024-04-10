import os
import logging
import pandas as pd
from langchain.chains import LLMChain
from langchain_community.document_loaders import JSONLoader
from helper import generate_hypothetical_embeddings, PROMPT_TEMPLATE_HYDE, extract_officer_data, preprocess_document
import spacy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from langchain_anthropic import ChatAnthropic
import sys
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import re
from fuzzywuzzy import fuzz
import tempfile
import shutil

import json



load_dotenv(find_dotenv())

nlp = spacy.load("en_core_web_lg")


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

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
            responses.append((response, page_number))  
        else:
            responses.append(("", None))  
    return responses


def deduplicate_officers(officer_data, threshold=90):
    prefixes = ["P.O.", "Officer", "Lieutenant", "Captain", "Sergeant", "Deputy", "Sheriff", 
                "ofc", "lt", "sgt", "cpt", "dep", "shf"]
    
    deduplicated_data = {}
    for officer in officer_data:
        original_name = str(officer.get("Officer Name", ""))

        if not original_name or original_name.lower() == 'nan':
            continue

        name_prefix = ""
        
        for prefix in prefixes:
            if original_name.startswith(prefix):
                name_prefix = prefix
                name = original_name[len(prefix):].strip()
                break
        else:
            name = original_name
        
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

def create_officer_network(officer_data):
    officer_network = {}

    for officer in officer_data:
        officer_name = officer["Officer Name"]
        officer_network[officer_name] = {
            "Location": [],
            "Department/Unit": [],
            "Date/Time": [],
        }

    for i in range(len(officer_data)):
        officer1 = officer_data[i]
        officer1_name = officer1["Officer Name"]

        for j in range(i + 1, len(officer_data)):
            officer2 = officer_data[j]
            officer2_name = officer2["Officer Name"]

            if officer1.get("Location") == officer2.get("Location"):
                officer_network[officer1_name]["Location"].append(officer2_name)
                officer_network[officer2_name]["Location"].append(officer1_name)

            if officer1.get("Department/Unit") == officer2.get("Department/Unit"):
                officer_network[officer1_name]["Department/Unit"].append(officer2_name)
                officer_network[officer2_name]["Department/Unit"].append(officer1_name)

            if officer1.get("Date/Time") == officer2.get("Date/Time"):
                officer_network[officer1_name]["Date/Time"].append(officer2_name)
                officer_network[officer2_name]["Date/Time"].append(officer1_name)

    return officer_network


def save_officer_network_json(officer_network, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(officer_network, json_file, indent=4)


def process_file(input_file_path, output_file_path, embeddings, query, template):
    if input_file_path.endswith(".json"):
        output_data = []

        db = preprocess_document(input_file_path, embeddings)

        responses = get_response_from_query(db, query, TEMPERATURE, K, template)

        for response, page_number in responses:
            officer_data = extract_officer_data(response, template)
            for item in officer_data:
                item["page_number"] = str(page_number)  
                item["fn"] = os.path.basename(input_file_path) 
            output_data.extend(officer_data)

        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    input_directory = "../../ocr/data/output"
    output_directory = "../data/output"

    embeddings = generate_hypothetical_embeddings()

    templates = [template_1, template_2]

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
        officer_network = create_officer_network(deduplicated_officer_data)


    output_file_path = os.path.join(output_directory, "deduplicated_officers.csv")
    output_df = pd.DataFrame(deduplicated_officer_data)
    output_df.to_csv(output_file_path, index=False)
    
    json_output_file = '../data/output/officer_network.json'
    save_officer_network_json(officer_network, json_output_file)
