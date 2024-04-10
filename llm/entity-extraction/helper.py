import re
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
import spacy
from langchain_community.document_loaders import JSONLoader
import logging


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)



nlp = spacy.load("en_core_web_lg")


PROMPT_TEMPLATE_HYDE = PromptTemplate(
    input_variables=["question"],
    template="""
    You're an AI assistant specializing in criminal justice research. 
    Your main focus is on identifying the names and providing detailed context of mention for each law enforcement personnel. 
    This includes police officers, detectives, deupties, lieutenants, sergeants, captains, technicians, coroners, investigators, patrolman, and criminalists, 
    as described in court transcripts.
    Be aware that the titles "Detective" and "Officer" might be used interchangeably.
    Be aware that the titles "Technician" and "Tech" might be used interchangeably.

    Question: {question}

    Roles and Responses:""",
)

def generate_hypothetical_embeddings():
    llm = OpenAI(api_key="")
    prompt = PROMPT_TEMPLATE_HYDE

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    base_embeddings = OpenAIEmbeddings(api_key="")

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings




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




def sort_retrived_documents(doc_list):
    docs = sorted(doc_list, key=lambda x: x[1], reverse=True)

    third = len(docs) // 3

    highest_third = docs[:third]
    middle_third = docs[third : 2 * third]
    lowest_third = docs[2 * third :]

    highest_third = sorted(highest_third, key=lambda x: x[1], reverse=True)
    middle_third = sorted(middle_third, key=lambda x: x[1], reverse=True)
    lowest_third = sorted(lowest_third, key=lambda x: x[1], reverse=True)

    docs = highest_third + lowest_third + middle_third
    return docs



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