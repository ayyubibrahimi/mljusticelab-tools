import os
import logging
import json
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain_anthropic import ChatAnthropic
import multiprocessing
import os
from functools import partial
import time
from langchain_together import ChatTogether
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# nltk.download('stopwords')


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

llm = ChatTogether(
    together_api_key="",
    model="mistralai/Mistral-7B-Instruct-v0.3",
)

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
            augmented_text += f"#({ent.text}: {label})#"
            prev_end = ent.end_char
        elif (ent.start_char, ent.end_char) in entity_map:
            label = entity_map[(ent.start_char, ent.end_char)]
            augmented_text += text[prev_end : ent.start_char]
            augmented_text += f"#({ent.text}: {label})#"
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


def process_page(docs, i, query, window_size, memory_log):
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)

    prompt_response = ChatPromptTemplate.from_template(summary_template)
    response_chain = prompt_response | llm | StrOutputParser()

    current_page = docs[i].page_content.replace("\n", " ")
    previous_page_ending = (
        docs[i - 1].page_content.replace("\n", " ")[-window_size:]
        if i > 0
        else ""
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
                "memory_log": memory_log,
                "previous_page_ending": previous_page_ending,
                "current_page": current_page,
                "next_page_beginning": next_page_beginning,
            }
        )
        response["page_content"] = processed_content
    
    time.sleep(3)  # Add a 1-second delay between each page processing
    
    return response


def process_batch(batch_docs, batch_start, query, window_size, memory_log):
    sorted_results = []
    for i in range(len(batch_docs)):
        result = process_page(batch_docs, i, query, window_size, memory_log)
        sorted_results.append(result)

    section_summaries = {"messages": sorted_results}
    combined_summaries, _ = combine_summaries([section_summaries], memory_log)
    start_page = sorted_results[0]["page_number"]
    end_page = sorted_results[-1]["page_number"]
    
    time.sleep(3)  # Add a 3-second delay between each batch processing
    
    return combined_summaries[0], start_page, end_page


def generate_summaries(docs, query, memory_log, window_size=100, batch_size=10, max_parallel_batches=3):
    batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
    combined_summaries = []

    for i in range(0, len(batches), max_parallel_batches):
        batch_range = batches[i:i+max_parallel_batches]
        pool = multiprocessing.Pool()
        results = pool.starmap(
            process_batch,
            [(batch, i * batch_size, query, window_size, memory_log) for i, batch in enumerate(batch_range)],
        )
        pool.close()
        pool.join()

        combined_summaries.extend([result[:3] for result in results])
        time.sleep(5)  # Add a 5-second delay between processing each set of parallel batches

    return combined_summaries

memory_log_template = """
As a Legal Clerk, update the memory log only when the new summary contains crucial information directly related to the main subject of the document. Maintain a concise memory log that focuses on the key aspects of the events, allegations, investigations, and outcomes described in the document.

## Guidelines ##

1. Review the current memory log and new summary to determine if an update is necessary.

2. If the new summary contains crucial information specific to the main subject of the document, identify the key details to include.

3. Focus on maintaining a clear understanding of the following aspects:
   - The main individuals involved and their roles, including their badge numbers, if available. 
   - The specific allegations, charges, or rule violations
   - Key events and actions during any investigations
   - Important evidence or findings
   - Legal proceedings, motions, or resolutions
   - Disciplinary actions or outcome

4. Do not infer any details that are not explicitly stated in the source document.

If the new summary does not contain any directly relevant information, return the memory log as is.

You must always output the contents of the memory log. Output your response in bullet point format.

IMPORTANT
Provide your response in a format similar to the examples below:

**Example**:
Incident Overview:

Details of Alleged Misconduct:

Officer's Statement and Context:

Witness Statements:

Investigation Findings:

Complaint and Recommended Disciplinary Action:

Case Status:


## Original Memory Log ##: {memory_log}

## New Summary ##: {summary}

## Original Memory Log or Updated Memory Log ##:
"""

memory_log_verification_template = """
As a Legal Clerk, compare the old memory log with the updated memory log to ensure that no important information specific to the main subject of the document has been accidentally deleted in the update process. The memory log should serve as a concise summary of the key aspects of the events, allegations, investigations, and outcomes described in the document.

Guidelines:
1. Carefully review both the old and updated memory logs.

2. Identify any crucial information present in the old memory log that is missing from the updated memory log, focusing on details directly related to the main subject of the document.
2. Identify any crucial information present in the old memory log that is missing from the updated memory log, focusing on details directly related to the main subject of the document.

3. Ensure that the following key details are retained in the memory log:
   - The main individuals involved and their roles, including their badge numbers, if available. 
   - The specific allegations, charges, or rule violations
   - Key events and actions during any investigations
   - Important evidence or findings
   - Legal proceedings, motions, or resolutions
   - Disciplinary actions or outcomes
   
4. If any important details have been omitted, include them in the final memory log output.

5. Do not infer any details that are not explicitly stated in the source document.

6. If no changes need to be made, return the original memory log as the final memory log. 

IMPORTANT
Provide your response in a format similar to the examples below:

**Example**:
Incident Overview:

Details of Alleged Misconduct:

Officer's Statement and Context:

Witness Statements:

Investigation Findings:

Complaint and Recommended Disciplinary Action:

Case Status:

## Original Memory Log ##: {old_memory_log}

## Updated Memory Log ##: {updated_memory_log}

## Updated Memory Log or New Memory Log##:
"""


def update_memory_log(memory_log, new_summary):
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", api_key="")

    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307", api_key="", temperature=0)


    memory_log_prompt = ChatPromptTemplate.from_template(memory_log_template)
    memory_log_chain = memory_log_prompt | llm | StrOutputParser()
    updated_memory_log = memory_log_chain.invoke({"summary": new_summary, "memory_log": memory_log})

    memory_log_verification_prompt = ChatPromptTemplate.from_template(memory_log_verification_template)
    memory_log_verification_chain = memory_log_verification_prompt | llm | StrOutputParser()
    final_memory_log = memory_log_verification_chain.invoke({"old_memory_log": memory_log, "updated_memory_log": updated_memory_log})

    return final_memory_log

def create_memory_log(docs):
    memory_log = ""
    num_pages = len(docs)
    
    # Process the first 10 pages
    for i in range(min(10, num_pages)):
        current_page = docs[i].page_content.replace("\n", " ")
        summary = process_page(docs, i, "", 0, memory_log)["page_content"]
        memory_log = update_memory_log(memory_log, summary)
        print(memory_log)
    
    # Process the last 10 pages (skipping the first 10 if already processed)
    start_index = max(10, num_pages - 10)
    for i in range(start_index, num_pages):
        current_page = docs[i].page_content.replace("\n", " ")
        summary = process_page(docs, i, "", 0, memory_log)["page_content"]
        memory_log = update_memory_log(memory_log, summary)
        # print(memory_log)
    
    return memory_log

# ### difference is that we're asking the model to preserve the original language 

summary_template = """
As a Legal Clerk, your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page, using the context from the memory log and surrounding pages when necessary for clarity or relevance. Follow these guidelines to create an accurate and thorough summary:

## Guidelines ##

1. Extract all important details from the current page, including but not limited to:
   - Individuals mentioned, including their full names, roles, badge numbers, and specific actions
   - Allegations, charges, and/or rule violations, providing case numbers and exact dates when available
   - Main events, actions, and/or observations described, including precise dates and locations when provided
   - Relevant evidence or findings presented
   - Legal proceedings, motions, disciplinary actions, or investigation outcomes, including specific dates, case law citations, and arguments made by involved parties
    - Include all relevant details from the current page, even if they do not fall under the categories of key information outlined in the guidelines.

2. Support the extracted data with additional context from the memory log and surrounding pages only when it enhances the understanding or relevance of the information in the current page. DO NOT be duplicative and DO NOT summarize the memory log itself. Use it only for context to understand and interpret the current page's information accurately.

3. Use clear, concise language and maintain accuracy by quoting or closely paraphrasing the original text.

4. DO NOT include any details not explicitly stated in either summary.

5. Present the summary in a bulletpoint format, using subheadings if needed to organize distinct aspects of the information.

## Memory Log ##

{memory_log}

## Previous Page Ending ##:

{previous_page_ending}

## Next Page Beginning ##:

{next_page_beginning}

## Current Page ##:

{current_page}

### Current Page Summary:
"""

combine_template = """
As a Legal Clerk, your task is to concatenate the provided page summaries into a single, comprehensive, and well-organized summary for the given section of the police report. This summary should present all relevant information from both the current combined summary and the new page summary in a detailed, chronological, and coherent manner without any duplication. 

## Guidelines ##:

1. Comprehensive Information Integration:
   - Review the current combined summary and the new page summary to extract all relevant information, ensuring no important details are overlooked.
   - Include full names, roles, badge numbers, specific events with dates, policy or rule violations, disciplinary actions, relevant evidence, legal actions, and outcomes related to the case from both sources.

3. Narrative Coherence:
   - Support the extracted data with additional context from the memory log and surrounding pages only when it enhances the understanding or relevance of the information in the current page. DO NOT be duplicative and DO NOT summarize the memory log itself. Use it only for context to understand and interpret the current page's information accurately.
   
4. Handling Contradictions:
   - If inconsistencies arise between the current combined summary and the new page summary, prioritize the most detailed and specific information.

5. Factual Accuracy:
   - DO NOT include any details not explicitly stated in either summary.

6. Formatting for Clarity:
   - Ensure that the updated combined summary is formatted as bullet points with a logical flow of information. If possible, organize the bullet points chronologically. 
   
IMPORTANT
Provide your response in a format similar to the examples below:

**Example**:
Incident Overview:
- Brief description of the incident, including date, location, and involved parties
- Overview of the alleged misconduct and the department's response

Details of Alleged Misconduct:
- How and when the alleged misconduct was discovered
- Details about the evidence that led to the discovery
- Specific policy violation or rule violation

Officer's Statement and Context:
- Officer's admission or denial of the allegations
- Any contextual information provided by the officer to explain their actions

Witness Statements:
- Relevant information provided by witnesses, if applicable
- How witness statements support or contradict the officer's account

Investigation Findings:
- Summary of the investigation's findings
- Specific violations of department policies or regulations
- Disciplinary action

Complaint and Recommended Disciplinary Action:
- Details about the formal complaint filed against the officer
- Recommended disciplinary actions based on the investigation's findings

Case Status:
- Current status of the case and any pending actions
- Brief summary of the key points and conclusion of the report

Memory Log and Summaries:

## Memory Log ##:
{memory_log}

## Current Combined Summary ##:
{current_combined_summary}

## New Page Summary ##:
{new_page_summary}

## Updated Combined Summary: ##
"""

verification_template = """
As a Legal Clerk, your task is to review the updated combined summary, which integrates content from the current combined summary and the new summary of a police investigative report. This verification process aims to ensure that all relevant information from both the current combined summary and the new summary is accurately retained in the updated combined summary, such as full names, roles, badge numbers, specific events with dates, policy or rule violations, disciplinary actions, relevant evidence, legal actions, and outcomes related to the case from both sources.

## Verification Guidelines ##

1. Comprehensive Information Integration:
   - Ensure that all important details from the new summary, such as critical people, key facts, key events, and significant details, are accurately incorporated into the updated combined summary.

2. Context Preservation:
   - Verify that all of the important information from both the current combined summary and the new summary is preserved in the updated combined summary.
   
3. Logical Flow:
   - Evaluate the updated combined summary for logical flow and coherence, ensuring that the newly integrated information fits seamlessly into the existing narrative. If possible, this information should be ordered chronologically. 
   
4. Factual Accuracy:
   - DO NOT include any details not explicitly stated in either summary.

   
IMPORTANT
Do not change the formatting style of the updated combined summary.
Follow the exact format as specified below.

Focus solely on integrating and verifying the content from the provided summaries.

## Current Combined Summary ##:
{current_combined_summary}

## New Summary ##:
{new_page_summary}

## Provide the updated combined summary below, ensuring that all relevant information from both the current combined summary and the new summary is accurately retained. If no updates are needed, return the current combined summary ##
"""

coherence_template = """
As a Legal Clerk, your task is to review the provided bullet point summary and reorganize it into a more coherent and well-structured format. Please follow these steps to improve the summary:

1. Carefully review the bullet point summary and identify all the points and key details, paying special attention to names, policy or rule violations, events, actions, disciplinary actions such as suspensions and terminations, dates, locations, case numbers, and legal references.

2. Organize the bullet points in chronological order, ensuring that the sequence of events is accurately represented and that all relevant details are included.

3. Factual Accuracy:
   - DO NOT include any details not explicitly stated in either summary.

4. Ensure that the reorganized bullet points:
   - Present a clear, logical progression of events, with all relevant details included
   - Use concise and unambiguous language
   - Do not introduce any new information or infer details not explicitly stated in the original summary

IMPORTANT
Provide your response in a format similar to the examples below:

**Example**:
Incident Overview:
- Brief description of the incident, including date, location, and involved parties
- Overview of the alleged misconduct and the department's response

Details of Alleged Misconduct:
- How and when the alleged misconduct was discovered
- Details about the evidence that led to the discovery
- Specific policy violation or rule violation

Officer's Statement and Context:
- Officer's admission or denial of the allegations
- Any contextual information provided by the officer to explain their actions

Witness Statements:
- Relevant information provided by witnesses, if applicable
- How witness statements support or contradict the officer's account

Investigation Findings:
- Summary of the investigation's findings
- Specific violations of department policies or regulations
- Disciplinary action

Complaint and Recommended Disciplinary Action:
- Details about the formal complaint filed against the officer
- Recommended disciplinary actions based on the investigation's findings

Case Status:
- Current status of the case and any pending actions
- Brief summary of the key points and conclusion of the report

## Original Bullet Point Summary ##:
{bulletpoint_summary}

Reorganized Bullet Point Summary:
"""

improvement_template = """
As a Legal Clerk, your task is to produce a verified bullet point summary by comparing an updated bullet point summary with the original bullet point summary. The goal is to create a final summary that captures all the essential information from both summaries, ensuring accuracy, coherence, and logical structure. Please follow these steps:

1. Carefully review both summaries and identify all the key points and details, paying special attention to specific names, policy or rule violations, disciplinary actions. events, actions, titles, dates, locations, and case numbers. Update the summary with any key details that are missing.

2. Ensure that the information is organized as bullet points into logical sections, such as those in the example below, with related points grouped under headers and presented in a clear chronological sequence.

3. Factual Accuracy:
   - DO NOT include any details not explicitly stated in either summary.

4. Verify that the final summary:
   - Includes all relevant information from both the updated and original summaries
   - Presents a coherent and logically structured account of the events
   - Uses clear and concise language
   - Does not introduce any new information or infer details not explicitly stated in either summary

IMPORTANT
Provide your response in a format similar to the examples below. Only include headers that are relevant to improving an understanding of the case. 

**Example**:
Incident Overview:
- Brief description of the incident, including date, location, and involved parties
- Overview of the alleged misconduct and the department's response

Details of Alleged Misconduct:
- Details about the specific policy violation or rule violation
- How and when the alleged misconduct was discovered
- Details about the evidence that led to the discovery

Officer's Statement and Context:
- Officer's admission or denial of the allegations
- Any contextual information provided by the officer to explain their actions

Witness Statements:
- Relevant information provided by witnesses, if applicable
- How witness statements support or contradict the officer's account

Investigation Findings:
- Summary of the investigation's findings
- Specific violations of department policies or regulations
- Disciplinary action

Complaint and Recommended Disciplinary Action:
- Details about the formal complaint filed against the officer
- Recommended disciplinary actions based on the investigation's findings

Case Status:
- Current status of the case and any pending actions
- Brief summary of the key points and conclusion of the report

## Updated Bullet Point Summary ##:
{coherent_summary}

## Original Bullet Point Summary ##:
{bulletpoint_summary}

## Return the Updated Bullet Point Summary or the Verified Bullet Point Summary Below ##:
"""

def format_and_improve_summary(bulletpoint_summary, summaries, memory_log):
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307", api_key="", temperature=0)
    
    # Format bulletpoint summary into coherent narrative
    prompt_response = ChatPromptTemplate.from_template(coherence_template)
    response_chain = prompt_response | llm | StrOutputParser()
    coherent_summary = response_chain.invoke({"bulletpoint_summary": bulletpoint_summary})
    coherent_memory_log = response_chain.invoke({"bulletpoint_summary": memory_log})
    
    # Improve coherent summary based on comparison with bulletpoint summary
    prompt_response = ChatPromptTemplate.from_template(improvement_template)
    response_chain = prompt_response | llm | StrOutputParser()
    improved_summary = response_chain.invoke({
        "coherent_summary": coherent_summary,
        "bulletpoint_summary": bulletpoint_summary
    })
    
    improved_memory_log = response_chain.invoke({
        "coherent_summary": coherent_memory_log,
        "bulletpoint_summary": memory_log
    })
    
    return improved_summary, improved_memory_log


def combine_summaries(summaries, memory_log):
    combiner_llm = llm
    combiner_prompt = ChatPromptTemplate.from_template(combine_template)
    combiner_chain = combiner_prompt | combiner_llm | StrOutputParser()

    verification_llm = llm
    verification_prompt = ChatPromptTemplate.from_template(verification_template)
    verification_chain = verification_prompt | verification_llm | StrOutputParser()

    combined_summaries = []

    for section_summaries in summaries:
        current_combined_summary = section_summaries["messages"][0]["page_content"]
        combined_page_numbers = section_summaries["messages"][0].get(
            "page_numbers", [section_summaries["messages"][0].get("page_number")]
        )

        for i in range(1, len(section_summaries["messages"])):
            new_page_summary = section_summaries["messages"][i]["page_content"]

            updated_combined_summary = combiner_chain.invoke(
                {
                    "current_combined_summary": current_combined_summary,
                    "new_page_summary": new_page_summary,
                    "memory_log": memory_log
                }
            )
            print(updated_combined_summary)

            verified_combined_summary = verification_chain.invoke(
                {
                    "updated_combined_summary": updated_combined_summary,
                    "current_combined_summary": current_combined_summary,
                    "new_page_summary": new_page_summary,
                    "memory_log": memory_log,
                }
            )
            print(verified_combined_summary)

            current_combined_summary = verified_combined_summary

            combined_page_numbers.extend(
                section_summaries["messages"][i].get("page_numbers", [section_summaries["messages"][i].get("page_number")])
            )

        improved_summary, improved_memory_log = format_and_improve_summary(current_combined_summary, summaries, memory_log)

        combined_summaries.append({"page_content": improved_summary, "page_numbers": combined_page_numbers})

    return combined_summaries, memory_log

final_combine_template = """
As a Legal Clerk, your task is to concatenate the provided page summaries into a single, comprehensive, and well-organized summary for the given section of the police report. This summary should present all relevant information from both the current combined summary and the new page summary in a detailed, chronological, and coherent manner without any duplication. Focus on ensuring that the data is only comprised of the important details. Do not include tangentially important information.

## Guidelines ##:

1. Comprehensive Information Integration:
   - Review the current combined summary and the new page summary to extract all relevant information, ensuring no important details are overlooked.
   - Include full names, roles, badge numbers, specific events with dates, policy or rule violations, disciplinary actions, relevant evidence, legal actions, and outcomes related to the case from both sources.

2. Handling Contradictions:
   - If inconsistencies arise between the current combined summary and the new summary, prioritize the most detailed and specific information.

3. Factual Accuracy:
   - DO NOT include any details not explicitly stated in either summary.

4. Formatting for Clarity:
   - Ensure that the information is organized as bullet points into logical sections, such as those in the example below, with related points grouped under headers and presented in a clear chronological sequence.

IMPORTANT
Provide your response in a format similar to the examples below. Only include headers that are relevant to improving an understanding of the case. 

**Example**:
Incident Overview:
- Brief description of the incident, including date, location, and involved parties
- Overview of the alleged misconduct and the department's response

Details of Alleged Misconduct:
- How and when the alleged misconduct was discovered
- Details about the evidence that led to the discovery
- Specific policy or rule violations

Officer's Statement and Context:
- Officer's admission or denial of the allegations
- Any contextual information provided by the officer to explain their actions

Witness Statements:
- Relevant information provided by witnesses, if applicable
- How witness statements support or contradict the officer's account

Investigation Findings:
- Summary of the investigation's findings
- Specific violations of department policies or regulations
- Disciplinary actions 

Complaint and Recommended Disciplinary Action:
- Details about the formal complaint filed against the officer
- Recommended disciplinary actions based on the investigation's findings

Case Status:
- Current status of the case and any pending actions
- Brief summary of the key points and conclusion of the report

Current Summary:
{current_summary}

New Summary:
{new_summary}

## Updated Combined Summary: ##
"""

final_verification_template = """
As a Legal Clerk, your task is to review the updated summary and ensure that it accurately captures all the information and specific details from the current summary and the new summary, styled as a narrative in bullet point format with a clear beginning, middle, and end, following the temporal timeline of the investigative documents. Focus on ensuring that the data is only comprised of the important details. Do not include tangentially important information. Please follow these steps:

1. Carefully compare the updated summary against the current summary and the new summary.

2. Verify that all of the key people, events, and specific details from both summaries are present in the updated summary, ensuring no information has been deleted, including full names, roles, badge numbers, specific events with dates, policy or rule violations, disciplinary actions, relevant evidence, legal actions, and outcomes. 

3. Ensure the summary is formatted as a series of bullet points that:
    - Present a clear, logical sequence of events, including all relevant details, forming a narrative with a distinct beginning, middle, and end
    - Employ concise and unambiguous language
    - Refrain from introducing new information or inferring details not explicitly stated in the original summaries

4. Check for any missing or inconsistent information, including specific details, between the updated summary and the original summaries.

5. Factual Accuracy:
   - DO NOT include any details not explicitly stated in either summary.

IMPORTANT
Provide your response in a format similar to the examples below:

**Example**:
Incident Overview:
- Brief description of the incident, including date, location, and involved parties
- Overview of the alleged misconduct and the department's response

Details of Alleged Misconduct:
- How and when the alleged misconduct was discovered
- Details about the evidence that led to the discovery
- Specific rule or policy violations

Officer's Statement and Context:
- Officer's admission or denial of the allegations
- Any contextual information provided by the officer to explain their actions

Witness Statements:
- Relevant information provided by witnesses, if applicable
- How witness statements support or contradict the officer's account

Investigation Findings:
- Summary of the investigation's findings
- Specific violations of department policies or regulations
- Disciplinary actions

Complaint and Recommended Disciplinary Action:
- Details about the formal complaint filed against the officer
- Recommended disciplinary actions based on the investigation's findings

Case Status:
- Current status of the case and any pending actions
- Brief summary of the key points and conclusion of the report

Current Summary:
{current_summary}

New Summary:
{new_summary}

Updated Summary:
{updated_summary}

## Return the Verified Summary or Return the Contents of The Updated Summary if No Changes Are Needed. Return this Summary Without Reference To Your Verification Check ##:
"""

def combine_final_summaries(summaries, memory_log):
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307", api_key="", temperature=0)

    combine_prompt_template = ChatPromptTemplate.from_template(final_combine_template)
    combine_chain = combine_prompt_template | llm | StrOutputParser()

    verification_prompt_template = ChatPromptTemplate.from_template(final_verification_template)
    verification_chain = verification_prompt_template | llm | StrOutputParser()

    current_summary = summaries[0][0]["page_content"]

    for i in range(1, len(summaries)):
        new_summary = summaries[i][0]["page_content"]
        
        # Logging each step
        print(f"Combining summaries {i} and {i+1}")
        print(f"Current Summary: {current_summary[:100]}...")  # Display first 100 chars for brevity
        print(f"New Summary: {new_summary[:100]}...")

        updated_summary = combine_chain.invoke({
            "current_summary": current_summary,
            "new_summary": new_summary
        })

        final_updated_summary = verification_chain.invoke({
            "current_summary": current_summary,
            "new_summary": new_summary,
            "updated_summary": updated_summary
        })
        
        # Log after each verification step
        print(f"Final Updated Summary after iteration {i}: {final_updated_summary[:100]}...")

        current_summary = final_updated_summary

    return {"page_content": current_summary}, memory_log


def save_summaries_to_json(summaries, memory_log, output_file):
    output_data = [{"page_content": memory_log, "page_number": ""}] + summaries
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

def save_summaries_to_text(summaries, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for summary, start_page, end_page in sorted(summaries, key=lambda x: x[1]):
            f.write(f"Pages {start_page}-{end_page}:\n{summary['page_content']}\n\n")

def save_memory_log_to_text(memory_log, output_file):
    memory_log_dir = os.path.dirname(output_file)
    os.makedirs(memory_log_dir, exist_ok=True)
    memory_log_file = os.path.join(memory_log_dir, "memory_log.txt")
    with open(memory_log_file, "w") as f:
        f.write(memory_log)

def save_final_summary_to_text(final_summary, output_file):
    final_summary_dir = os.path.dirname(output_file)
    os.makedirs(final_summary_dir, exist_ok=True)
    final_summary_file = os.path.join(final_summary_dir, "final_summary.txt")
    with open(final_summary_file, "w") as f:
        f.write(final_summary["page_content"])


# Can we attempt to create summaries w bigger and smaller intervals (5, 20) and maybe create a more detailed memory log if so


if __name__ == "__main__":
    base_input_directory = "../../ocr/data/output/test"
    base_output_directory = "../data/output/test"

    for input_path in os.listdir(base_input_directory):
        input_file_path = os.path.join(base_input_directory, input_path)

        if os.path.isfile(input_file_path) and input_path.endswith(".json"):
            # Handle individual JSON file
            output_json_path = os.path.join(
                base_output_directory, f"{os.path.splitext(input_path)[0]}_summary_v3.json"
            )
            output_text_path = os.path.join(
                base_output_directory, f"{os.path.splitext(input_path)[0]}_summary_v3.txt"
            )

            # Check if the output files already exist
            if os.path.exists(output_json_path) and os.path.exists(output_text_path):
                print(f"Skipping {input_path} - output files already exist.")
                continue

            docs = load_and_split(input_file_path)
            memory_log = create_memory_log(docs)  
            query = "Generate a timeline of events based on the police report."
            combined_summaries = generate_summaries(docs, query, memory_log)
            final_summary, memory_log = combine_final_summaries(combined_summaries, memory_log)


            os.makedirs(base_output_directory, exist_ok=True)
            save_summaries_to_json(combined_summaries, memory_log, output_json_path)
            save_summaries_to_text(combined_summaries, output_text_path)
            save_memory_log_to_text(memory_log, output_text_path)
            save_final_summary_to_text(final_summary, output_text_path)

        elif os.path.isdir(input_file_path):
            # Handle directory
            output_directory = os.path.join(base_output_directory, input_path)

            # Create the output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            for filename in os.listdir(input_file_path):
                if filename.endswith(".json"):
                    json_path = os.path.join(input_file_path, filename)
                    output_json_path = os.path.join(
                        output_directory, f"{os.path.splitext(filename)[0]}_summary_v3.json"
                    )
                    output_text_path = os.path.join(
                        output_directory, f"{os.path.splitext(filename)[0]}_summary_v3.txt"
                    )

                    # Check if the output files already exist
                    if os.path.exists(output_json_path) and os.path.exists(output_text_path):
                        print(f"Skipping {filename} - output files already exist.")
                        continue

                    docs = load_and_split(json_path)
                    memory_log = create_memory_log(docs)  
                    query = "Generate a timeline of events based on the police report."
                    combined_summaries = generate_summaries(docs, query, memory_log)
                    final_summary, memory_log = combine_final_summaries(combined_summaries, memory_log)

                    save_summaries_to_json(combined_summaries, memory_log, output_json_path)
                    save_summaries_to_text(combined_summaries, output_text_path)
                    save_final_summary_to_text(final_summary, output_text_path)

                    save_memory_log_to_text(memory_log, output_text_path)