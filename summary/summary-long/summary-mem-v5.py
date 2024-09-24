import os
import logging
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
import multiprocessing
from collections import namedtuple
from functools import partial
from dotenv import find_dotenv, load_dotenv
import concurrent.futures
from datetime import datetime
import time

load_dotenv(find_dotenv())

Doc = namedtuple("Doc", ["page_content", "metadata"])

llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def word_count(text):
    return len(text.split())


def load_and_split(file_path):
    logger.info(f"Processing document: {file_path}")

    with open(file_path, "r") as file:
        file_content = file.read()
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            data = {}

    print(f"Keys in parsed JSON data: {data.keys()}")

    if "files" in data:
        if data["files"]:
            docs = []
            original_page_number = 1
            for message in data["files"]:
                page_content = message.get("page_content", "")
                if word_count(page_content) >= 50:
                    doc = Doc(
                        page_content=page_content,
                        metadata={"seq_num": original_page_number},
                    )
                else:
                    doc = Doc(
                        page_content="No data to be processed on this page",
                        metadata={"seq_num": original_page_number},
                    )
                docs.append(doc)
                original_page_number += 1
            # print(docs)

            logger.info(f"Data loaded from document: {file_path}")
            return docs


memory_log_template = """
<task_description>
As a Legal Clerk, your task is to review the new summary and update the memory log only when the new summary contains crucial information directly related to the main subject of the document. Maintain a memory log that focuses on the key aspects of the events, allegations, investigations, and outcomes described in the document.
</task_description>

<guidelines>
1. Review and Compare:
   • Carefully review the current memory log and the new summary.
   • Determine if the new summary contains crucial information that is not already in the memory log.

2. Identify Crucial Information:
   • Focus on information specific to the main subject of the document.
   • Look for key details related to events, allegations, investigations, and outcomes.

3. Update Selectively:
   • Only update the memory log if the new summary contains crucial information not already present.
   • If updating, integrate the new information seamlessly into the existing log.

4. Maintain Conciseness:
   • Keep the memory log focused and concise.
   • Avoid redundancy or unnecessary details.

5. Ensure Accuracy:
   • Only include information that is directly stated in the document.
   • Do not infer or speculate beyond what is explicitly mentioned.

6. Preserve Original Structure:
   • If no update is necessary, reproduce the original memory log without changes.
</guidelines>

<essential_information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
b. Primary parties involved (full names, roles, badge numbers if applicable)
j. Allegations of misconduct and any associated information
c. Key legal issues, claims, charges, or arguments
k. Disciplinary outcomes or their current status
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 
For each type of essential information classification, be specific when referring to people, places, and dates. 
</essential_information>

<thinking_process>
Before updating the memory log, consider:
1. Does the new summary contain any crucial information not already in the memory log?
2. How does this new information relate to the main subject of the document?
3. Can this new information be integrated into the existing log without disrupting its flow?
4. Is this information essential to understanding the key aspects of the case?
5. Am I maintaining the conciseness of the log while including all crucial details?
</thinking_process>

<warnings>
- Do not add information that is not directly stated in the document
- Avoid speculation or inference beyond what is explicitly mentioned
- Do not remove or alter existing crucial information in the memory log
- Ensure that any updates maintain the chronological and logical flow of events
- Be cautious of potential inconsistencies between the new summary and existing log
</warnings>

<reference_materials>
## Original Memory Log ##
{memory_log}

## New Summary ##
{summary}
</reference_materials>

<output_instruction>
Based on your review of the current memory log and the new summary, provide either an updated memory log incorporating the crucial new information, or reproduce the original memory log if no update is necessary. Ensure the output maintains a concise focus on key aspects of events, allegations, investigations, and outcomes related to the main subject of the document:
</output_instruction>
"""


def create_memory_log(docs, pages_to_concatenate=1):
    memory_log = ""
    num_pages = len(docs)

    def concatenate_pages(start_idx, num_pages_to_concat):
        combined_content = ""
        for j in range(num_pages_to_concat):
            if start_idx + j < num_pages:
                combined_content += (
                    docs[start_idx + j].page_content.replace("\n", " ") + " "
                )
        return combined_content.strip()

    def find_next_data_page(start_idx):
        for i in range(start_idx, num_pages, pages_to_concatenate):
            combined_content = concatenate_pages(i, pages_to_concatenate)
            if "No data to be processed on this page" not in combined_content:
                return i
        return None

    # Process the first pages with data
    start_idx = find_next_data_page(0)
    if start_idx is not None:
        end_idx = min(start_idx + 1, num_pages)
        for i in range(start_idx, end_idx, pages_to_concatenate):
            combined_content = concatenate_pages(i, pages_to_concatenate)
            summary = process_memory_log_page(docs, i, combined_content, 0, memory_log)[
                "page_content"
            ]
            memory_log = update_memory_log(memory_log, summary)
            logger.info(f"Initial Memory Log: {memory_log}...")  # Log first 100 chars

    # Process the last pages with data (skipping the first ones if already processed)
    start_index = max(end_idx if start_idx is not None else 0, num_pages - 1)
    last_data_idx = find_next_data_page(start_index)
    if last_data_idx is not None:
        for i in range(last_data_idx, num_pages, pages_to_concatenate):
            combined_content = concatenate_pages(i, pages_to_concatenate)
            if "No data to be processed on this page" in combined_content:
                continue
            summary = process_memory_log_page(docs, i, combined_content, 0, memory_log)[
                "page_content"
            ]
            memory_log = update_memory_log(memory_log, summary)
            logger.info(f"Updated Memory Log: {memory_log}...")  # Log last 100 chars

    return memory_log


def update_memory_log(memory_log, new_summary):
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    memory_log_prompt = ChatPromptTemplate.from_template(memory_log_template)
    memory_log_chain = memory_log_prompt | llm | StrOutputParser()

    updated_memory_log = memory_log_chain.invoke(
        {"summary": new_summary, "memory_log": memory_log}
    )

    return updated_memory_log


def process_memory_log_page(docs, i, current_page, window_size, memory_log):
    prompt_response = ChatPromptTemplate.from_template(summary_template)
    response_chain = prompt_response | llm | StrOutputParser()

    previous_page_ending = (
        docs[i - 1].page_content.replace("\n", " ")[-window_size:] if i > 0 else ""
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
                "current_page": current_page,
            }
        )
        response["page_content"] = processed_content

    return response


page_summary_template = """
<document_classification>
First, determine if this document is a legal document or another document type. Consider the following:
- Does it contain legal terminology, case numbers, or references to laws and regulations?
- Is it structured like a legal document (e.g., contracts, court filings, police reports)?
- Does it discuss legal proceedings, rights, or obligations?

Based on your analysis, classify this document as either:
1. Legal Document
2. Other Document Type
</document_classification>

<task_description>
Your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page. 
</task_description>


<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:

b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, or charges
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 

For each type of essential information, be specific when referring to people, places, and dates. 
</legal_document_essential_information>


<other_document_type_guidelines>
If classified as Other Document Type, follow these guidelines:

Ensure the summary includes the following elements, if present:
b. Main topics or themes
c. Key individuals or organizations mentioned
d. Important dates, events, or milestones
e. Significant data or findings
f. Main arguments or conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context

</other_document_type_guidelines>

<thinking_process>
Before summarizing, consider:
1. Is this a legal document or another document type?
2. What are the main topics on this page?
3. What context from the memory log is relevant?
4. What are the most important pieces of information to extract based on the document type?
</thinking_process>

<output_format>
Present the summary using the following structure:
- Main topic 1
  • Sub-topic 1.1
  • Sub-topic 1.2
- Main events
  • Sub-event 2.1
  • Sub-event 2.2
- Main persons
  • Sub-person 2.1
  • Sub-person 2.2
- Main actions
  • Sub-action 2.1
  • Sub-action 2.2
</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
</warnings>

<reference_materials>
Current Page:
{current_page}
</reference_materials>

<output_instruction>
First, state the document classification (Legal Document or Other Document Type) and provide a brief explanation for your decision. Then, generate the current page summary following the appropriate guidelines based on the classification.
</output_instruction>
"""

summary_template = """
<document_classification>
First, determine if this document is a legal document or another document type. Consider the following:
- Does it contain legal terminology, case numbers, or references to laws and regulations?
- Is it structured like a legal document (e.g., contracts, court filings, police reports)?
- Does it discuss legal proceedings, rights, or obligations?

Based on your analysis, classify this document as either:
1. Legal Document
2. Other Document Type
</document_classification>

<task_description>
Your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page.
</task_description>


<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:

b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, or charges
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 
For each type of essential information, be specific when referring to people, places, and dates. 
</legal_document_essential_information>


<other_document_type_guidelines>
If classified as Other Document Type, follow these guidelines:

Ensure the summary includes the following elements, if present:
b. Main topics or themes
c. Key individuals or organizations mentioned
d. Important dates, events, or milestones
e. Significant data or findings
f. Main arguments or conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context

</other_document_type_guidelines>

<thinking_process>
Before summarizing, consider:
1. Is this a legal document or another document type?
2. What are the main topics on this page?
3. What are the most important pieces of information to extract based on the document type?
</thinking_process>

<output_format>
Present the summary using the following structure:
- Main topic 1
  • Sub-topic 1.1
  • Sub-topic 1.2
- Main events
  • Sub-event 2.1
  • Sub-event 2.2
- Main persons
  • Sub-person 3.1
  • Sub-person 3.2
- Main actions
  • Sub-action 4.1
  • Sub-action 4.2
- Main legal pissue
  • Sub-legal ossue 5.1
  • Sub-legal issue 5.2
- Main legal procedure
  • Sub-legal procedure 6.1
  • Sub-legal procedure 6.2
</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
</warnings>

<reference_materials>
Current Page:
{current_page}
</reference_materials>

<output_instruction>
First, state the document classification (Legal Document or Other Document Type) and provide a brief explanation for your decision. Then, generate the current page summary following the appropriate guidelines based on the classification.
</output_instruction>
"""


combine_template = """
<task_description>
Your task is to combine the provided summaries into a single, comprehensive, and well-organized final summary for the given document. Your primary goal is to preserve ALL important information from both summaries, creating a comprehensive summary without any omissions.
</task_description>

<document_type_check>
First, determine if the summaries are from a legal document or another document type based on the classification provided in each summary. State the document type before proceeding with the combination.
</document_type_check>

<guidelines>
1. Comprehensive Information Integration:
   • Include ALL important information from both summaries, even if it results in a longer combined summary.
   • Use bullet points to capture all important details. 

2. Handling Contradictions:
   • When encountering contradictions, include both versions and clearly mark them as conflicting.

3. Factual Accuracy:
   • Include only details explicitly stated in either summary.
   • If information is incomplete or unclear, maintain that ambiguity rather than making assumptions.

4. Completeness Check:
   • After combining, review both original summaries to ensure no important information has been omitted.
   • If any omissions are found, immediately add them to the combined summary with appropriate source tags.
</guidelines>

<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:
b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, or charges
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 
For each type of essential information, be specific when referring to people, places, and dates. 
</legal_document_essential_information>

<other_document_type_essential_information>
If the document is classified as another document type, ensure the summary includes the following elements, if present:
b. Main topics or themes
c. Key individuals or organizations mentioned
d. Important dates, events, or milestones
e. Significant data, statistics, or findings
f. Main arguments or conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context
</other_document_type_essential_information>

<thinking_process>
Before and during the combination of summaries, consider:
1. Is this a legal document or another document type?
2. What are the main topics covered across all summaries?
3. Are there any contradictions or inconsistencies between summaries that need to be highlighted?
4. What information provides crucial context, and how can I ensure it's not lost in the combination?
5. Have I double-checked that no important information from either summary has been omitted?
</thinking_process>

<warnings>
- Prioritize completeness. It's better to include all important information than to omit details for brevity.
- Do not include speculative information or draw conclusions not explicitly stated in the summaries.
- Do not alter the meaning or context of any information when integrating it into the combined summary.
</warnings>

<reference_materials>
Summary 1:
{summary_1}

Summary 2:
{summary_2}
</reference_materials>

<output_instruction>
Generate the final combined summary below, ensuring it adheres to all guidelines, includes all essential information based on the document type, and is presented in a clear, bullet-point format:

## Updated Combined Summary: ##
</output_instruction>
"""

verification_template = """
<task_description>
Your task is to meticulously review the combined summary, which integrates content from two individual summaries (summary 1 and summary 2) of a document. This verification process aims to ensure that ALL relevant information from both original summaries is accurately contained within the combined summary, including key details, events, findings, and outcomes related to the document from both sources.
</task_description>

<document_type_check>
First, confirm the document type (Legal Document or Other Document Type) based on the classification provided in the combined summary. State the document type before proceeding with the verification.
</document_type_check>

<verification_guidelines>
1. Systematic Comparison:
   • Create a checklist of all important points from both original summaries.
   • Systematically check each point against the combined summary, marking items as present or missing.

2. Information Preservation:
   • Ensure that ALL important details from both summaries are accurately incorporated into the combined summary.

3. Missing Information Addition:
   • For any information found missing during the review, explicitly add it to the verified summary. 
</verification_guidelines>

<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:
b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, or charges
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 

For each type of essential information, be specific when referring to people, places, and dates. 
</legal_document_essential_information>

<other_document_type_essential_information>
If the document is classified as another document type, ensure the summary includes the following elements, if present:
b. Main topics or themes
c. Key individuals or organizations mentioned
d. Important dates, events, or milestones
e. Significant data, statistics, or findings
f. Main arguments or conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context
</other_document_type_essential_information>

<thinking_process>
During the verification process, consider:
1. Have I created a comprehensive checklist of all important points from both original summaries?
2. Am I systematically comparing each point to ensure nothing is missed?
3. After my initial review, did I re-read the original summaries to catch any overlooked details?
4. If I found any missing information, have I added it to the verified summary with proper source attribution?
</thinking_process>

<warnings>
- Prioritize completeness over conciseness. 
- Do not alter the meaning or context of any information during the verification process.
- Ensure that all ambiguities or uncertainties from the original summaries are maintained.
- Do not add speculative information or draw conclusions not present in the original summaries or combined summary.
- If significant information is missing or inaccurately represented, be prepared to recommend a re-combination of the summaries.
</warnings>


<reference_materials>
Current Combined Summary:
{current_combined_summary}

Summary 1:
{summary_1}

Summary 2:
{summary_2}
</reference_materials>

<output_instruction>
First, confirm the document type (Legal Document or Other Document Type) based on the classification in the combined summary. Then, provide the verified summary below. If no changes are needed, return the contents of the Updated Summary. Present the summary in a clear, bullet-point format, organized chronologically for legal documents or thematically for other document types. Do not include any reference to your verification check in the final output.

## Verified Summary: ##
</output_instruction>
"""


def process_interval(interval_index, start_index, interval_size, docs, memory_log):
    end_index = start_index + interval_size
    interval_docs = docs[start_index:end_index]

    page_summaries = generate_summaries(interval_docs, memory_log)
    combined_summary, _ = process_summaries(page_summaries, memory_log)

    return combined_summary


def process_file(filename, input_directory, output_directory, memory_log):
    json_path = os.path.join(input_directory, filename)
    docs = load_and_split(json_path)

    total_pages = len(docs)

    # Determine the number of intervals based on the document size
    if total_pages <= 10:
        num_intervals = 1
    elif total_pages <= 50:
        num_intervals = 2
    else:
        num_intervals = 4

    base_interval_size = total_pages // num_intervals
    extra_pages = total_pages % num_intervals

    interval_sizes = [
        base_interval_size + (1 if i < extra_pages else 0) for i in range(num_intervals)
    ]

    with multiprocessing.Pool() as pool:
        process_interval_partial = partial(
            process_interval, docs=docs, memory_log=memory_log
        )
        interval_summaries = pool.starmap(
            process_interval_partial,
            [
                (i, sum(interval_sizes[:i]), interval_sizes[i])
                for i in range(num_intervals)
            ],
        )

    for interval, summary in enumerate(interval_summaries):
        interval_output_json_path = os.path.join(
            output_directory,
            f"{os.path.splitext(filename)[0]}_interval_{interval+1}_summary.json",
        )
        with open(interval_output_json_path, "w") as f:
            json.dump(summary, f, indent=2)

    return interval_summaries


def process_batch(batch, num_pages_to_concat, memory_log):
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    prompt_response = ChatPromptTemplate.from_template(page_summary_template)
    response_chain = prompt_response | llm | StrOutputParser()

    results = []
    i = 0
    while i < len(batch):
        concat_pages = []
        page_numbers = []
        while len(concat_pages) < num_pages_to_concat and i < len(batch):
            current_page = batch[i].page_content.replace("\n", " ")
            if "No data to be processed on this page" not in current_page:
                concat_pages.append(current_page)
                page_numbers.append(batch[i].metadata.get("seq_num"))
            i += 1

        if concat_pages:
            current_page = " ".join(concat_pages)
            processed_content = response_chain.invoke({"current_page": current_page})

            results.append(
                {"page_content": processed_content, "page_numbers": page_numbers}
            )

        if not concat_pages:
            break

    return results


def create_batches(docs, batch_size):
    return [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]


def generate_summaries(docs, memory_log, batch_size=10, num_pages_to_concat=2):
    batches = create_batches(docs, batch_size)

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_batch, batch, num_pages_to_concat, memory_log)
            for batch in batches
        ]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    return results


def combine_summaries(summaries, memory_log):
    combiner_llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    combiner_prompt = ChatPromptTemplate.from_template(combine_template)
    combiner_chain = combiner_prompt | combiner_llm | StrOutputParser()

    verification_llm = ChatAnthropic(
        model_name="claude-3-haiku-20240307", temperature=0
    )
    verification_prompt = ChatPromptTemplate.from_template(verification_template)
    verification_chain = verification_prompt | verification_llm | StrOutputParser()

    verified_summary = summaries[0]["page_content"]
    combined_page_numbers = summaries[0].get(
        "page_numbers", [summaries[0].get("page_number")]
    )

    for i in range(1, len(summaries)):
        combined_summary = combiner_chain.invoke(
            {"summary_1": verified_summary, "summary_2": summaries[i]["page_content"]}
        )

        verified_summary = verification_chain.invoke(
            {
                "current_combined_summary": combined_summary,
                "summary_1": verified_summary,
                "summary_2": summaries[i]["page_content"],
            }
        )

        combined_page_numbers.extend(
            summaries[i].get("page_numbers", [summaries[i].get("page_number")])
        )

    return {
        "page_content": verified_summary,
        "page_numbers": combined_page_numbers,
    }, memory_log


final_combine_template = """
<task_description>
Your task is to combine the provided summaries into a single, comprehensive, and well-organized final summary for the given document. Your primary goal is to preserve ALL important information from both summaries, creating a comprehensive summary without any omissions.
</task_description>

<document_type_check>
First, determine if the summaries are from a legal document or another document type based on the classification provided in each summary. State the document type before proceeding with the combination.
</document_type_check>

<guidelines>
1. Comprehensive Information Integration:
   • Include ALL important information from both summaries, even if it results in a longer combined summary.
   • Use bullet points to capture all important details. 

2. Handling Contradictions:
   • When encountering contradictions, include both versions and clearly mark them as conflicting.

3. Factual Accuracy:
   • Include only details explicitly stated in either summary.
   • If information is incomplete or unclear, maintain that ambiguity rather than making assumptions.

4. Completeness Check:
   • After combining, review both original summaries to ensure no important information has been omitted.
   • If any omissions are found, immediately add them to the combined summary with appropriate source tags.
</guidelines>

<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:
b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, charges, or arguments
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 

For each type of essential information, be specific when referring to people, places, and dates. 
</legal_document_essential_information>

<other_document_type_essential_information>
If the document is classified as another document type, ensure the summary includes the following elements, if present:
b. Main topics and themes
c. Key individuals and/or organizations mentioned
d. Important dates, events, or milestones
e. Significant data, statistics, or findings
f. Main arguments and conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context
</other_document_type_essential_information>

<thinking_process>
Before and during the combination of summaries, consider:
1. Is this a legal document or another document type?
2. What are the main topics covered across all summaries?
3. Are there any contradictions or inconsistencies between summaries that need to be highlighted?
4. What information provides crucial context, and how can I ensure it's not lost in the combination?
5. Have I double-checked that no important information from either summary has been omitted?
</thinking_process>

<warnings>
- Prioritize completeness. It's better to include all important information than to omit details for brevity.
- Do not include speculative information or draw conclusions not explicitly stated in the summaries.
- Do not alter the meaning or context of any information when integrating it into the combined summary.
</warnings>

<reference_materials>
Summary 1:
{summary_1}

Summary 2:
{summary_2}
</reference_materials>

<output_instruction>
Generate the final combined summary below, ensuring it adheres to all guidelines, includes all essential information based on the document type, and is presented in a clear, bullet-point format:

## Updated Combined Summary: ##
</output_instruction>
"""

final_verification_template = """
<task_description>
Your task is to meticulously review the combined summary, which integrates content from two individual summaries (summary 1 and summary 2) of a document. This verification process aims to ensure that ALL relevant information from both original summaries is accurately contained within the combined summary, including key details, events, findings, and outcomes related to the document from both sources. Include relevant information from the memory log if it improves the overall summary.
</task_description>

<document_type_check>
First, confirm the document type (Legal Document or Other Document Type) based on the classification provided in the combined summary. State the document type before proceeding with the verification.
</document_type_check>

<verification_guidelines>
1. Systematic Comparison:
   • Create a checklist of all important points from both original summaries.
   • Systematically check each point against the combined summary, marking items as present or missing.

2. Information Preservation:
   • Ensure that ALL important details from both summaries are accurately incorporated into the combined summary.

3. Missing Information Addition:
   • For any information found missing during the review, explicitly add it to the verified summary. 
</verification_guidelines>

<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:
b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, charges, or arguments
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 

For each type of essential information, be specific when referring to people, places, and dates. 
</legal_document_essential_information>

<other_document_type_essential_information>
If the document is classified as another document type, ensure the summary includes the following elements, if present:
b. Main topics or themes
c. Key individuals or organizations mentioned
d. Important dates, events, or milestones
e. Significant data, statistics, or findings
f. Main arguments or conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context
</other_document_type_essential_information>

<thinking_process>
During the verification process, consider:
1. Have I created a comprehensive checklist of all important points from both original summaries?
2. Am I systematically comparing each point to ensure nothing is missed?
3. After my initial review, did I re-read the original summaries to catch any overlooked details?
4. If I found any missing information, have I added it to the verified summary with proper source attribution?
</thinking_process>

<warnings>
- Prioritize completeness over conciseness. 
- Do not alter the meaning or context of any information during the verification process.
- Ensure that all ambiguities or uncertainties from the original summaries are maintained.
- Do not add speculative information or draw conclusions not present in the original summaries or combined summary.
- If significant information is missing or inaccurately represented, be prepared to recommend a re-combination of the summaries.
</warnings>


<reference_materials>
Current Combined Summary:
{current_combined_summary}

Summary 1:
{summary_1}

Summary 2:
{summary_2}
</reference_materials>

<output_instruction>
First, confirm the document type (Legal Document or Other Document Type) based on the classification in the combined summary. Then, provide the verified summary below. If no changes are needed, return the contents of the Updated Summary. Present the summary in a clear, bullet-point format, organized chronologically for legal documents or thematically for other document types. Do not include any reference to your verification check in the final output.

## Verified Summary: ##
</output_instruction>
"""

condense_template = """
<task_description>
As a Legal Clerk, your task is to condense the summaries into a single summary.
</task_description>

<essential_information>
Ensure the condensed summary includes ALL of the following elements (if present in the summary). First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summaries that you are condensing. 

Some essential information that will contribute to a comprehensive summary include but are not limited to:
b. Primary parties involved (full names, roles, badge numbers if applicable)
j. Allegations of misconduct and any associated information
c. Key legal issues, claims, charges, or arguments
k. Disciplinary outcomes or their current status
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 
For each type of essential information classification, be specific when referring to people, places, and dates. 
</essential_information>

<thinking_process>
Before condensing the summary, consider:
1. What are the most critical pieces of information that must be retained?
2. How can I organize the information to present a clear summary?
4. How can I ensure that the condensed summary remains coherent and comprehensive?
5. Are there any redundancies in the merged summary that can be eliminated?
</thinking_process>

<warnings>
- Do not introduce new information not present in the merged summary
- Avoid altering the meaning or context of any information during the condensing process
- Do not omit any essential details, even if struggling to meet the 5-paragraph limit
- Ensure that all information remains accurately attributed to the correct parties and events
- Be cautious of potential inconsistencies and address them appropriately
- Do not include speculative or inferential information
</warnings>

<reference_materials>
## Input Summary ##
{summaries}
</reference_materials>

<output_instruction>
Provide the condensed summary below, ensuring that all essential information from the merged summary is retained, accurately presented, and organized in a clear, chronological, and logical manner. The condensed summary should not exceed 5 paragraphs:
</output_instruction>
"""

improve_summary_template = """
As a Legal Clerk, your task is to review and potentially improve the final summary of a legal document. You will be provided with the current summary and a memory log containing additional information from the document processing. Your goal is to enhance the summary by incorporating any relevant missing information from the memory log.

<task description>
Guidelines:
1. Review the current summary and the memory log carefully.
2. Identify any significant information in the memory log that is not present in the  summary.
3. If you find relevant missing information, integrate it into the summary while maintaining its concise nature (1-5 paragraphs).
4. Ensure that the additional information genuinely enhances the summary's comprehensiveness and relevance.
5. Maintain the original structure and flow of the summary as much as possible.
6. If no significant improvements are needed, return the original summary unchanged.
7. Condense any information that can be more concise. 
<task description>


<essential information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
b. Primary parties involved (full names, roles, badge numbers if applicable)
j. Allegations of misconduct and any associated information
c. Key legal issues, claims, charges, or arguments
k. Disciplinary outcomes or their current status
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 

For each type of essential information classification, be specific when referring to people, places, and dates. 
<essential information>

<output_format>
Present the summary using the following structure:
- Main topic 1
  • Sub-topic 1.1
  • Sub-topic 1.2
- Main events
  • Sub-event 2.1
  • Sub-event 2.2
- Main persons
  • Sub-person 2.1
  • Sub-person 2.2
- Main actions
  • Sub-action 2.1
  • Sub-action 2.2
</output_format>

<reference documents>
Summary:
{summary}

Memory Log:
{memory_log}
</reference documents>

Please provide the improved summary below. If no improvements are needed, simply reproduce the original summary:

Improved summary:
"""


organization_template = """
<task_description>
As a Legal Clerk, your task is to reorganize the content from concatenated summaries into a single, coherent, and comprehensive summary. 
</task_description>

<essential_information>
Ensure the reorganized summary includes ALL unique elements from the concatenated summaries. Your objective is to provide a thorough understanding of the case without repetition. Focus on:

a. All unique parties involved (with full details where available)
b. All distinct allegations of misconduct
c. All unique legal issues, claims, or charges
d. All critical events or incidents (without repeating duplicate entries)
e. All findings or decisions (consolidating where appropriate)
f. All significant evidence or testimonies (without repetition)
g. All unique outcomes, rulings, or disciplinary actions
h. Current status of all aspects of the case
i. Any unique pending actions or future proceedings
j. All distinct procedural events

When consolidating information, ensure that no unique details are lost in the process.
</essential_information>

<output_format>
Present the reorganized summary using the following structure:
1. Legal Issues, Claims, and Charges
2. Key Events and Incidents
3. Main Findings and Decisions

- Ensure each section contains all relevant unique information from the concatenated summaries. 
- Ensure that each bulletpoint is organized into a coherent bulletpoint that is unambiguous and clearly connects to other bulletpoints in a logical and coherent manner. 
</output_format>

<thinking_process>
When reorganizing the summary:
1. Have I identified and merged all duplicate information?
2. Have I created a chronological timeline of events, eliminating repetitions?
3. Have I ensured that the merged information remains accurate and context is preserved?
</thinking_process>

<warnings>
- Carefully review the entire concatenated summary before starting reorganization.
- Do not omit any unique information present in the original concatenated summary.
- When merging duplicate information, ensure all unique details from each instance are retained.
- If there are any contradictions in the duplicated information, include both versions and note the discrepancy.
- Maintain the original meaning and context of all information during reorganization.
</warnings>

<reference_materials>
## Concatenated Summary ##
{summaries}
</reference_materials>

<output_instruction>
Provide the reorganized summary below. Ensure that all unique information from the concatenated summary is retained, duplicate information is consolidated, and the result is presented in a clear, logical manner. The length of the output may vary depending on the amount of unique information present - focus on completeness rather than brevity.
</output_instruction>
"""

def combine_and_verify(summary_1, summary_2, memory_log):
    condense_llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    condense_prompt_template = ChatPromptTemplate.from_template(condense_template)

    combiner_llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    combine_prompt_template = ChatPromptTemplate.from_template(final_combine_template)
    combine_chain = combine_prompt_template | combiner_llm | StrOutputParser()

    verification_llm = ChatAnthropic(
        model_name="claude-3-haiku-20240307", temperature=0
    )
    verification_prompt_template = ChatPromptTemplate.from_template(
        final_verification_template
    )
    verification_chain = (
        verification_prompt_template | verification_llm | StrOutputParser()
    )

    improve_llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    improve_prompt_template = ChatPromptTemplate.from_template(improve_summary_template)

    improve_chain = improve_prompt_template | improve_llm | StrOutputParser()

    combined_summary = combine_chain.invoke(
        {"summary_1": summary_1, "summary_2": summary_2}
    )

    verified_summary = verification_chain.invoke(
        {
            "summary_1": summary_1,
            "summary_2": summary_2,
            "current_combined_summary": combined_summary,
        }
    )

    # improved summary
    improved_summary = improve_chain.invoke(
        {"summary": verified_summary, "memory_log": memory_log}
    )
    return improved_summary


def final_summarization(summaries, memory_log):
    if len(summaries) % 2 != 0:
        raise ValueError("The number of input summaries must be even.")

    result = []
    mid = len(summaries) // 2

    first_half = combine_and_verify(summaries[0], summaries[1], memory_log)
    for i in range(2, mid):
        first_half = combine_and_verify(first_half, summaries[i], memory_log)
    result.append(first_half)

    second_half = combine_and_verify(summaries[mid], summaries[mid + 1], memory_log)
    for i in range(mid + 2, len(summaries)):
        second_half = combine_and_verify(second_half, summaries[i], memory_log)
    result.append(second_half)

    return result


def organize_final_summary(summaries):
    condense_llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    condense_prompt_template = ChatPromptTemplate.from_template(organization_template)
    condense_chain = condense_prompt_template | condense_llm | StrOutputParser()

    combined_summary = "\n\n".join(summaries)
    summary = condense_chain.invoke({"summaries": combined_summary})

    return summary


def save_final_summaries_to_file(final_summaries, condensed_summary, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*50}\n")
        f.write("Final Summaries\n")
        f.write(f"{'='*50}\n\n")

        f.write("First Half Summary:\n")
        f.write(f"{'-'*20}\n")
        f.write(final_summaries[0])
        f.write("\n\n")

        f.write("Second Half Summary:\n")
        f.write(f"{'-'*20}\n")
        f.write(final_summaries[1])
        f.write("\n\n")

        f.write(f"{'='*50}\n")
        f.write("Condensed Final Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(condensed_summary)

    print(f"Final summaries and condensed summary saved to {output_file_path}")


def save_summaries_to_file(summaries, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as f:
        for i, summary in enumerate(summaries, 1):
            f.write(f"{'='*50}\n")
            f.write(f"Batch {i} Summary\n")
            f.write(f"{'='*50}\n\n")

            f.write(f"Pages: {summary['page_numbers']}\n\n")
            f.write(summary["page_content"])
            f.write("\n\n")


def process_summaries(summaries, memory_log):
    combined_summary, updated_memory_log = combine_summaries(summaries, memory_log)
    return combined_summary, updated_memory_log


def save_memory_log_to_text(memory_log, output_file_prefix):
    memory_log_file = f"{output_file_prefix}_memory_log.txt"
    with open(memory_log_file, "w") as f:
        f.write(memory_log)

if __name__ == "__main__":
    input_directory = "../../ocr/data/output/test"
    output_directory = "../data/output"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_start_time = time.time()

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            print(f"Processing file: {filename}")
            start_time = time.time()

            json_path = os.path.join(input_directory, filename)
            docs = load_and_split(json_path)

            memory_log = create_memory_log(docs)
            memory_path = os.path.join(
                output_directory,
                f"{timestamp}_{os.path.splitext(filename)[0]}_memory_log.txt",
            )
            save_memory_log_to_text(memory_log, memory_path)

            file_summaries = process_file(
                filename, input_directory, output_directory, memory_log
            )

            intermediate_summaries_path = os.path.join(
                output_directory,
                f"{timestamp}_{os.path.splitext(filename)[0]}_intermediate_summaries.txt",
            )
            save_summaries_to_file(file_summaries, intermediate_summaries_path)

            final_summary = final_summarization(file_summaries, memory_log)
            condensed_summary = organize_final_summary(final_summary)
            final_summary_file_path = os.path.join(
                output_directory,
                f"{timestamp}_{os.path.splitext(filename)[0]}_final_summary.txt",
            )
            save_final_summaries_to_file(
                final_summary, condensed_summary, final_summary_file_path
            )

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Completed processing file: {filename}")
            print(f"Time taken to process {filename}: {execution_time:.2f} seconds")

    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    print("All files processed successfully.")
    print(f"Total time taken to process all files: {total_execution_time:.2f} seconds")