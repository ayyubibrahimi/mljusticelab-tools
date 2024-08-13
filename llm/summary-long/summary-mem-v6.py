import os
import logging
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import multiprocessing
from collections import namedtuple
from functools import partial
from dotenv import find_dotenv, load_dotenv
import concurrent.futures
from datetime import datetime
import time
import re

load_dotenv(find_dotenv())

Doc = namedtuple("Doc", ["page_content", "metadata"])

llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)

# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

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

def load_and_split(file_path):
    logger.info(f"Processing document: {file_path}")

    with open(file_path, "r") as file:
        file_content = file.read()
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            data = {}

    logger.info(f"Keys in parsed JSON data: {data.keys()}")

    if "files" in data and data["files"]:
        docs = []
        original_page_number = 1
        for message in data["files"]:
            page_content = message.get("page_content", "")
            if word_count(page_content) >= 50:
                formatted_content = format_content(page_content)
                doc = Doc(
                    page_content=formatted_content,
                    metadata={"seq_num": original_page_number},
                )
            else:
                doc = Doc(
                    page_content="No data to be processed on this page",
                    metadata={"seq_num": original_page_number},
                )
            docs.append(doc)
            original_page_number += 1

        logger.info(f"Data loaded and formatted from document: {file_path}")
        return docs
    else:
        logger.warning(f"No valid data found in document: {file_path}")
        return []

memory_log_template = """
<task_description>
As a Legal Clerk, your task is to review the new summary and update the memory log only when the new summary contains crucial information that will contribute to a better understanding of the document.
</task_description>

<guidelines>
1. Review and Compare:
   • Carefully review the current memory log and the new summary.
   • Determine if the new summary contains crucial information that is not already in the memory log.

Update Selectively:
• Only update the memory log if the new summary contains crucial information not already present.
• If updating, integrate the new information seamlessly into the existing log.

Maintain Conciseness:
• Avoid redundancy or unnecessary details.

Ensure Accuracy:
• Only include information that is directly stated in the document.
• Do not infer or speculate beyond what is explicitly mentioned.
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
- Do not add information that is not directly stated in the document
- Avoid speculation or inference beyond what is explicitly mentioned
- Do not remove or alter existing crucial information in the memory log
- Ensure that any updates maintain the chronological and logical flow of events
- Be cautious of potential inconsistencies between the new summary and existing log
</warnings>

<reference_materials>
Original Memory Log
{memory_log}
New Summary
{summary}
</reference_materials>

<output_instruction>
Based on your review of the current memory log and the new summary, provide the full memory log. If updates are necessary, incorporate the crucial new information into the existing log. If no updates are needed, reproduce the original memory log in its entirety. In either case, ensure that the output maintains a concise focus on key aspects of events, allegations, investigations, and outcomes related to the main subject of the document.
Begin your response with one of the following statements:

If updates were made: "Updated Memory Log:"
If no updates were needed: "Original Memory Log (No updates needed):"

Then, provide the full text of the memory log.
</output_instruction>
"""


def create_memory_log(docs, pages_to_concatenate=2):
    memory_log = ""
    num_pages = len(docs)

    def concatenate_pages(start_idx, num_pages_to_concat):
        combined_content = ""
        current_idx = start_idx
        pages_concatenated = 0

        while pages_concatenated < num_pages_to_concat and current_idx < num_pages:
            page_content = docs[current_idx].page_content.replace("\n", " ")
            if "No data to be processed on this page" not in page_content:
                combined_content += page_content + " "
                pages_concatenated += 1
            current_idx += 1

        return combined_content.strip(), current_idx - start_idx

    def find_next_data_page(start_idx):
        current_idx = start_idx
        while current_idx < num_pages:
            content, pages_checked = concatenate_pages(current_idx, 1)
            if content:
                return current_idx
            current_idx += pages_checked
        return None

    # Process the first 10 pages with data
    start_idx = find_next_data_page(0)
    if start_idx is not None:
        pages_processed = 0
        while pages_processed < 5 and start_idx < num_pages:
            combined_content, pages_checked = concatenate_pages(start_idx, pages_to_concatenate)
            if combined_content:
                summary = process_memory_log_page(docs, start_idx, combined_content, 0, memory_log)["page_content"]
                memory_log = update_memory_log(memory_log, summary)
                print(f"Updated Memory Log: {memory_log}")
                pages_processed += 1
            start_idx += pages_checked

    # Process the last 10 pages with data
    end_idx = num_pages - 1
    pages_processed = 0
    while pages_processed < 5 and end_idx >= start_idx:
        last_data_idx = find_next_data_page(end_idx)
        if last_data_idx is None:
            break
        combined_content, pages_checked = concatenate_pages(last_data_idx, pages_to_concatenate)
        if combined_content:
            summary = process_memory_log_page(docs, last_data_idx, combined_content, 0, memory_log)["page_content"]
            memory_log = update_memory_log(memory_log, summary)
            pages_processed += 1
        end_idx = last_data_idx - 1

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
    max_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=max_cpus) as pool:
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
    max_workers = os.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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

    verification_llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    verification_prompt = ChatPromptTemplate.from_template(verification_template)
    verification_chain = verification_prompt | verification_llm | StrOutputParser()

    def combine_and_verify(summary_1, summary_2):
        combined_summary = combiner_chain.invoke({
            "summary_1": summary_1["page_content"],
            "summary_2": summary_2["page_content"]
        })
        verified_summary = verification_chain.invoke({
            "current_combined_summary": combined_summary,
            "summary_1": summary_1["page_content"],
            "summary_2": summary_2["page_content"]
        })
        return {
            "page_content": verified_summary,
            "page_numbers": summary_1["page_numbers"] + summary_2["page_numbers"]
        }

    def parallel_combine(summaries):
        if len(summaries) == 1:
            return summaries[0]

        mid = len(summaries) // 2
        left_half = summaries[:mid]
        right_half = summaries[mid:]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_left = executor.submit(parallel_combine, left_half)
            future_right = executor.submit(parallel_combine, right_half)

            combined_left = future_left.result()
            combined_right = future_right.result()

        return combine_and_verify(combined_left, combined_right)

    final_summary = parallel_combine(summaries)
    return final_summary, memory_log


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
As a Legal Clerk, your task is to enhance the existing summary of a legal document by incorporating important information from the memory log. The current summary contains specific details, while the memory log provides high-level, important information. 
Your goal is to improve the summary by adding only the most important missing information from the memory log without removing any existing content.

<task description>
Guidelines:
1. Carefully review both the current summary and the memory log.
2. Identify the most information in the memory log that is not present in the current summary.
3. Add the missing important information to the appropriate sections of the summary.
4. Preserve all existing content in the current summary.
5. Ensure that added information enhances the summary's comprehensiveness and relevance.
6. Maintain the original structure and flow of the summary as much as possible.
7. If no significant additions are needed, return the original summary unchanged.
</task description>

<essential information>
When reviewing the memory log, pay special attention to these elements if they're not already in the summary:
a. Primary parties involved (full names, roles, badge numbers if applicable)
b. Key legal issues, claims, charges, or arguments not mentioned in the summary
c. Critical events or incidents (with specific dates, times, and locations)
d. Main findings or decisions not covered in the summary
e. Significant evidence or testimonies missing from the summary
f. Important outcomes or rulings not mentioned
g. Updates to the current status of the matter
h. Any pending actions or future proceedings not included
i. Allegations of misconduct and any associated information
j. Disciplinary outcomes or their current status
k. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements)
For any new information added, be specific when referring to people, places, and dates.
</essential information>

<thinking_process>
Before adding to the summary, consider:
1. What important information is present in the memory log but missing from the current summary?
2. Where in the existing summary structure does this new information best fit?
3. How can I integrate this information smoothly without disrupting the existing flow?
4. Does this new information provide additional context or clarity to the existing summary?
</thinking_process>

<output_format>
Present the enhanced summary using the existing structure of the current summary. Add new information from the memory log where appropriate, clearly marking additions with [ADDED] tags. For example:

Enhanced section:
- Main topic 1
  • Sub-topic 1.1
  • Sub-topic 1.2
  • [ADDED] New sub-topic 1.3 from memory log

- Main topic 2
  • Sub-topic 2.1
  • Sub-topic 2.2
  • [ADDED] New sub-topic 2.3 from memory log

- Main topic 3
  • Sub-topic 3.1
  • Sub-topic 3.2
  • [ADDED] New sub-topic 3.3 from memory log

Maintain this format throughout the summary, inserting new information where it fits best within the existing structure.
</output_format>

<reference documents>
Current Summary:
{summary}

Memory Log:
{memory_log}
</reference documents>

Please provide the enhanced summary below, with new information clearly marked.

Enhanced Summary:
"""


organization_template = """
<task_description>
As a Legal Clerk, your task is to reorganize the content from concatenated summaries into a single, coherent, and comprehensive summary. Your primary goal is to eliminate redundancies while ensuring all unique information is retained.
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
3. Main Findings, Decisions and Actions
Ensure each section contains all relevant unique information from the concatenated summaries.
</output_format>

<thinking_process>
When reorganizing the summary:
1. Have I cross-referenced all information to ensure all unique details are captured? 
2. Have I identified and merged all duplicate information?
3. Have I created a chronological timeline of events, eliminating repetitions?
4. Have I consolidated all related information under appropriate headings?
5. Have I ensured that the merged information remains accurate and context is preserved?
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
    start_time = time.time()

    input_directory = "../../ocr/data/output/test"
    output_directory = "../data/output"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            print(f"Processing file: {filename}")

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

            print(f"Completed processing file: {filename}")

    end_time = time.time()
    total_time = end_time - start_time

    print("All files processed successfully.")
    print(f"Total execution time: {total_time:.2f} seconds")