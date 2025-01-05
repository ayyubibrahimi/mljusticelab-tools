import os
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from collections import namedtuple
import re

Doc = namedtuple("Doc", ["page_content", "metadata"])

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def get_api_key():
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError("API key not set in environment variables")
    return api_key

api_key = get_api_key()
llm = ChatAnthropic(model_name="claude-3-haiku-20240307", api_key=api_key, temperature=0)

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
            print(docs)

            logger.info(f"Data loaded from document: {file_path}")
            return docs


memory_log_template = """
As a Legal Clerk, update the memory log only when the new summary contains crucial information directly related to the main subject of the document. Maintain a concise memory log that focuses on the key aspects of the events, allegations, investigations, and outcomes described in the document.

## Guidelines ##

1. Review the current memory log and new summary to determine if an update is necessary.

2. If the new summary contains crucial information specific to the main subject of the document, identify the key details to include.

3. ONLY INCLUDE INFORMATION ABOUT THE FOLLOWING KEY DETAILS:
   - The specific allegations, charges, or rule violations
   - Key events and actions during any investigations
   - Important evidence or findings
   - Legal proceedings, motions, or resolutions
   - Disciplinary actions or outcome

4. DO NOT infer any details that are not explicitly stated in the source document. Only return information contained within the document. You can do this by returning the exact language and phrases used. 

5. Reproduce the updated memory log or the original memory log, if it does not need to be updated with information from the new summary. 

IMPORTANT
Provide your response in a format similar to the examples below. Limit the memory log to these categories:

**Example**:
Incident Overview:

Details of Alleged Misconduct:

Investigation Findings:

Recommended Disciplinary Action:

## Original Memory Log ##: {memory_log}

## New Summary ##: {summary}

## Original Memory Log or Updated Memory Log ##:
"""

memory_log_verification_template = """
As a Legal Clerk, compare the old memory log with the updated memory log to ensure that no important information specific to the main subject of the document has been accidentally deleted in the update process. The memory log should serve as a concise summary of the key aspects of the events, allegations, investigations, and outcomes described in the document.

Guidelines:
1. Carefully review both the old and updated memory logs.

2. Identify any crucial information present in the old memory log that is missing from the updated memory log, focusing on details directly related to the main subject of the document.

3. Reproduce the updated memory log if it does not need to be updated with information from the original memory log. 

4. DO NOT infer any details that are not explicitly stated in the source document. Only return information contained within the document. You can do this by returning the exact language and phrases used. 

## Original Memory Log ##: {old_memory_log}

## Updated Memory Log ##: {updated_memory_log}

## Updated Memory Log or New Memory Log##:
"""


def update_memory_log(memory_log, new_summary):
    memory_log_prompt = ChatPromptTemplate.from_template(memory_log_template)
    memory_log_chain = memory_log_prompt | llm | StrOutputParser()
    updated_memory_log = memory_log_chain.invoke(
        {"summary": new_summary, "memory_log": memory_log}
    )
    memory_log_verification_prompt = ChatPromptTemplate.from_template(
        memory_log_verification_template
    )
    memory_log_verification_chain = (
        memory_log_verification_prompt | llm | StrOutputParser()
    )
    final_memory_log = memory_log_verification_chain.invoke(
        {"old_memory_log": memory_log, "updated_memory_log": updated_memory_log}
    )
    return final_memory_log


def create_memory_log(docs, custom_template, pages_to_concatenate=2):
    memory_log = ""
    num_pages = len(docs)

    # Helper function to concatenate specified number of pages
    def concatenate_pages(start_idx, num_pages_to_concat):
        combined_content = ""
        for j in range(num_pages_to_concat):
            if start_idx + j < num_pages:
                combined_content += (
                    docs[start_idx + j].page_content.replace("\n", " ") + " "
                )
        return combined_content.strip()

    # Process the first pages based on the concatenation length
    for i in range(0, min(10, num_pages), pages_to_concatenate):
        combined_content = concatenate_pages(i, pages_to_concatenate)
        summary = process_memory_log_page(docs, i, combined_content, 0, memory_log, custom_template)[
            "page_content"
        ]
        memory_log = update_memory_log(memory_log, summary)
        print(f"ORIGINAL MEMORY LOG {memory_log}")

    # Process the last pages (skipping the first ones if already processed)
    start_index = max(10, num_pages - 10)
    for i in range(start_index, num_pages, pages_to_concatenate):
        combined_content = concatenate_pages(i, pages_to_concatenate)
        summary = process_memory_log_page(docs, i, combined_content, 0, memory_log, custom_template)[
            "page_content"
        ]
        memory_log = update_memory_log(memory_log, summary)
        # print(f"UPDATED MEMORY LOG {memory_log}")

    return memory_log


def process_memory_log_page(docs, i, current_page, window_size, memory_log, custom_template):
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
            {   "custom_template": custom_template,
                "memory_log": memory_log,
                "previous_page_ending": previous_page_ending,
                "current_page": current_page,
                "next_page_beginning": next_page_beginning,
            }
        )
        response["page_content"] = processed_content

    return response


summary_template = """
As a Legal Clerk, your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page, using the context from the memory log and surrounding pages when necessary for clarity or relevance. Follow these guidelines to create an accurate and thorough summary:

## Guidelines ##

{custom_template}

2. Support the extracted data with additional context from the memory log and surrounding pages to enhance the understanding or relevance of the information in the current page. The memory log should be used to help you understand what is relevant and what is irrelevant. 

4. DO NOT include any details not explicitly stated in any of the documents.

5. Present the summary in a bullet point format, using subheadings if needed to organize distinct aspects of the information.

### Memory Log and Documents To Review ###

## Previous Page Ending ##:

{previous_page_ending}

## Next Page Beginning ##:

{next_page_beginning}

## Current Page ##:

{current_page}

### Current Page Summary:
"""

def process_page(docs, i, query, window_size, memory_log, pages_per_chunk, custom_template):
    prompt_response = ChatPromptTemplate.from_template(summary_template)
    response_chain = prompt_response | llm | StrOutputParser()

    current_pages = []
    page_numbers = []
    for j in range(pages_per_chunk):
        if i + j < len(docs):
            current_page = docs[i + j].page_content.replace("\n", " ")
            current_pages.append(current_page)
            page_number = docs[i + j].metadata.get("seq_num")
            page_numbers.append(page_number)

    previous_page_ending = (
        docs[i - 1].page_content.replace("\n", " ")[-window_size:] if i > 0 else ""
    )
    next_page_beginning = (
        docs[i + pages_per_chunk].page_content.replace("\n", " ")[:window_size]
        if i + pages_per_chunk < len(docs)
        else ""
    )

    response = {"page_content": "", "page_numbers": page_numbers}
    if current_pages:
        processed_content = response_chain.invoke(
            {   "custom_template": custom_template,
                "previous_page_ending": previous_page_ending,
                "current_page": " ".join(current_pages),
                "next_page_beginning": next_page_beginning,
            }
        )
        print(f"Processed Content: {processed_content}")
        response["page_content"] = processed_content

    return response


def process_batch(
    batch_docs, batch_start, query, window_size, memory_log, pages_per_chunk, custom_template
):
    sorted_results = []
    for i in range(0, len(batch_docs), pages_per_chunk):
        result = process_page(
            batch_docs, i, query, window_size, memory_log, pages_per_chunk, custom_template
        )
        sorted_results.append(result)

    section_summaries = {"messages": sorted_results}
    combined_summaries, _ = combine_summaries([section_summaries], memory_log)
    start_page = sorted_results[0]["page_numbers"][0]
    end_page = sorted_results[-1]["page_numbers"][-1]

    return combined_summaries[0], start_page, end_page


def generate_summaries(
    docs, query, memory_log, custom_template, window_size=100, batch_size=12, pages_per_chunk=2
):
    batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]
    combined_summaries = []

    with ThreadPoolExecutor() as executor:
        future_to_batch = {
            executor.submit(
                process_batch,
                batch,
                i * batch_size,
                query,
                window_size,
                memory_log,
                pages_per_chunk,
                custom_template
            ): i
            for i, batch in enumerate(batches)
        }

        results = []
        for future in as_completed(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                result = future.result()
                results.append((batch_index, result))
            except Exception as exc:
                print(f"Batch {batch_index} generated an exception: {exc}")

        # Sort results by batch_index to ensure order
        results.sort(key=lambda x: x[0])
        combined_summaries = [result for _, result in results]

    return combined_summaries


combine_template = """
As a Legal Clerk, your task is to concatenate the provided page summaries into a single, comprehensive, and well-organized summary for the given section of the police report. This summary should present all relevant information from both the current combined summary and the new page summary in a detailed, chronological, and coherent manner without any duplication. 

## Guidelines ##:

1. Comprehensive Information Integration:
   - Review the current combined summary and the new page summary to extract the most important information that is relevant to producing a summary. 
   - Include full names, roles, badge numbers, specific events with dates, policy or rule violations, disciplinary actions, relevant evidence, legal actions, and outcomes related to the case from both sources.

3. Narrative Coherence:
    Support the extracted data with additional context from the memory log and surrounding pages to enhances the understanding or relevance of the information in the current page. The memory log should be used to help you understand what is relevant and what is irrelevant. 
    
4. Handling Contradictions:
   - If inconsistencies arise between the current combined summary and the new page summary, prioritize the most detailed and specific information. If the information is incomplete, do not include it. 

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

## Original Bullet Point Summary ##:
{bulletpoint_summary}

Reorganized Bullet Point Summary:
"""

improvement_template = """
# Improved Summary Integration Prompt

As a Legal Clerk, your task is to review, integrate, and potentially improve the final condensed summary of a legal document. You will be provided with the current final condensed summary and all individual summaries from the document processing. Your goal is to enhance the condensed summary by incorporating any relevant missing information from the individual summaries, ensuring a comprehensive yet concise overview of the legal document.

## Guidelines:

1. Review the current final condensed summary and all individual summaries carefully.
2. Identify any significant information in the individual summaries that is not present in the final condensed summary.
3. If you find relevant missing information, integrate it into the condensed summary while maintaining its concise nature (1-5 paragraphs).
4. Ensure that the additional information genuinely enhances the summary's comprehensiveness and relevance.
5. Maintain the original structure and flow of the condensed summary as much as possible.
6. If no significant improvements are needed, return the original condensed summary unchanged.

## Key Elements to Include (if present in any of the summaries):

a. Type of legal document and its purpose
b. Primary parties involved (names and roles)
c. Key legal issues, claims, or charges
d. Critical events or incidents (with dates)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings

## Structural Integrity:

- Begin with an introductory sentence stating the type of document and its overall purpose.
- Organize the summary in paragraphs, ensuring a logical and chronological (if applicable) flow of information.

## Information Synthesis:

- Combine information to create a unified, non-redundant summary that provides a clear overview of the document.
- Ensure all critical details are accurately represented, without omitting any significant information present in any of the summaries.

## Clarity and Objectivity:

- Use clear, precise, and unambiguous language.
- Maintain an objective tone, avoiding interpretation or speculation.

## Handling Discrepancies:

- If there are conflicting pieces of information, include all versions with a brief note about the discrepancy.
- If information is unclear or seems incomplete in all summaries, note this uncertainty in the final summary.

## Contextual Relevance:

- Provide enough context for each point to be understood without referring to the full document.
- Ensure that the relationships between different pieces of information are clear.


## Updated Bullet Point Summary ##:
{coherent_summary}

## Original Bullet Point Summary ##:
{bulletpoint_summary}

## Return the Updated Bullet Point Summary or the Verified Bullet Point Summary Below ##:
"""



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
                }
            )
            # print(updated_combined_summary)

            verified_combined_summary = verification_chain.invoke(
                {
                    "updated_combined_summary": updated_combined_summary,
                    "current_combined_summary": current_combined_summary,
                    "new_page_summary": new_page_summary,
                }
            )
            print("VERIFIED COMBINED SUMMARY: {verified_combined_summary}")

            current_combined_summary = verified_combined_summary

            combined_page_numbers.extend(
                section_summaries["messages"][i].get(
                    "page_numbers",
                    [section_summaries["messages"][i].get("page_number")],
                )
            )

        improved_summary, improved_memory_log = format_and_improve_summary(
            current_combined_summary, summaries, memory_log
        )

        combined_summaries.append(
            {"page_content": improved_summary, "page_numbers": combined_page_numbers}
        )

    return combined_summaries, memory_log


def format_and_improve_summary(bulletpoint_summary, summaries, memory_log):
    # Format bulletpoint summary into coherent narrative
    prompt_response = ChatPromptTemplate.from_template(coherence_template)
    response_chain = prompt_response | llm | StrOutputParser()
    coherent_summary = response_chain.invoke(
        {"bulletpoint_summary": bulletpoint_summary}
    )
    coherent_memory_log = response_chain.invoke({"bulletpoint_summary": memory_log})

    # Improve coherent summary based on comparison with bulletpoint summary
    prompt_response = ChatPromptTemplate.from_template(improvement_template)
    response_chain = prompt_response | llm | StrOutputParser()
    improved_summary = response_chain.invoke(
        {
            "coherent_summary": coherent_summary,
            "bulletpoint_summary": bulletpoint_summary,
        }
    )

    improved_memory_log = response_chain.invoke(
        {"coherent_summary": coherent_memory_log, "bulletpoint_summary": memory_log}
    )

    return improved_summary, improved_memory_log

def clean_summary(summary):
    # Remove any introductory text followed by a colon
    cleaned_summary = re.sub(r'^[^:]+:\s*', '', summary.strip())
    
    # Split the summary into lines
    summary_lines = cleaned_summary.split('\n')
    
    # Remove any empty lines at the beginning
    while summary_lines and not summary_lines[0].strip():
        summary_lines.pop(0)
    
    # Join the lines back together
    return '\n'.join(summary_lines).strip()

def save_summaries_to_json(summaries, output_file):
    output_data = []
    for summary, start_page, end_page in sorted(summaries, key=lambda x: x[1]):
        cleaned_summary = clean_summary(summary["page_content"])
        output_data.append(
            {
                "sentence": cleaned_summary,
                "filename": os.path.basename(output_file),
                "start_page": start_page,
                "end_page": end_page,
            }
        )

    return {"files": output_data}

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Please provide the path to the directory, the selected model, and the output path as command-line arguments."
        )
        sys.exit(1)

    input_directory = sys.argv[1]
    selected_model = sys.argv[2]
    custom_template = sys.argv[3]
    output_path = sys.argv[4]
    output_data = []

    custom_template = str(custom_template)


    try:
        for entry in os.listdir(input_directory):
            entry_path = os.path.join(input_directory, entry)

            if os.path.isfile(entry_path) and entry.endswith(".json"):
                # Process individual JSON file
                docs = load_and_split(entry_path)
                memory_log = create_memory_log(docs, custom_template)
                query = "Generate a timeline of events based on the police report."
                combined_summaries = generate_summaries(docs, query, memory_log, custom_template)
                output_data.append(save_summaries_to_json(combined_summaries, entry))

            elif os.path.isdir(entry_path):
                # Process directory containing JSON files
                for filename in os.listdir(entry_path):
                    if filename.endswith(".json"):
                        input_file_path = os.path.join(entry_path, filename)

                        docs = load_and_split(
                            input_file_path
                        )  # Changed from entry_path to input_file_path
                        memory_log = create_memory_log(docs, custom_template)
                        query = (
                            "Generate a timeline of events based on the police report."
                        )
                        combined_summaries = generate_summaries(docs, query, memory_log, custom_template)
                        output_data.append(
                            save_summaries_to_json(combined_summaries, filename)
                        )

        # Convert the output data to JSON string
        with open(output_path, "w") as output_file:
            json.dump(output_data, output_file, indent=4)

    except Exception as e:
        logger.error(f"Error processing JSON: {str(e)}")
        print(json.dumps({"success": False, "message": "Failed to process JSON"}))
        sys.exit(1)
