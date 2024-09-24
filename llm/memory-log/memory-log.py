import os
import logging
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from collections import namedtuple
from datetime import datetime
import time
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import find_dotenv, load_dotenv
import concurrent.futures
from functools import partial

load_dotenv(find_dotenv())

Doc = namedtuple("Doc", ["page_content", "metadata"])

# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="", temperature=0)

llm = ChatAnthropic(model_name="claude-3-haiku-20240307",  temperature=0)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


memory_log_template = """
As a Legal Clerk, your task is to review the new summary and update the memory log only when the new summary contains crucial information directly related to the main subject of the document.
Be advised that the document can contain multiple main subjects. 
Maintain a concise memory log that focuses on the key aspects of the persons, events, actions, allegations, investigations, and outcomes described in the document.
</task_description>

<guidelines>
1. Review and Compare:
   • Carefully review the current memory log and the new summary.
   • Determine if the new summary contains crucial information related to the main subject(s) of the document that is not already in the memory log.

2. Identify Crucial Information:
   • Focus on information specific to the main subject(s) of the document.
   • Look for key details related to persons, events, actions, allegations, investigations, and outcomes.

3. Update Selectively:
   • Only update the memory log if the new summary contains crucial information not already present.
   • If updating, integrate the new information seamlessly into the existing log. Do not remove any unique information from the initial memory log. 

4. Maintain Conciseness:
   • Keep the memory log focused and concise.
   • Avoid redundancy or unnecessary details.

5. Ensure Accuracy:
   • Only include information that is directly stated in the document.
   • Do not infer or speculate beyond what is explicitly mentioned.

6. Preserve Original Structure:
   • If no update is necessary, return the original memory log with all of the unique elements in place.
</guidelines>

<essential_information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

1. All important persons involved and their roles and status.
1.1 All unique key legal and procedural policy violations and policy infractions related to these persons.
1.2 All unique key legal or procedural outcomes of those policy violations and infractions related to these persons.  

For each type of essential information classification, be specific when referring to people, places, and dates. 

Organize your output under the following headers:

1. Main Legal or Procedural Events, Incidents and Procedural Events:

2. Main Legal or Procedural Policy Violations or Policy Infractions: 

3. Main Legal or Procedural Outcomes and Disciplinary Actions:


Do not add information to these headers unless they belong to the main subject of the document.
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


summary_template_for_memory_log = """
<task_description>
As a Legal Clerk, your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page, using the context from the memory log and surrounding pages when necessary for clarity or relevance.
</task_description>

<reference_materials>
## Current Page ##
{current_page}
</reference_materials>

<guidelines>
1. Extract all essential information from the current page.
4. DO NOT include any details not explicitly stated in any of the documents.
5. Present the summary in a bullet point format, using subheadings to organize distinct aspects of the information.
6. If the someone's identity is ambiguous, refer to them as "unidentified person". 
7. If some of the information can not be summarized with confidence of its correctness, omit it from your summary.
</guidelines>

<essential_information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
1. Primary parties involved (full names, roles, badge numbers if applicable)
2. Key legal or procedural  issues, claims, charges, or allegations of misconduct, 
3. Key legal or procedural findings or decisions, such as outcomes, actions, disciplinary outcomes or their current status
4. Critical events or incidents (with specific dates, times and locations)
5. Significant evidence or testimonies
6. Current status of the matter or any pending actions or future proceedings

For each type of essential information category, be specific when referring to people, places, and dates. 
</essential_information>

<thinking_process>
Before summarizing, consider:
1. What are the main topics on this page?
2. How do these topics relate to a coherent narrative?
</thinking_process>

<output_format>
Present the summary using the following structure:
- Main topic 1
  • Subtopic 1.1
  • Subtopic 1.2
- Main topic 2
  • Subtopic 2.1
  • Subtopic 2.2
- Main topic 3
  • Subtopic 3.1
  • Subtopic 3.2
</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
</warnings>

<output_instruction>
Generate the current page summary below:
</output_instruction>
"""


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

    if "messages" in data and data["messages"]:
        docs = []
        original_page_number = 1
        for message in data["messages"]:
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
    

def create_memory_log(docs, max_workers=2, delay=1):
    memory_log = ""

    def process_and_update(i, doc, next_doc, current_memory_log):
        page_content = doc.page_content.replace("\n", " ")
        next_page_content = next_doc.page_content.replace("\n", " ") if next_doc else ""
        
        if "No data to be processed on this page" not in page_content:
            time.sleep(delay)  # Add delay before processing
            summary = process_memory_log_page(docs, i, page_content, 0, current_memory_log)["page_content"]
            
            if next_doc and "No data to be processed on this page" not in next_page_content:
                time.sleep(delay)  # Add delay before processing next page
                next_summary = process_memory_log_page(docs, i+1, next_page_content, 0, current_memory_log)["page_content"]
                combined_summary = f"{summary}\n\n{next_summary}"
            else:
                combined_summary = summary
            
            time.sleep(delay)  # Add delay before updating memory log
            updated_memory_log = update_memory_log(current_memory_log, combined_summary)
            print(f"Updated Memory Log after page {i + 1}" + (f" and {i + 2}" if next_doc else ""))
            return updated_memory_log
        else:
            print(f"Skipping page {i + 1} as it has no data to be processed")
            return current_memory_log

    def parallel_process(docs, start, end, current_memory_log):
        if start == end:
            return process_and_update(start, docs[start], None, current_memory_log)
        elif start + 1 == end:
            return process_and_update(start, docs[start], docs[end], current_memory_log)

        mid = (start + end) // 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future1 = executor.submit(parallel_process, docs, start, mid, current_memory_log)
            future2 = executor.submit(parallel_process, docs, mid + 1, end, current_memory_log)
            result1 = future1.result()
            result2 = future2.result()

        time.sleep(delay)  # Add delay before final memory log update
        return update_memory_log(result1, result2)

    return parallel_process(docs, 0, len(docs) - 1, memory_log)


def process_memory_log_page(docs, i, current_page, window_size, memory_log):
    prompt_response = ChatPromptTemplate.from_template(summary_template_for_memory_log)
    response_chain = prompt_response | llm | StrOutputParser()

    page_number = docs[i].metadata.get("seq_num")
    processed_content = response_chain.invoke(
        {
            "current_page": current_page,
        }
    )
    return {"page_content": processed_content, "page_number": page_number}

def update_memory_log(memory_log, new_summary):
    memory_log_prompt = ChatPromptTemplate.from_template(memory_log_template)
    memory_log_chain = memory_log_prompt | llm | StrOutputParser()

    updated_memory_log = memory_log_chain.invoke(
        {"summary": new_summary, "memory_log": memory_log}
    )
    print(f"UPDATED MEMORY LOG: {updated_memory_log}")

    return updated_memory_log

if __name__ == "__main__":
    start_time = time.time()

    input_directory = "data/input"
    output_directory = "data/output/"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            print(f"Processing file: {filename}")

            json_path = os.path.join(input_directory, filename)
            docs = load_and_split(json_path)

            memory_log = create_memory_log(docs)
            memory_path = os.path.join(
                output_directory,
                "memory_log.txt",
            )

            # Save the memory log to a file
            with open(memory_path, "w") as f:
                f.write(memory_log)

            print(f"Memory log saved to: {memory_path}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")