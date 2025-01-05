import os
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from collections import namedtuple
import re

Doc = namedtuple("Doc", ["page_content", "metadata"])

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_key():
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError("API key not set in environment variables")
    return api_key

api_key = get_api_key()

llm = ChatAnthropic(model_name="claude-3-haiku-20240307", api_key=api_key, temperature=0)

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
                    page_content = format_content(page_content)
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


summary_template = """
<task_description>
As a Legal Clerk, your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page, using the context from the memory log and surrounding pages when necessary for clarity or relevance.
</task_description>

<guidelines>
1. Extract all essential information from the current page.
2. Support the extracted data with additional context from the memory log and surrounding pages to enhance the understanding or relevance of the information in the current page.
3. Use the memory log to help you understand what is relevant and what is irrelevant.
4. DO NOT include any details not explicitly stated in any of the documents.
5. Present the summary in a bullet point format, using subheadings to organize distinct aspects of the information.
6. If the someone's identity is ambiguous, refer to them as "unidentified person". 
7. If some of the information can not be summarized with confidence of its correctness, omit it from your summary.
</guidelines>

<essential_information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
{custom_template}

a. Type and purpose of the legal document (e.g., police report, internal investigation)
b. Primary parties involved (full names, roles, badge numbers if applicable)
j. Allegations of misconduct and any associated information
c. Key legal issues, claims, or charges
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
Before summarizing, consider:
1. What are the main topics on this page?
2. How does the information relate to previous pages?
3. What context from the memory log is relevant?
</thinking_process>

<output_format>
Present the summary using the following structure:
- Main topic 1
  • Subtopic 1.1
  • Subtopic 1.2
- Main topic 2
  • Subtopic 2.1
  • Subtopic 2.2
</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
</warnings>

<reference_materials>
## Current Page ##
{current_page}
</reference_materials>

<output_instruction>
Generate the current page summary below:
</output_instruction>
"""

improvement_template = """
<task_description>
As a Legal Clerk, your task is to review and improve the generated summary of a legal document page. You will be provided with the raw page content and the initially generated summary. Your goal is to create an improved summary that captures all important information accurately and presents it in a clear, bullet-point format.
</task_description>

<guidelines>
• Review the raw page content and the initial summary carefully.
• Ensure all important information from the raw page is included in the summary.
• Extract all essential information from the current page.
• Support the extracted data with additional context from the memory log and surrounding pages to enhance the understanding or relevance of the information in the current page.
• Use the memory log to help you understand what is relevant and what is irrelevant.
• DO NOT include any details not explicitly stated in any of the documents.
• Present the summary in a bullet point format, using subheadings to organize distinct aspects of the information.
• If someone's identity is ambiguous, refer to them as "unidentified person". 
• If some of the information cannot be summarized with confidence of its correctness, omit it from your summary.
• Ensure that the summary is accurate and not misleading.
</guidelines>

<essential_information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
{custom_template}

a. Type and purpose of the legal document (e.g., police report, internal investigation)
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

For each type of essential information classification, be specific when referring to people, places, and dates. 
</essential_information>

<thinking_process>
Before improving the summary, consider:
1. What are the main topics on this page?
2. How does the information relate to previous pages?
3. What context from the memory log is relevant?
4. Are there any gaps or inaccuracies in the initial summary?
5. How can the information be organized more effectively?
</thinking_process>

<output_format>
Present the improved summary using the following structure:
- Main topic 1
  • Subtopic 1.1
  • Subtopic 1.2
- Main topic 2
  • Subtopic 2.1
  • Subtopic 2.2
</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
- If no details are present for a particular topic, state that there are no details
</warnings>

<reference_materials>
## Raw Page Content ##
{raw_page}

## Initial Summary ##
{initial_summary}
</reference_materials>

<output_instruction>
Generate the improved page summary below:
</output_instruction>
"""


def process_page(doc, custom_template):
    if doc.page_content == "No data to be processed on this page":
        return {"page_content": "No data to be processed on this page", "page_number": doc.metadata.get("seq_num")}

    prompt_response = ChatPromptTemplate.from_template(summary_template)
    response_chain = prompt_response | llm | StrOutputParser()

    improvement_prompt = ChatPromptTemplate.from_template(improvement_template)
    improvement_chain = improvement_prompt | llm | StrOutputParser()

    current_page = doc.page_content.replace("\n", " ")
    page_number = doc.metadata.get("seq_num")

    processed_content = response_chain.invoke(
        {
            "custom_template": custom_template,
            "current_page": current_page,
        }
    )

    improved_summary = improvement_chain.invoke(
        {   
            "custom_template": custom_template,
            "raw_page": current_page,
            "initial_summary": processed_content,
        }
    )

    print(f"Processed Content: {processed_content}")

    return {"page_content": improved_summary, "page_number": page_number}

def generate_summaries(docs, custom_template):
    summaries = []

    with ThreadPoolExecutor(max_workers=35) as executor:
        future_to_page = {executor.submit(process_page, doc, custom_template): doc for doc in docs}
        
        # Process completed tasks
        for future in as_completed(future_to_page):
            doc = future_to_page[future]
            try:
                result = future.result()
                summaries.append(result)
            except Exception as exc:
                print(f'Page {doc.metadata.get("seq_num")} generated an exception: {exc}')
    
    summaries.sort(key=lambda x: x["page_number"])
    return summaries

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
    for summary in summaries:
        cleaned_summary = clean_summary(summary["page_content"])
        output_data.append(
            {
                "sentence": cleaned_summary,
                "filename": os.path.basename(output_file),
                "start_page": summary["page_number"],
                "end_page": summary["page_number"],
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
                query = "Generate a timeline of events based on the police report."
                combined_summaries = generate_summaries(docs, custom_template)
                output_data.append(save_summaries_to_json(combined_summaries, entry))

            elif os.path.isdir(entry_path):
                # Process directory containing JSON files
                for filename in os.listdir(entry_path):
                    if filename.endswith(".json"):
                        input_file_path = os.path.join(entry_path, filename)

                        docs = load_and_split(
                            input_file_path
                        )  # Changed from entry_path to input_file_path
                        combined_summaries = generate_summaries(docs, custom_template)
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
