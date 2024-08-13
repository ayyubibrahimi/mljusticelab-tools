import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_anthropic import ChatAnthropic
from collections import namedtuple
from dotenv import find_dotenv, load_dotenv
import logging
import re
import json
from datetime import datetime
import time
import os

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


load_dotenv(find_dotenv())

Doc = namedtuple("Doc", ["page_content", "metadata"])

llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)


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
    

summary_template = """
<task_description>
As a Legal Clerk, your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page, paying special attention to sentences wrapped in <principal component> tags, as these represent the most statistically significant points.
</task_description>
<guidelines>
1. Extract all essential information from the current page.
2. Pay extra attention to content within <principal component> tags.
3. DO NOT include any details not explicitly stated in the current page.
4. Present the summary in a bullet point format, using subheadings to organize distinct aspects of the information.
</guidelines>
<essential_information>
Ensure the summary includes ALL of the following elements, if present:
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

page_summary_verification_template = """
As a Legal Document Verifier, your task is to review the original document and its summary, then produce an enhanced summary that incorporates important missing information and corrects any inaccuracies.

<task description>
Guidelines:
1. Carefully review the original document and its existing summary.
2. Identify important information in the original document that is missing from the summary.
3. Check for any inaccuracies or misrepresentations in the summary.
4. Create an enhanced summary by:
   a. Adding missing important information
   b. Correcting any inaccuracies
   c. Preserving accurate existing content
5. Ensure that the enhanced summary is comprehensive, accurate, and relevant.
6. Maintain the original structure and flow of the summary as much as possible.
7. If no significant changes are needed, return the original summary with a note stating it's accurate and complete.
</task description>

<essential information>
When reviewing the original document, pay special attention to these elements if they're not already in the summary or are inaccurately represented:
a. Primary parties involved (full names, roles, badge numbers if applicable)
b. Key legal issues, claims, charges, or arguments
c. Critical events or incidents (with specific dates, times, and locations)
d. Main findings or decisions
e. Significant evidence or testimonies
f. Important outcomes or rulings
g. Current status of the matter
h. Any pending actions or future proceedings
i. Allegations of misconduct and any associated information
j. Disciplinary outcomes or their current status
k. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements)
For any new information added or corrections made, be specific when referring to people, places, and dates.
</essential information>

<thinking_process>
Before enhancing the summary, consider:
1. What important information is present in the original document but missing from the current summary?
2. Are there any inaccuracies or misrepresentations in the summary compared to the original document?
3. Where in the existing summary structure does new information or corrections best fit?
4. How can I integrate this information smoothly without disrupting the existing flow?
5. Does this new information or correction provide additional context or clarity to the existing summary?
</thinking_process>

<output_format>
Present the enhanced summary using the existing structure of the current summary.

Only add new information or corrections when they are necessary to improve the accuracy of the summary, clearly marking additions with [ADDED] tags and corrections with [CORRECTED] tags. For example:

Enhanced Summary:

- Main topic 1
  • Sub-topic 1.1
  • Sub-topic 1.2
  • [CORRECTED] Accurate sub-topic 1.3

- Main topic 2
  • Sub-topic 2.1
  • Sub-topic 2.2
  • [ADDED] Accurate sub-topic 2.3

- Main topic 3
  • Sub-topic 3.1
  • Sub-topic 3.2
  • Sub-topic 3.3 

Maintain this format throughout the summary, inserting new information or corrections where they fit best within the existing structure.
</output_format>

<reference documents>
Original Document:
{original_document}

Current Summary:
{current_summary}
</reference documents>

Please provide the enhanced summary below, with new information and corrections clearly marked.

Enhanced Summary:
"""

def pca_summarize(text, num_sentences=5):
    sentences = [sent.strip() for sent in text.split('.') if sent.strip()]
    if len(sentences) <= num_sentences:
        return text  # Return the original text if there are fewer sentences than requested

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Adjust the number of components based on the data
    n_components = min(num_sentences, len(sentences) - 1, tfidf_matrix.shape[1])
    
    if n_components == 0:
        return text  # Return the original text if we can't perform PCA

    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(tfidf_matrix.toarray())
    sentence_scores = np.sum(np.square(pca_matrix), axis=1)
    top_sentence_indices = sentence_scores.argsort()[-n_components:][::-1]
    
    # Create a dictionary to store the indices of principal components
    pc_indices = {i: True for i in top_sentence_indices}
    
    # Wrap principal components in tags
    tagged_sentences = []
    for i, sentence in enumerate(sentences):
        if i in pc_indices:
            tagged_sentences.append(f"<principal component>{sentence}</principal component>")
        else:
            tagged_sentences.append(sentence)
    
    return '. '.join(tagged_sentences) + '.'


def process_page(docs, i, pages_per_chunk):
    prompt_response = ChatPromptTemplate.from_template(summary_template)
    verification_prompt = ChatPromptTemplate.from_template(page_summary_verification_template)

    response_chain = prompt_response | llm | StrOutputParser()
    verification_chain = verification_prompt | llm | StrOutputParser()
    
    current_pages = []
    page_numbers = []
    pages_with_data = 0
    j = 0

    while pages_with_data < pages_per_chunk and i + j < len(docs):
        current_doc = docs[i + j]
        current_page = current_doc.page_content.replace("\n", " ")
        current_pages.append(current_page)
        page_numbers.append(current_doc.metadata.get("seq_num"))
        
        if current_page != "No data to be processed on this page":
            pages_with_data += 1
        
        j += 1

    response = {"page_content": "", "page_numbers": page_numbers}
    
    if current_pages:
        full_text = " ".join(current_pages)
        try:
            augmented_text = pca_summarize(full_text)
            print(f"Augmented Text: {augmented_text}")
        except Exception as e:
            print(f"Error in PCA summarization: {str(e)}")
            augmented_text = full_text  # Use the full text if PCA fails
        
        processed_content = response_chain.invoke({
            "current_page": augmented_text
        })

        verified_summary = verification_chain.invoke({
            "original_document": full_text,
            "current_summary": processed_content,
        })
        
        response["page_content"] = verified_summary
    
    return response

def process_document(docs, output_directory, base_filename):
    pages_per_chunk = 2
    verified_summaries = []

    i = 0
    while i < len(docs):
        result = process_page(docs, i, pages_per_chunk)
        page_range = f"Pages {result['page_numbers'][0]}-{result['page_numbers'][-1]}"
        verified_summaries.append(f"--- {page_range} ---\n{result['page_content']}\n\n")
        i += len(result['page_numbers'])

    # Write verified summaries to a file
    verified_summary_path = os.path.join(output_directory, f"{base_filename}_verified_summary.txt")
    with open(verified_summary_path, 'w', encoding='utf-8') as f:
        f.write("".join(verified_summaries))

if __name__ == "__main__":
    start_time = time.time()

    input_directory = "../../stage-1/data/output/spinoza"
    output_directory = "../data/output"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            print(f"Processing file: {filename}")

            json_path = os.path.join(input_directory, filename)
            docs = load_and_split(json_path)

            base_filename = os.path.splitext(filename)[0] + f"_{timestamp}"
            process_document(docs, output_directory, base_filename)

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")