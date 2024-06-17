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
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


llm = ChatAnthropic(model_name="claude-3-haiku-20240307", api_key="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", temperature=0)


nlp = spacy.load("en_core_web_lg")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="max"
)

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

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


def process_page(docs, i, query, window_size):
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
                "previous_page_ending": previous_page_ending,
                "current_page": current_page,
                "next_page_beginning": next_page_beginning,
            }
        )
        response["page_content"] = processed_content
    
    time.sleep(3)  # Add a 1-second delay between each page processing
    
    return response


def process_batch(batch_docs, batch_start, query, window_size):
    sorted_results = []
    for i in range(len(batch_docs)):
        result = process_page(batch_docs, i, query, window_size)
        sorted_results.append(result)

    section_summaries = {"messages": sorted_results}
    combined_summaries = combine_summaries([section_summaries])
    start_page = sorted_results[0]["page_number"]
    end_page = sorted_results[-1]["page_number"]
    
    time.sleep(3)  # Add a 3-second delay between each batch processing
    
    return combined_summaries[0], start_page, end_page


def generate_summaries(docs, query, window_size=500, batch_size=6, max_parallel_batches=3):
    batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
    combined_summaries = []

    for i in range(0, len(batches), max_parallel_batches):
        batch_range = batches[i:i+max_parallel_batches]
        pool = multiprocessing.Pool()
        results = pool.starmap(
            process_batch,
            [(batch, i * batch_size, query, window_size) for i, batch in enumerate(batch_range)],
        )
        pool.close()
        pool.join()

        combined_summaries.extend([result[:3] for result in results])
        time.sleep(5)  # Add a 5-second delay between processing each set of parallel batches

    return combined_summaries


# ### difference is that we're asking the model to preserve the original language 

summary_template = """
As an AI assistant, your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page, using the context from the surrounding pages when necessary for clarity or relevance. Follow these guidelines to create an accurate and thorough summary:

## Guidelines ##

1. Extract all important details from the current page, including but not limited to:
   - Individuals mentioned, including their full names, roles, badge numbers, and specific actions
   - Allegations, charges, or rule violations discussed, providing case numbers and exact dates when available
   - Main events, actions, and observations described, including precise dates and locations when provided
   - Relevant evidence or findings presented, quoting directly from the text when appropriate
   - Legal proceedings, motions, disciplinary actions, or outcomes mentioned, including specific dates, case law citations, and arguments made by involved parties

2. Use clear, concise language and maintain accuracy by quoting or closely paraphrasing the original text. Prioritize direct quotes for key statements or evidence.

3. If the current page lacks clarity or contains ambiguous information, note the ambiguity and provide the most likely interpretation based on the available context.

4. Present the summary in a bulletpoint format, using subheadings if needed to organize distinct aspects of the information.

5. Include all relevant details from the current page, even if they do not fall under the categories of key information outlined in the guidelines.

6. Avoid repetition of information from previous summaries, but include important details that may have been omitted from earlier summaries.

**Example Format 1**:


**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

- The document outlines the background of the case against Officer Adams, including the charges and his defense.
- Details of the early settlement conference are provided.
- The initial proposed settlement was rejected by the Police Commission, leading to a revised agreement.
- Adams admits to three specifications of misconduct, waives his right to further administrative remedies, and accepts the recommended discipline.
- The discipline includes a 15-day suspension (with 5 days held in abeyance) and mandatory retraining.
- Adams waives his right to judicial or administrative review of the agreement, including a waiver regarding the department's Brady Committee review.
- The agreement is signed by the parties and approved by the Police Commission.

**Example Format 2**:

**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

- The document describes the incident involving Officer Smith on January 15, 2020, where he was accused of using excessive force.
- Witnesses reported unprovoked strikes by Officer Smith during an arrest.
- The internal investigation concluded that Officer Smith violated department policy.
- Officer Smith was suspended for 30 days and required to undergo use-of-force retraining.
- The disciplinary action was reviewed and approved by the Police Commission.

## Previous Page Ending ##:

{previous_page_ending}

## Next Page Beginning ##:

{next_page_beginning}

## Current Page ##:

{current_page}

### Current Page Summary:
"""

combine_template = """
As an AI assistant, your task is to iteratively combine the provided page summaries into a single, comprehensive, and well-organized summary for the given section of the police report. This summary should present all relevant information from both the current combined summary and the new page summary in a detailed, chronological (when applicable), and coherent manner, closely following the format of the individual page summaries. Follow these guidelines:

## Guidelines ##

1. Comprehensive Information Integration:
   - Review the current combined summary and the new page summary to extract all relevant information, ensuring no important details are overlooked.
   - Include full names, roles, badge numbers, exact dates, specific events, relevant evidence, legal actions, and outcomes related to the case from both sources.
   - Include direct quotes for key statements, evidence, or arguments, when relevant.

2. Chronological Order (if applicable):
   - If the summaries cover a sequence of events, arrange the details in a clear chronological order, providing specific dates whenever available to enhance the narrative's flow and comprehension.

3. Narrative Coherence:
   - Ensure that the information from both the current combined summary and the new page summary is integrated seamlessly, maintaining a coherent narrative flow.

4. Comprehensive Coverage:
   - Ensure that the updated combined summary incorporates all relevant details from both the current combined summary and the new page summary, creating a comprehensive overview of the information presented in the police report.
   - If a detail is repeated across summaries, include it only once in the most appropriate context.

5. Handling Contradictions:
   - If inconsistencies arise between the current combined summary and the new page summary, prioritize the most detailed and specific information.
   - If contradictions cannot be resolved, note the discrepancy and present both versions of the information.

6. Formatting for Clarity:
   - Use bullet points to organize different aspects of the information effectively, maintaining a structure similar to the individual page summaries.
   - Ensure that the updated combined summary is well-structured and easy to follow, with a logical flow of information.

7. Factual Accuracy:
   - Include all details explicitly stated in both the current combined summary and the new page summary, noting any ambiguities or unclear information when necessary.
   - Avoid introducing any new information or making inferences not directly supported by the summaries.

**Example Format 1**:


**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

- The document outlines the background of the case against Officer Adams, including the charges and his defense.
- Details of the early settlement conference are provided.
- The initial proposed settlement was rejected by the Police Commission, leading to a revised agreement.
- Adams admits to three specifications of misconduct, waives his right to further administrative remedies, and accepts the recommended discipline.
- The discipline includes a 15-day suspension (with 5 days held in abeyance) and mandatory retraining.
- Adams waives his right to judicial or administrative review of the agreement, including a waiver regarding the department's Brady Committee review.
- The agreement is signed by the parties and approved by the Police Commission.

**Example Format 2**:

**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

- The document describes the incident involving Officer Smith on January 15, 2020, where he was accused of using excessive force.
- Witnesses reported unprovoked strikes by Officer Smith during an arrest.
- The internal investigation concluded that Officer Smith violated department policy.
- Officer Smith was suspended for 30 days and required to undergo use-of-force retraining.
- The disciplinary action was reviewed and approved by the Police Commission.

## Summaries ##

## Current Combined Summary ##:
{current_combined_summary}

## New Page Summary ##:
{new_page_summary}

## Updated Combined Summary: ##
"""


verification_template = """
As an AI assistant, your task is to review the updated combined summary, which integrates content from the current combined summary and the new summary of a police investigative report. This verification process aims to ensure that all relevant information from both the current combined summary and the new summary is accurately retained in the updated combined summary.

## Verification Guidelines ##

1. Comprehensive Information Integration:
   - Verify that all relevant information from the current combined summary is retained in the updated combined summary.
   - Ensure that all important details from the new summary, such as critical facts, key events, and significant details, are accurately incorporated into the updated combined summary.
   - Confirm that no crucial details, such as full names, roles, badge numbers, exact dates, and direct quotes, have been omitted during the combination process.

2. Consistency:
   - Check for any contradictions or discrepancies between the information from the current combined summary, the new summary, and the updated combined summary.
   - Ensure that the information from both sources is consistently presented in the updated combined summary, maintaining the same level of detail and accuracy.

3. Context Preservation:
   - Verify that the context of the information from both the current combined summary and the new summary is preserved in the updated combined summary.
   - Ensure that the integration of the new information does not alter the meaning or significance of the existing content in the combined summary.

4. Logical Flow:
   - Evaluate the updated combined summary for logical flow and coherence, ensuring that the newly integrated information fits seamlessly into the existing narrative.
   - Confirm that the addition of the new information does not disrupt the chronological order (when applicable) or the overall structure of the combined summary.

5. Factual Accuracy:
   - Confirm that the information from both the current combined summary and the new summary has been accurately integrated into the updated combined summary, without any alterations or misrepresentations.
   - Verify that no new information or inferences not directly supported by either source have been introduced during the combination process.

**Example Format 1**:

**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

- The document outlines the background of the case against Officer Adams, including the charges and his defense.
- Details of the early settlement conference are provided.
- The initial proposed settlement was rejected by the Police Commission, leading to a revised agreement.
- Adams admits to three specifications of misconduct, waives his right to further administrative remedies, and accepts the recommended discipline.
- The discipline includes a 15-day suspension (with 5 days held in abeyance) and mandatory retraining.
- Adams waives his right to judicial or administrative review of the agreement, including a waiver regarding the department's Brady Committee review.
- The agreement is signed by the parties and approved by the Police Commission.

**Example Format 2**:

**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

- The document describes the incident involving Officer Smith on January 15, 2020, where he was accused of using excessive force.
- Witnesses reported unprovoked strikes by Officer Smith during an arrest.
- The internal investigation concluded that Officer Smith violated department policy.
- Officer Smith was suspended for 30 days and required to undergo use-of-force retraining.
- The disciplinary action was reviewed and approved by the Police Commission.

## Current Combined Summary ##:
{current_combined_summary}

## New Summary ##:
{new_page_summary}

## Provide the updated combined summary below, ensuring that all relevant information from both the current combined summary and the new summary is accurately retained. If no updates are needed, return the current combined summary ##
"""

coherence_template = """
As an AI assistant, your task is to review the provided bullet point summary and reorganize it into a more coherent and well-structured format. Please follow these steps to improve the summary:

1. Carefully review the bullet point summary and identify all the points and key details, paying special attention to specific names, badge numbers, dates, locations, case numbers, and direct quotes.

2. Organize the bullet points in a chronological order, ensuring that the sequence of events is accurately represented and that all relevant details are included.

3. Ensure that the reorganized bullet points:
   - Present a clear, logical progression of events, with all relevant details included
   - Use concise and unambiguous language, with direct quotes where appropriate
   - Do not introduce any new information or infer details not explicitly stated in the original summary

**Example Format 1**:

Verified Bullet Point Summary:

**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

- The document outlines the background of the case against Officer Adams, including the charges and his defense.
- Details of the early settlement conference are provided.
- The initial proposed settlement was rejected by the Police Commission, leading to a revised agreement.
- Adams admits to three specifications of misconduct, waives his right to further administrative remedies, and accepts the recommended discipline.
- The discipline includes a 15-day suspension (with 5 days held in abeyance) and mandatory retraining.
- Adams waives his right to judicial or administrative review of the agreement, including a waiver regarding the department's Brady Committee review.
- The agreement is signed by the parties and approved by the Police Commission.

**Example Format 2**:

**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

Verified Bullet Point Summary:
- The document describes the incident involving Officer Smith on January 15, 2020, where he was accused of using excessive force.
- Witnesses reported unprovoked strikes by Officer Smith during an arrest.
- The internal investigation concluded that Officer Smith violated department policy.
- Officer Smith was suspended for 30 days and required to undergo use-of-force retraining.
- The disciplinary action was reviewed and approved by the Police Commission.

Please provide your revised bullet point summary below.

## Original Bullet Point Summary ##:
{bulletpoint_summary}

Reorganized Bullet Point Summary:
"""

improvement_template = """
As an AI assistant, your task is to produce a verified bullet point summary by comparing an updated bullet point summary with the original bullet point summary. The goal is to create a final summary that captures all the essential information from both summaries, ensuring accuracy, coherence, and logical structure. Please follow these steps:

1. Carefully review the bullet point summary and identify all the points and key details, paying special attention to specific names, badge numbers, dates, locations, case numbers, and direct quotes.

2. Organize the bullet points in a chronological order, ensuring that the sequence of events is accurately represented and that all relevant details are included.

3. Ensure that the reorganized bullet points:
   - Present a clear, logical progression of events, with all relevant details included
   - Use concise and unambiguous language, with direct quotes where appropriate
   - Do not introduce any new information or infer details not explicitly stated in the original summary

**Example Format 1**:


**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

- The document outlines the background of the case against Officer Adams, including the charges and his defense.
- Details of the early settlement conference are provided.
- The initial proposed settlement was rejected by the Police Commission, leading to a revised agreement.
- Adams admits to three specifications of misconduct, waives his right to further administrative remedies, and accepts the recommended discipline.
- The discipline includes a 15-day suspension (with 5 days held in abeyance) and mandatory retraining.
- Adams waives his right to judicial or administrative review of the agreement, including a waiver regarding the department's Brady Committee review.
- The agreement is signed by the parties and approved by the Police Commission.

**Example Format 2**:

**Note**: The example provided below is for illustrative purposes only. Do not use any of the information from the example in your summary. The example is a toy example meant to demonstrate the format and style of the summary.

- The document describes the incident involving Officer Smith on January 15, 2020, where he was accused of using excessive force.
- Witnesses reported unprovoked strikes by Officer Smith during an arrest.
- The internal investigation concluded that Officer Smith violated department policy.
- Officer Smith was suspended for 30 days and required to undergo use-of-force retraining.
- The disciplinary action was reviewed and approved by the Police Commission.

## Updated Bullet Point Summary ##:
{coherent_summary}

## Original Bullet Point Summary ##:
{bulletpoint_summary}

## Return the Updated Bullet Point Summary or the Verified Bullet Point Summary Below ##:
"""


def format_and_improve_summary(bulletpoint_summary, summaries):    
    # Format bulletpoint summary into coherent narrative
    prompt_response = ChatPromptTemplate.from_template(coherence_template)
    response_chain = prompt_response | llm | StrOutputParser()
    coherent_summary = response_chain.invoke({"bulletpoint_summary": bulletpoint_summary})
    
    # Improve coherent summary based on comparison with bulletpoint summary
    prompt_response = ChatPromptTemplate.from_template(improvement_template)
    response_chain = prompt_response | llm | StrOutputParser()
    improved_summary = response_chain.invoke({
        "coherent_summary": coherent_summary,
        "bulletpoint_summary": bulletpoint_summary
    })
    
    
    return improved_summary


def combine_summaries(summaries):
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

            verified_combined_summary = verification_chain.invoke(
                {
                    "updated_combined_summary": updated_combined_summary,
                    "current_combined_summary": current_combined_summary,
                    "new_page_summary": new_page_summary,
                }
            )

            current_combined_summary = verified_combined_summary
            combined_page_numbers.extend(
                section_summaries["messages"][i].get("page_numbers", [section_summaries["messages"][i].get("page_number")])
            )

        # Create a formatted string with the page numbers at the top
        page_range = f"Pages {min(combined_page_numbers)}-{max(combined_page_numbers)}"
        improved_summary = f"{page_range}\n{format_and_improve_summary(current_combined_summary, summaries)}"
        combined_summaries.append(improved_summary)

    return combined_summaries  # Return list of formatted summaries


def write_json_output(combined_summaries, filename):
    output_data = []
    for summary in combined_summaries:
        output_data.append({
            "sentence": summary,  # Append each summary separately
            "filename": filename
        })
    return output_data

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide the path to the directory and the selected model as command-line arguments.")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    selected_model = sys.argv[2]
    output_data = []

    try:
        for entry in os.listdir(input_directory):
            entry_path = os.path.join(input_directory, entry)
            
            if os.path.isfile(entry_path) and entry.endswith(".json"):
                # Process individual JSON file
                docs = load_and_split(entry_path)
                query = "Generate a timeline of events based on the police report."
                combined_summaries = generate_summaries(docs, query)
                output_data.extend(write_json_output(combined_summaries, entry))

            
            elif os.path.isdir(entry_path):
                # Process directory containing JSON files
                for filename in os.listdir(entry_path):
                    if filename.endswith(".json"):
                        input_file_path = os.path.join(entry_path, filename)
                        
                        docs = load_and_split(input_file_path)  # Changed from entry_path to input_file_path
                        query = "Generate a timeline of events based on the police report."
                        combined_summaries = generate_summaries(docs, query)
                        output_data.extend(write_json_output(combined_summaries, filename))

        # Convert the output data to JSON string
        json_output = json.dumps(output_data)
        # Print the JSON output
        print(json_output, end='')
    
    except Exception as e:
        logger.error(f"Error processing JSON: {str(e)}")
        print(json.dumps({"success": False, "message": "Failed to process JSON"}))
        sys.exit(1)