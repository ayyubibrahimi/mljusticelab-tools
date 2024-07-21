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

#### difference is that we're asking the model to preserve the original language 

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
</guidelines>

<essential_information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
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
## Previous Page Ending ##
{previous_page_ending}

## Next Page Beginning ##
{next_page_beginning}

## Current Page ##
{current_page}
</reference_materials>

<output_instruction>
Generate the current page summary below:
</output_instruction>
"""


def process_page(docs, i, query, window_size):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", api_key="")
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
    return response

def generate_summaries(docs, query, window_size=500):
    pool = multiprocessing.Pool()
    results = pool.starmap(
        process_page,
        [(docs, i, query, window_size) for i in range(len(docs))],
    )
    pool.close()
    pool.join()

    return results


combine_template = """
<task_description>
As a Legal Clerk, your task is to concatenate the provided page summaries into a single, comprehensive, and well-organized summary for the given section of the police report. Your goal is to create the best possible summary by taking the most important and relevant information from each provided summary and combining them into a detailed, chronological, and coherent summary without any duplication.
</task_description>

<guidelines>
1. Comprehensive Information Integration:
   • Review the the summaries to extract the most important information that is relevant to producing a summary.

2. Handling Contradictions:
   • If inconsistencies arise between the summaries, prioritize the most detailed and specific information.
   • If the information is incomplete, do not include it.

3. Factual Accuracy:
   • DO NOT include any details not explicitly stated in either summary.

4. Formatting for Clarity:
   • Ensure that the combined summary is formatted as bullet points with a logical flow of information.
   • If possible, organize the bullet points chronologically.
</guidelines>

<essential_information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
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
Before combining the summaries, consider:
1. What are the main topics covered across all summaries?
2. How can the information be organized chronologically?
3. Are there any contradictions or inconsistencies between summaries?
4. What information from the memory log provides crucial context?
5. How can I ensure all essential information is included without duplication?
</thinking_process>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the summaries
- Do not omit critical information even if it appears in multiple summaries
- Ensure that all information is accurately attributed to the correct parties and events
</warnings>

<reference_materials>
Summary 1: 
{summary1}

Summary 2: 
{summary2}
</reference_materials>

<output_instruction>
Generate the combined summary below, ensuring it adheres to all guidelines, includes all essential information, and is presented in a clear, bullet-point format:
</output_instruction>
"""

verification_template = """
<task_description>
As a Legal Clerk, your task is to review the updated combined summary, which integrates content two individual summaries (summary1 and summary2) of a police investigative report. This verification process aims to ensure that all relevant information from both summaries are contained within the current combined summary, such as full names, roles, badge numbers, specific events with dates, policy or rule violations, disciplinary actions, relevant evidence, legal actions, and outcomes related to the case from both sources.
</task_description>

<verification_guidelines>
1. Comprehensive Information Integration:
   • Ensure that all important details from both summaries, such as critical people, key facts, key events, and significant details, are accurately incorporated into the updated combined summary.

2. Context Preservation:
   • Verify that all of the important information from both summaries are preserved in the updated combined summary.

3. Logical Flow:
   • Evaluate the updated combined summary for logical flow and coherence, ensuring that the newly integrated information fits seamlessly into the existing narrative.
   • If possible, order the information chronologically.

4. Factual Accuracy:
   • DO NOT include any details not explicitly stated in either summary.
</verification_guidelines>

<essential_information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
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
Before verifying and updating the combined summary, consider:
1. What are the key differences between the current combined summary and summary 1 and summary 2?
2. Are there any new pieces of information in either summary 1 or summary 2 that are not in the current combined summary?
3. How can I ensure the updated summary maintains a logical and chronological flow?
3. Are there any contradictions or inconsistencies between the summaries that need to be addressed?
</thinking_process>

<warnings>
- Do not include speculative information
- Avoid including irrelevant details
- Do not draw conclusions not explicitly stated in the summaries
- Do not omit critical information from either summary
- Ensure that all information is accurately attributed to the correct parties and events
- Do not alter the meaning or context of any information when integrating it into the updated summary
</warnings>

<reference_materials>

Memory Log:
The memory log contains a running list of important facts that should be considered throughout the entire document, regardless of the specific page being analyzed. Keep these facts in mind when reviewing and updating the combined summary.
{memory_log}

## Combined Summary (merged version of summary1 and summary2) ##: 
{combined_summary}

## Summary1 (first individual summary) ##: 
{summary1}

## Summary2 (second individual summary) ##: 
{summary2}
</reference_materials>

<output_instruction>
Provide the updated combined summary below, ensuring that all relevant information from both the current combined summary and the new summary is accurately retained. If no updates are needed, return the current combined summary. Present the summary in a clear, bullet-point format, organized chronologically where possible:
</output_instruction>
"""


def combine_summaries(summaries, memory_log):
    # combiner_llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
    combiner_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", api_key="")
    combiner_prompt = ChatPromptTemplate.from_template(combine_template)
    combiner_chain = combiner_prompt | combiner_llm | StrOutputParser()

    # verification_llm = ChatAnthropic(model_name="claude-3-haiku-20240307")

    verification_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", api_key="")
    verification_prompt = ChatPromptTemplate.from_template(verification_template)
    verification_chain = verification_prompt | verification_llm | StrOutputParser()

    verified_summary = summaries[0]["page_content"]
    combined_page_numbers = summaries[0].get(
        "page_numbers", [summaries[0].get("page_number")]
    )

    for i in range(1, len(summaries)):
        combined_summary = combiner_chain.invoke(
            {"summary1": verified_summary, "summary2": summaries[i]["page_content"]}
        )

        if i % 5 == 0:
            new_summary = ' '.join(summary['page_content'] for summary in summaries[i-5:i])
            memory_log = update_memory_log(memory_log, new_summary)
            print(memory_log)

        verified_summary = verification_chain.invoke(
            {
                "combined_summary": combined_summary,
                "summary1": verified_summary,
                "summary2": summaries[i]["page_content"],
                "memory_log": memory_log,
            }
        )

        combined_page_numbers.extend(
            summaries[i].get("page_numbers", [summaries[i].get("page_number")])
        )

    # print("Combined summary content:", verified_summary)

    return {"page_content": verified_summary, "page_numbers": combined_page_numbers}, memory_log



def map_sentences_to_pages(combined_summary, docs, num_iterations=3, ensemble_models=None):
    if ensemble_models is None:
        ensemble_models = [
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ]

    sentences = re.split(r'\n-\s*', combined_summary["page_content"])
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    split_sentences = []
    for sentence in sentences:
        if '\n' in sentence:
            sub_sentences = sentence.split('\n')
            split_sentences.extend(sub_sentences)
        else:
            split_sentences.append(sentence)
    split_sentences = [sent.strip() for sent in split_sentences if sent.strip()]

    sentence_to_page = {}
    for idx, sentence in enumerate(split_sentences):
        page_similarities_ensemble = []
        for model_name in ensemble_models:
            sentence_model = SentenceTransformer(model_name)
            page_similarities_iterations = []
            for _ in range(num_iterations):
                sentence_embedding = sentence_model.encode([sentence])
                page_embeddings = [sentence_model.encode(doc.page_content) for doc in docs]
                page_similarities = []
                for page_idx, doc in enumerate(docs):
                    similarity = cosine_similarity(sentence_embedding, [page_embeddings[page_idx]])[0][0]
                    page_similarities.append((doc.metadata["seq_num"], similarity))
                page_similarities.sort(key=lambda x: x[1], reverse=True)
                page_similarities_iterations.append(page_similarities[:3])
            page_similarities_ensemble.append(page_similarities_iterations)

        page_number_scores = {}
        for model_similarities in page_similarities_ensemble:
            for page_similarities in model_similarities:
                for page_number, score in page_similarities:
                    if page_number not in page_number_scores:
                        page_number_scores[page_number] = []
                    page_number_scores[page_number].append(score)

        final_page_similarities = []
        for page_number, scores in page_number_scores.items():
            avg_score = sum(scores) / len(scores)
            final_page_similarities.append((page_number, avg_score))
        final_page_similarities.sort(key=lambda x: x[1], reverse=True)

        page_number = final_page_similarities[0][0] if final_page_similarities else None
        score = final_page_similarities[0][1] if final_page_similarities else None
        page_number_candidate_2 = final_page_similarities[1][0] if len(final_page_similarities) > 1 else None
        page_number_candidate_2_score = final_page_similarities[1][1] if len(final_page_similarities) > 1 else None
        page_number_candidate_3 = final_page_similarities[2][0] if len(final_page_similarities) > 2 else None
        page_number_candidate_3_score = final_page_similarities[2][1] if len(final_page_similarities) > 2 else None

        sentence_to_page[sentence] = {
            "page_number": page_number,
            "page_number_score": score,
            "page_number_candidate_2": page_number_candidate_2,
            "page_number_candidate_2_score": page_number_candidate_2_score,
            "page_number_candidate_3": page_number_candidate_3,
            "page_number_candidate_3_score": page_number_candidate_3_score,
        }

    return sentence_to_page


def longest_common_substring(s1, s2):
    # Function to find the longest common substring between two strings
    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_pos = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
    
    start_pos = end_pos - max_length
    return s1[start_pos:end_pos]


memory_log_template = """
<task_description>
As a Legal Clerk, your task is to review the new summary and update the memory log only when the new summary contains crucial information directly related to the main subject of the document. Maintain a concise memory log that focuses on the key aspects of the events, allegations, investigations, and outcomes described in the document.
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

def update_memory_log(memory_log, new_summary):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", api_key="")
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
    memory_log_prompt = ChatPromptTemplate.from_template(memory_log_template)
    memory_log_chain = memory_log_prompt | llm | StrOutputParser()

    updated_memory_log = memory_log_chain.invoke({"summary": new_summary, "memory_log": memory_log})

    return updated_memory_log


def process_summaries(summaries, docs, memory_log):
    combined_summary, updated_memory_log = combine_summaries(summaries, memory_log)
    sentence_to_page = map_sentences_to_pages(combined_summary, docs)

    return combined_summary, sentence_to_page, updated_memory_log


def write_json_output(sentence_to_page, output_file_path):
    output_data = []
    for sentence, page_number in sentence_to_page.items():
        page_number_dict = {
            "sentence": sentence,
            "page_number": int(page_number["page_number"]),
            "page_number_score": float(page_number["page_number_score"]),
            "page_number_candidate_2": int(page_number["page_number_candidate_2"])
            if page_number["page_number_candidate_2"] is not None
            else None,
            "page_number_candidate_2_score": float(
                page_number["page_number_candidate_2_score"]
            )
            if page_number["page_number_candidate_2_score"] is not None
            else None,
            "page_number_candidate_3": int(page_number["page_number_candidate_3"])
            if page_number["page_number_candidate_3"] is not None
            else None,
            "page_number_candidate_3_score": float(
                page_number["page_number_candidate_3_score"]
            )
            if page_number["page_number_candidate_3_score"] is not None
            else None,
        }
        output_data.append(page_number_dict)
        # print(output_data)

    with open(output_file_path, "w") as file:
        json.dump(output_data, file, indent=4)

if __name__ == "__main__":
    input_directory = "../../ocr/data/output"
    output_directory = "../data/output"

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            json_path = os.path.join(input_directory, filename)
            docs = load_and_split(json_path)
            query = "Generate a timeline of events based on the police report."

            interval_size = 20
            num_intervals = (len(docs) + interval_size - 1) // interval_size
            interval_summaries = []

            memory_log = ""

            for interval in range(num_intervals):
                start_index = interval * interval_size
                end_index = min((interval + 1) * interval_size, len(docs))
                interval_docs = docs[start_index:end_index]

                page_summaries = generate_summaries(interval_docs, query)
                combined_summary, sentence_to_page, memory_log = process_summaries(page_summaries, interval_docs, memory_log)

                interval_output_json_path = os.path.join(
                    output_directory, f"{os.path.splitext(filename)[0]}_interval_{interval+1}_summary.json"
                )
                write_json_output(sentence_to_page, interval_output_json_path)

                # Split the combined summary into sentences
                sentences = combined_summary["page_content"].split("\n- ")
                for sentence in sentences:
                    # Skip empty sentences
                    if sentence.strip():
                        sentence_info = sentence_to_page.get(sentence.strip(), {})
                        interval_summaries.append({
                            "sentence": sentence.strip(),
                            "page_number": sentence_info.get("page_number", None),
                            "page_number_score": sentence_info.get("page_number_score", None),
                            "page_number_candidate_2": sentence_info.get("page_number_candidate_2", None),
                            "page_number_candidate_2_score": sentence_info.get("page_number_candidate_2_score", None),
                            "page_number_candidate_3": sentence_info.get("page_number_candidate_3", None),
                            "page_number_candidate_3_score": sentence_info.get("page_number_candidate_3_score", None)
                        })

            combined_output_json_path = os.path.join(
                output_directory, f"{os.path.splitext(filename)[0]}_combined_summary.json"
            )
            with open(combined_output_json_path, "w") as file:
                json.dump(interval_summaries, file, indent=4)