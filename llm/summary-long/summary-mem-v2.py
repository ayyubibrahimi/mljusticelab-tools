import os
import logging
from dotenv import find_dotenv, load_dotenv
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


# nltk.download('stopwords')

load_dotenv(find_dotenv())

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
As an AI assistant, your task is to generate a concise and chronological summary of the events described in the provided police report excerpt. Use your understanding of the context and the following guidelines to create an accurate timeline:

- Identify and extract key events, such as incidents, arrests, witness statements, and evidence collection. 
- Determine the sequence of events based on the information provided, paying attention to temporal indicators like dates, times, and phrases such as "before", "after", and "during".
- Organize the events in true chronological order, based on when they actually occurred, rather than from the perspective of the writer or any individual involved.
- Use clear and concise language to describe each event in the timeline.
- Do not infer any details that are not explicitly stated. If the text is too poorly OCR'd to derive an event, ignore this piece of the report.
- Give preference to using the exact phrases and sentences from the original text. Only paraphrase or modify the language when absolutely necessary for coherence or to resolve inconsistencies.

Given the context from the previous page ending, the current page, and the next page beginning, generate a summary of the events in chronological order using bullet points.

Example:
- [Bullet point 1]
- [Bullet point 2]
- [Bullet point 3]
- [Bullet point 4]
- [Bullet point 5]

Previous Page Ending: {previous_page_ending}
Current Page: {current_page}
Next Page Beginning: {next_page_beginning}

Chronological Summary:
"""


def generate_summaries(docs, query, window_size=500):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
    prompt_response = ChatPromptTemplate.from_template(summary_template)
    response_chain = prompt_response | llm | StrOutputParser()
    output = []

    for i in range(len(docs)):
        current_page = docs[i].page_content.replace("\n", " ")
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
                    "question": query,
                    "previous_page_ending": previous_page_ending,
                    "current_page": current_page,
                    "next_page_beginning": next_page_beginning,
                }
            )
            response["page_content"] = processed_content
            # print(response)
        output.append(response)

    # Write the output to a file named "output" in the "../data/" directory
    # with open("../data/output/output.json", "w") as file:
    #     json.dump(output, file, indent=2)

    # print("Generated page summaries:", output)

    return output


combine_template = """
As an AI assistant, your task is to combine the provided summaries of a police report into a single, comprehensive, and chronological summary. Please follow these guidelines:

1. Carefully review the summaries to identify and include all relevant information, such as:
   - Key events and actions taken by individuals involved
   - Dates and times of significant occurrences
   - Locations where events took place
   - Important details about the crime, investigation, and evidence
   - Relevant background information about the individuals involved

2. Organize the information in a clear and logical timeline, ensuring that the sequence of events is accurately represented.

3. Maintain a coherent narrative flow throughout the combined summary, linking related events and details to provide a comprehensive overview of the case.

4. When combining information from multiple summaries, give preference to using the exact phrases and sentences from the original text. Only paraphrase or modify the language when absolutely necessary for coherence or to resolve inconsistencies.

5. Ensure that all critical information from the individual summaries is included in the final combined summary, without omitting any significant details.

6. Aim to create a detailed and informative summary that captures the full scope of the case from the raw document using bulletpoints. 

7. Identify and resolve any inconsistencies or contradictions between the summaries by cross-checking information and giving preference to the most coherent and logically consistent details.

The updated summary should follow a bulletpoint format, for example:

- Point 1
- Point 2
- Point 3

Summary 1: {summary1}
Summary 2: {summary2}

Combined Summary:
"""


verification_template = """
Please carefully review the combined summary of the police report and compare it against the original summaries and the provided memory log to identify any areas for improvement, such as missing key details, events, or important information.

Memory Log:
{memory_log}

Update the combined summary to ensure it:
1. Includes all the critical events and details from the original summaries and aligns with the information in the memory log
2. Check for and resolve any inconsistencies, contradictions, or logical errors in the combined summary, ensuring that the information is coherent and logically consistent throughout.

Even if the combined summary already covers the main points, look for opportunities to further align the language with the original text while maintaining clarity and coherence.

Given the context from the combined summary, the first original summary, the second original summary, and the memory log, generate an updated summary using bullet points.

The updated summary should follow a bulletpoint format, for example:

- Point 1
- Point 2
- Point 3

Combined Summary: {combined_summary}
First Original Summary: {summary1}
Second Original Summary: {summary2}

Updated Summary:
"""

def combine_summaries(summaries, memory_log):
    # combiner_llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
    combiner_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    combiner_prompt = ChatPromptTemplate.from_template(combine_template)
    combiner_chain = combiner_prompt | combiner_llm | StrOutputParser()

    # verification_llm = ChatAnthropic(model_name="claude-3-haiku-20240307")

    verification_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
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
As an AI assistant, your task is to update the existing memory log based on the new summary provided. The current memory log is based on the previous iterations, and you are now reviewing the next subset of pages. This memory log is being used to help you understand the structure and key aspects of the entire document, even if you are viewing parts of the document that are not currently in your context window. These police investigative files were collected by an attorney. Please follow these guidelines:

1. Review the current memory log and identify the key facts, events, and details that are still relevant for understanding the overall narrative of the investigative files.
2. Analyze the new summary and identify any additional critical information, such as:
   - New developments or findings in the investigation
   - Important dates, times, and locations related to the events in the investigative files
   - Statements or interviews from involved parties, witnesses, or other relevant individuals
   - Physical evidence, documentation, or expert opinions collected during the investigation
   - Relevant laws, regulations, or police procedures that may be applicable to the case
3. Ensure that the updated memory log maintains consistency and coherence with the previous information while incorporating the new findings.
4. If any information in the current memory log is no longer relevant or contradicts the new summary, remove or update it accordingly.
5. When prioritizing information, consider what details would be essential for understanding the context and key aspects of the investigative files, even if they were initially mentioned in earlier pages. Keep in mind that this memory log serves as a reference to help you comprehend the entire document structure.
6. Limit the updated memory log to a maximum of 10-12 bullet points, focusing on the most crucial information from the police investigative files.
7. Use clear, concise, and objective language to describe each point, ensuring that the information is easily understandable and serves as a quick reference for you to grasp the overall structure and content of the document.

Please update the memory log based on the provided current memory log and the new summary:

Current memory log: {memory_log}
New summary: {summary}

Updated Memory Log:
"""

def update_memory_log(memory_log, new_summary):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
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