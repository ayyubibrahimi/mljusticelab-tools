import os
import logging
from dotenv import find_dotenv, load_dotenv
import json
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
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
            augmented_text += f"({ent.text}: {label})"
            prev_end = ent.end_char
        elif (ent.start_char, ent.end_char) in entity_map:
            label = entity_map[(ent.start_char, ent.end_char)]
            augmented_text += text[prev_end : ent.start_char]
            augmented_text += f"({ent.text}: {label})"
            prev_end = ent.end_char

    augmented_text += text[prev_end:]
    print(augmented_text)
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


summary_template = """
As an AI assistant, your task is to generate a concise and chronological summary of the events described in the provided police report excerpt. Use your understanding of the context and the following guidelines to create an accurate timeline:

- Identify and extract key events, such as incidents, arrests, witness statements, and evidence collection. 
- Determine the sequence of events based on the information provided, paying attention to temporal indicators like dates, times, and phrases such as "before", "after", and "during".
- Focus on the most critical actions and developments that contribute to the overall narrative.
- Use clear and concise language to describe each event in the timeline.
- Begin the summary by setting the scene, introducing the people, property, and other relevant information before describing the actions.
- Organize the events in true chronological order, based on when they actually occurred, rather than from the perspective of the writer or any individual involved.
- After narrating the main events, include additional facts such as evidence collected, pictures taken, witness statements, recovered property, and any other necessary details.
- Do not infer any details that are not explicitly stated. If the text is too poorly OCR'd to derive an event, ignore this piece of the report. 

Given the context from the previous page ending, the current page, and the next page beginning, generate a summary of the events in chronological order.

Previous Page Ending: {previous_page_ending}
Current Page: {current_page}
Next Page Beginning: {next_page_beginning}

Chronological Event Summary:
"""


def generate_summaries(docs, query, window_size=500):
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
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
        output.append(response)

    # Write the output to a file named "output" in the "../data/" directory
    # with open("../data/output/output.json", "w") as file:
    #     json.dump(output, file, indent=2)

    print("Generated page summaries:", output)

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

4. Use concise and precise language to convey the information effectively, avoiding repetition or redundancy.

5. Ensure that all critical information from the individual summaries is included in the final combined summary, without omitting any significant details.

6. If there are any discrepancies or contradictions between the summaries, use your best judgment to resolve them based on the overall context and the reliability of the information sources.

7. Aim to create a detailed and informative summary that captures the full scope of the case, including the crime, investigation, arrests, and any relevant background information.

Summary 1: {summary1}
Summary 2: {summary2}

Combined Comprehensive Summary:
"""


verification_template = """
Please carefully review the combined summary of the police report and compare it against the original summaries to identify any areas for improvement, such as missing key details, events, or important information.

Update the combined summary to ensure it:
1. Includes all the critical events and details from the original summaries
2. Maintains a clear and logical structure
3. Uses concise and coherent language
4. Is free of any inconsistencies or redundancies
5. Is free from information that is not included in the original summaries.

Even if the combined summary already covers the main points, look for opportunities to enhance its clarity, coherence, and completeness. Make minor improvements wherever possible to ensure the final summary is of the highest quality.

Given the context from the combined summary, the first original summary, and the second original summary, generate an updated summary.

Combined Summary: {combined_summary}
First Original Summary: {summary1}
Second Original Summary: {summary2}

Updated Summary:
"""

def combine_summaries(summaries):
    combiner_llm = ChatAnthropic(model_name="claude-3-haiku-20240307")
    # combiner_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    combiner_prompt = ChatPromptTemplate.from_template(combine_template)
    combiner_chain = combiner_prompt | combiner_llm | StrOutputParser()

    verification_llm = ChatAnthropic(model_name="claude-3-haiku-20240307")

    # refiner_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
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

        verified_summary = verification_chain.invoke(
            {
                "combined_summary": combined_summary,
                "summary1": verified_summary,
                "summary2": summaries[i]["page_content"],
            }
        )

        combined_page_numbers.extend(
            summaries[i].get("page_numbers", [summaries[i].get("page_number")])
        )

    print("Combined summary content:", verified_summary)

    return {"page_content": verified_summary, "page_numbers": combined_page_numbers}


def map_sentences_to_pages(combined_summary, summaries):
    sentence_embeddings = sentence_model.encode(
        [str(sent).strip() for sent in nlp(combined_summary["page_content"]).sents]
    )
    page_embeddings = [
        sentence_model.encode(summary["page_content"]) for summary in summaries
    ]

    sentence_to_page = {}
    for idx, sentence in enumerate(nlp(combined_summary["page_content"]).sents):
        page_similarities = []
        for page_idx, page_summary in enumerate(summaries):
            similarity = cosine_similarity(
                [sentence_embeddings[idx]], [page_embeddings[page_idx]]
            )[0][0]
            page_similarities.append((page_summary.get("page_number"), similarity))
        page_similarities.sort(key=lambda x: x[1], reverse=True)
        page_number, score = page_similarities[0]
        page_number_candidate_2 = (
            page_similarities[1][0] if len(page_similarities) > 1 else None
        )
        page_number_candidate_2_score = (
            page_similarities[1][1] if len(page_similarities) > 1 else None
        )
        page_number_candidate_3 = (
            page_similarities[2][0] if len(page_similarities) > 2 else None
        )
        page_number_candidate_3_score = (
            page_similarities[2][1] if len(page_similarities) > 2 else None
        )
        sentence_to_page[str(sentence).strip()] = {
            "page_number": page_number,
            "page_number_score": score,
            "page_number_candidate_2": page_number_candidate_2,
            "page_number_candidate_2_score": page_number_candidate_2_score,
            "page_number_candidate_3": page_number_candidate_3,
            "page_number_candidate_3_score": page_number_candidate_3_score,
        }

    return sentence_to_page

def process_summaries(summaries):
    combined_summary = combine_summaries(summaries)
    sentence_to_page = map_sentences_to_pages(combined_summary, summaries)
    print("Sentence to page mapping:", sentence_to_page)
    return combined_summary, sentence_to_page



def write_json_output(combined_summary, sentence_to_page, output_file_path):
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
            page_summaries = generate_summaries(docs, query)
            combined_summary, sentence_to_page = process_summaries(page_summaries)

            output_json_path = os.path.join(
                output_directory, f"{os.path.splitext(filename)[0]}_summary.json"
            )
            write_json_output(combined_summary, sentence_to_page, output_json_path)
