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
        "DATE": "Date",
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


generate_template = """
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


def generate_timeline(docs, query, window_size=500):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    prompt_response = ChatPromptTemplate.from_template(generate_template)
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
    with open("../data/output/output.json", "w") as file:
        json.dump(output, file, indent=2)

    # print("Generated page summaries:", output)

    return output


combine_template = """
As an AI assistant, your task is to combine the provided summaries of a police report into a single, comprehensive, and chronological summary. Please follow these guidelines:

1. Carefully review all the summaries to identify and include all relevant information, such as:
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


def combine_summaries(summaries):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    prompt_response = ChatPromptTemplate.from_template(combine_template)

    response_chain = prompt_response | llm | StrOutputParser()

    combined_summary = summaries[0]["page_content"]
    combined_page_numbers = summaries[0].get(
        "page_numbers", [summaries[0].get("page_number")]
    )

    for i in range(1, len(summaries)):
        processed_content = response_chain.invoke(
            {"summary1": combined_summary, "summary2": summaries[i]["page_content"]}
        )

        combined_summary = processed_content
        combined_page_numbers.extend(
            summaries[i].get("page_numbers", [summaries[i].get("page_number")])
        )

    # print("Combined summary content:", combined_summary)

    return {"page_content": combined_summary, "page_numbers": combined_page_numbers}


def map_sentences_to_pages(combined_summary, summaries):
    sentence_embeddings = sentence_model.encode(
        [str(sent).strip() for sent in nlp(combined_summary["page_content"]).sents]
    )
    page_embeddings = [
        sentence_model.encode(summary["page_content"]) for summary in summaries
    ]

    sentence_to_page = {}
    for idx, sentence in enumerate(nlp(combined_summary["page_content"]).sents):
        max_similarity = 0
        page_number = None
        for page_idx, page_summary in enumerate(summaries):
            similarity = cosine_similarity(
                [sentence_embeddings[idx]], [page_embeddings[page_idx]]
            )[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                page_number = page_summary.get("page_number")
        sentence_to_page[str(sentence).strip()] = page_number

    return sentence_to_page


def process_summaries(summaries):
    combined_summary = combine_summaries(summaries)
    sentence_to_page = map_sentences_to_pages(combined_summary, summaries)
    # print("Sentence to page mapping:", sentence_to_page)
    return combined_summary, sentence_to_page


cross_reference_template = """
As an AI assistant, your task is to compare the ground truth summary with the summary of summaries and identify any missing or inconsistent information. Please follow these steps to augment the summary of summaries:

Carefully review the ground truth summary and identify all key events, details, and relevant information, such as:
Significant actions taken by individuals involved
Precise dates, times, and locations of events
Critical details about the crime, investigation, arrests, and evidence
Important background information about the individuals involved
Compare the identified key information from the ground truth summary with the content of the summary of summaries.
For each piece of key information from the ground truth summary, determine if it is: a) Present in the summary of summaries and consistent b) Present in the summary of summaries but inconsistent or incomplete c) Missing from the summary of summaries entirely
Based on your analysis, augment the summary of summaries:
For information that is present and consistent, no changes are needed.
For information that is present but inconsistent or incomplete, update the relevant parts of the summary of summaries to match the ground truth.
For information that is missing, add it to the summary of summaries in the most appropriate location to maintain chronological order and narrative flow.
Ensure that the augmented summary of summaries:
Includes all the key information from the ground truth summary
Maintains a coherent structure and logical flow
Uses clear and concise language
Is free of inconsistencies or contradictions
If there is any information in the summary of summaries that directly conflicts with the ground truth summary, prioritize the information from the ground truth summary.

After augmenting the summary of summaries, review it once more to ensure it is a comprehensive, accurate, and well-structured representation of the events described in the ground truth summary.

Your augmented summary must be at least 1000 tokens in length. 

Groundtruth Summary:
{groundtruth}

Summary of Summaries:
{summary_of_summaries}

Augmented Summary of Summaries:
"""


def cross_reference_summaries(groundtruth, summary, summaries):
    # llm = ChatOpenAI(model_name="gpt-4-0125-preview")
    # llm = ChatAnthropic(model_name="claude-3-haiku-20240307", anthropic_api_key="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")
    # llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", anthropic_api_key="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")

    llm = ChatAnthropic(
        model_name="claude-3-opus-20240229",
        anthropic_api_key="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",
    )

    prompt_response = ChatPromptTemplate.from_template(cross_reference_template)
    response_chain = prompt_response | llm | StrOutputParser()

    response = response_chain.invoke(
        {"groundtruth": groundtruth, "summary_of_summaries": summary}
    )

    # print("Augmented Summary:", response)

    augmented_summary = {"page_content": response}
    sentence_to_page = map_sentences_to_pages(augmented_summary, summaries)
    # print("Updated Sentence to Page Mapping:", sentence_to_page)

    return response, sentence_to_page


comparison_template = """
As an AI assistant, 
your task is to compare the groundtruth summary with the generated summary of summaries and provide a score from 0 to 10 indicating how well the summary of summaries reflects the information in the groundtruth.

roundtruth Summary:
{groundtruth}

Summary of Summaries:
{summary_of_summaries}

Score: 
1-10

"""


def compare_summaries(groundtruth, summary):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    prompt_response = ChatPromptTemplate.from_template(comparison_template)
    response_chain = prompt_response | llm | StrOutputParser()

    response = response_chain.invoke(
        {"groundtruth": groundtruth, "summary_of_summaries": summary}
    )

    print(response)

    return response


def write_json_output(combined_summary, sentence_to_page, output_file_path):
    output_data = []
    for sentence, page_number in sentence_to_page.items():
        output_data.append({"sentence": sentence, "page_number": page_number})

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
            page_summaries = generate_timeline(docs, query)
            print(page_summaries)
            combined_summary, sentence_to_page = process_summaries(page_summaries)

            augmented_summary, updated_sentence_to_page = cross_reference_summaries(
                page_summaries, combined_summary, page_summaries
            )

            comparison_score = compare_summaries(page_summaries, combined_summary)

            output_json_path = os.path.join(
                output_directory, f"{os.path.splitext(filename)[0]}_summary.json"
            )
            write_json_output(
                augmented_summary, updated_sentence_to_page, output_json_path
            )
