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

load_dotenv(find_dotenv())

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def augment_named_entities(text):
    doc = nlp(text)
    ner_results = ner_pipeline(text)
    entity_map = {}
    for entity in ner_results:
        start, end, label = entity['start'], entity['end'], entity['entity_group']
        entity_map[(start, end)] = label

    augmented_text = ""
    prev_end = 0
    for ent in doc.ents:
        if ent.label_ in ['DATE', 'PERSON']:
            augmented_text += text[prev_end:ent.start_char]
            augmented_text += f"{ent.text}: {ent.label_}"
            prev_end = ent.end_char
        elif (ent.start_char, ent.end_char) in entity_map:
            label = entity_map[(ent.start_char, ent.end_char)]
            augmented_text += text[prev_end:ent.start_char]
            augmented_text += f"{ent.text}: {label}"
            prev_end = ent.end_char

    augmented_text += text[prev_end:]
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
You are an AI assistant tasked with generating a detailed timeline of events based on the provided police report. Please extract relevant information from the report to create a chronological timeline.

Previous Page Ending:
{previous_page_ending}

Current Page:
{current_page}

Next Page Beginning:
{next_page_beginning}

Timeline of Events:
"""

def generate_timeline(docs, query, window_size=500):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    prompt_response = ChatPromptTemplate.from_template(generate_template)
    response_chain = prompt_response | llm | StrOutputParser()
    output = []

    for i in range(len(docs)):
        current_page = docs[i].page_content.replace('\n', ' ')
        previous_page_ending = docs[i-1].page_content.replace('\n', ' ')[-window_size:] if i > 0 else ""
        next_page_beginning = docs[i+1].page_content.replace('\n', ' ')[:window_size] if i < len(docs) - 1 else ""
        page_number = docs[i].metadata.get('seq_num')

        response = {"page_content": "", "page_number": page_number}
        if current_page:
            processed_content = response_chain.invoke({
                "question": query,
                "previous_page_ending": previous_page_ending,
                "current_page": current_page,
                "next_page_beginning": next_page_beginning
            })
            response["page_content"] = processed_content
        output.append(response)
    
    print("Generated page summaries:", output)  

    return output

combine_template = """
You are an AI assistant tasked with combining two summaries of a police report into a single, coherent summary. The summaries may contain overlapping information, so please consolidate and organize the information chronologically.

Summary 1:
{summary1}

Summary 2:
{summary2}

Combined Summary:
"""

def write_json_output(combined_summary, sentence_to_page, output_file_path):
    output_data = []
    for sentence, page_number in sentence_to_page.items():
        output_data.append({
            "sentence": sentence,
            "page_number": page_number
        })

    with open(output_file_path, 'w') as file:
        json.dump(output_data, file, indent=4)

def combine_summaries(summaries):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    prompt_response = ChatPromptTemplate.from_template(combine_template)
    response_chain = prompt_response | llm | StrOutputParser()

    combined_summary = summaries[0]["page_content"]
    combined_page_numbers = summaries[0].get("page_numbers", [summaries[0].get("page_number")])

    for i in range(1, len(summaries)):
        processed_content = response_chain.invoke({
            "summary1": combined_summary,
            "summary2": summaries[i]["page_content"]
        })

        combined_summary = processed_content
        combined_page_numbers.extend(summaries[i].get("page_numbers", [summaries[i].get("page_number")]))
    
    print("Combined summary content:", combined_summary)  

    return {"page_content": combined_summary, "page_numbers": combined_page_numbers}

def map_sentences_to_pages(combined_summary, summaries):
    sentence_embeddings = sentence_model.encode([str(sent).strip() for sent in nlp(combined_summary["page_content"]).sents])
    page_embeddings = [sentence_model.encode(summary["page_content"]) for summary in summaries]
    
    sentence_to_page = {}
    for idx, sentence in enumerate(nlp(combined_summary["page_content"]).sents):
        max_similarity = 0
        page_number = None
        for page_idx, page_summary in enumerate(summaries):
            similarity = cosine_similarity([sentence_embeddings[idx]], [page_embeddings[page_idx]])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                page_number = page_summary.get("page_number")
        sentence_to_page[str(sentence).strip()] = page_number

    return sentence_to_page

def process_summaries(summaries):
    combined_summary = combine_summaries(summaries)
    sentence_to_page = map_sentences_to_pages(combined_summary, summaries)
    print("Sentence to page mapping:", sentence_to_page)  
    return combined_summary, sentence_to_page


if __name__ == "__main__":
    input_directory = "../data/output-ocr"
    output_directory = "../data/output-llm"
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            json_path = os.path.join(input_directory, filename)
            docs = load_and_split(json_path)
            query = "Generate a timeline of events based on the police report."
            page_summaries = generate_timeline(docs, query)
            combined_summary, sentence_to_page = process_summaries(page_summaries)

            output_json_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_summary.json")
            write_json_output(combined_summary, sentence_to_page, output_json_path)
