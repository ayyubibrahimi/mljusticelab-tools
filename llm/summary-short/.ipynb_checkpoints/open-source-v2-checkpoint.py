import os
import logging
import json
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langchain_community.document_loaders import JSONLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from langchain_community.chat_models.huggingface import ChatHuggingFace


from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" 


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

Current Page: {current_page}

Chronological Event Summary:
"""


def generate_timeline(docs, query, window_size=500, similarity_threshold=0.15):
    HUGGINGFACEHUB_API_TOKEN = "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"
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    repo_id = "mistralai/Mixtral-8x7B-v0.1"
   
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
    )

    chat_model = ChatHuggingFace(llm=llm)
    prompt_response = ChatPromptTemplate.from_template(generate_template)
    response_chain = prompt_response | chat_model | StrOutputParser()

    vectorizer = TfidfVectorizer()
    output = []

    for doc in docs:
        current_page = doc.page_content.replace("\n", " ")
        page_number = doc.metadata.get("seq_num") 

        response = {"page_content": "", "page_number": page_number, "similarity_score": 0.0}

        if current_page:
            processed_content = response_chain.invoke(
                {
                    "question": query,
                    "current_page": current_page,
                }
            )

            corpus = [current_page, processed_content]
            tf_idf_matrix = vectorizer.fit_transform(corpus)
            similarity_score = cosine_similarity(tf_idf_matrix[0:1], tf_idf_matrix[1:2])[0][0]

            response["page_content"] = processed_content
            response["similarity_score"] = similarity_score

            print(response)

            if similarity_score >= similarity_threshold:
                output.append(response)

    return output


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
