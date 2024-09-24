import os
import logging
from dotenv import find_dotenv, load_dotenv
import json
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from datetime import datetime

load_dotenv(find_dotenv())

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

def load_and_split(json_path):
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".messages[]",
        content_key="page_content",
    )
    data = loader.load()
    return data


def create_graph(doc):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["EVENT", "PERSON"],
        node_properties=["DATE", "PERSON"],
    )
    graph_document = llm_transformer.convert_to_graph_documents([doc])
    return graph_document[0]

def parse_and_save_relationships(graph_document, output_file):
    with open(output_file, 'a') as file:
        for relationship in graph_document.relationships:
            file.write(f"Relationships: {relationship} \n")

            # source_node = relationship.source
            # target_node = relationship.target
            # relationship_type = relationship.type

            # file.write(f"Source: {source_node.id} ({source_node.type})\n")
            # file.write(f"Target: {target_node.id} ({target_node.type})\n")
            # file.write(f"Relationship: {relationship_type}\n")
            # file.write("\n")

if __name__ == "__main__":
    input_directory = "../../ocr/data/output/test/testing.json"
    output_file = "data/relationships.txt"

    docs = load_and_split(input_directory)

    with open(output_file, 'w') as file:
        file.write("")  # Clear the file before appending

    for doc in docs:
        graph_document = create_graph(doc)
        parse_and_save_relationships(graph_document, output_file)