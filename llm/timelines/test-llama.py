import pandas as pd
from llama_index.core import Document


from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from langchain_community.document_loaders import JSONLoader

from llama_index.core import PropertyGraphIndex

def load_and_split(json_path):
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".messages[]",
        content_key="page_content",
    )
    data = loader.load()
    return data

# Path to your JSON file
input_directory = "../../ocr/data/output/test/testing.json"
data = load_and_split(input_directory)

documents = [Document(text=d.page_content) for d in data]

# Example of the data structure
print(documents[0])

# Set OpenAI API Key

# Define LLMs
llm = OpenAI(model="gpt-3.50-turbo", temperature=0.0)
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

# Define entities and relations
entities = Literal["PERSON", "LOCATION", "ORGANIZATION", "PRODUCT", "EVENT"]
relations = Literal[
    "SUPPLIER_OF",
    "COMPETITOR",
    "PARTNERSHIP",
    "ACQUISITION",
    "WORKS_AT",
    "SUBSIDIARY",
    "BOARD_MEMBER",
    "CEO",
    "PROVIDES",
    "HAS_EVENT",
    "IN_LOCATION",
]

# Validation schema
validation_schema = {
    "Person": ["WORKS_AT", "HAS_EVENT"],
    "Organization": [
        "HAS_EVENT",
        "IN_LOCATION",
    ],
    "Event": ["HAS_EVENT", "IN_LOCATION"],
    "Location": ["HAPPENED_AT", "IN_LOCATION"],
}

# Create SchemaLLMPathExtractor
kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,
)

print(kg_extractor)