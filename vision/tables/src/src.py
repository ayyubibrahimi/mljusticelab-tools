import os
import base64
import csv
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import instructor
from typing import Iterable
from pydantic import BaseModel
from io import StringIO
from typing import Annotated, Any
from pydantic import Field, BeforeValidator, PlainSerializer, InstanceOf, WithJsonSchema
import pandas as pd

load_dotenv(find_dotenv())

input_path = "../data/input"
output_path = "../data/output"

client = OpenAI()

client = instructor.patch(client)

# Define MarkdownDataFrame type
def md_to_df(data: Any) -> Any:
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .applymap(lambda x: x.strip() if isinstance(x, str) else x)
        )
    return data

MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    PlainSerializer(lambda df: df.to_markdown()),
    WithJsonSchema(
        {
            "type": "string",
            "description": "The markdown representation of the table, each one should be tidy, do not try to join tables that should be separate",
        }
    ),
]

# Define Table class
class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame

# Define extract_table function
def extract_table(image_path: str) -> Iterable[Table]:
    with open(image_path, "rb") as image_file:
        return client.chat.completions.create(
            model="gpt-4o-mini",  # Using the model specified in your initial code
            response_model=Iterable[Table],
            max_tokens=1800,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract table from image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"}},
                    ],
                }
            ],
        )

# Function to save DataFrame to CSV
def save_to_csv(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)

# Main function to process images
def process_images():
    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(input_path, filename)
            print(f"Processing image: {filename}")
            
            try:
                tables = extract_table(image_path)
                for i, table in enumerate(tables, 1):
                    print(f"\nTable {i}:")
                    print(f"Caption: {table.caption}")
                    print("Data:")
                    print(table.dataframe)
                    
                    # Save to CSV
                    base_name = os.path.splitext(filename)[0]
                    csv_filename = f"{base_name}_table_{i}.csv"
                    csv_path = os.path.join(output_path, csv_filename)
                    save_to_csv(table.dataframe, csv_path)
                    print(f"Saved to CSV: {csv_path}")
                    
                    print("\n" + "="*50 + "\n")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    process_images()