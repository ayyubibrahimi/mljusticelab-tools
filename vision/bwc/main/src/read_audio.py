import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import logging
import re
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_llm():
    return ChatOpenAI(model="gpt-4o-mini")

llm = setup_llm()

prompt = """
<task_instructions>
Analyze the following audio transcription. 
Where in this transcription is there interaction between two individuals where force was likely used? 

The result of this force may have resulted in an arrest. Identify the timestamp where this interaction began. 
</task_instructions>

<materials_to_analyze>
{input}
</materials_to_analyze>

<output_instructions>
Return the timestamp in the form of (start_time=, end_time=)
</output_instructions>
"""

def format_transcription(json_data):
    formatted_string = ""
    for entry in json_data:
        start = f"{entry['start_timestamp']:.2f}"
        end = f"{entry['end_timestamp']:.2f}"
        formatted_string += f"[{start} - {end}] {entry['text']} "
    return formatted_string.strip()

def adjust_timestamps(results):
    match = re.search(r'start_time=(\d+(?:\.\d+)?), end_time=(\d+(?:\.\d+)?)', results)
    
    if match:
        start_time = float(match.group(1))
        
        # Subtract 60 seconds from start_time
        new_start_time = max(0, start_time - 25)  # Ensure it doesn't go below 0
        
        # Set end_time to 60 seconds more than the original start_time
        new_end_time = start_time + 10
        
        new_results = f"(start_time={new_start_time:.2f}, end_time={new_end_time:.2f})"
        
        return new_results
    else:
        return results

def process_audio(input_path):
    verifier_response = ChatPromptTemplate.from_template(prompt)
    final_chain = verifier_response | llm | StrOutputParser()
    logging.info(f"Processing audio transcription: {input_path}")

    with open(input_path, 'r') as file:
        json_data = json.load(file)
    
    formatted_input = format_transcription(json_data)
    
    results = final_chain.invoke(
        {
            "input": formatted_input,
        }
    )
    print(f"ORIGINAL RESULTS {results}")
    results = adjust_timestamps(results)
    print(f"ADJUSTED RESULTS {results}")
    return results
