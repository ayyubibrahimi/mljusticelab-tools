import pandas as pd
import json
import re
from transformers import AutoTokenizer

def read_tbl():
    # dfa = pd.read_csv("../data/input/training_data_san_bernardino.csv")
    # dfb = pd.read_csv("../data/input/training_data_fresno.csv")

    # df = pd.concat([dfa, dfb], ignore_index=True)
    df = pd.read_csv("../data/input/training_data.csv")
    return df

def create_prompt(ocr_text):
    prefix = """
<task_description>
As an Information Analyst, your task is to generate a comprehensive, structured summary of all significant information from the provided document. Extract and organize all key facts, events, relationships, and contextual details, regardless of document type.
</task_description>

<guidelines>
1. Extract ALL material information:
- Key events and their sequence
- Important individuals and their roles
- Locations and timestamps
- Quantitative data and metrics
- Critical decisions or outcomes

2. Maintain strict accuracy:
- Use precise terminology from the source
- Preserve exact dates and numbers
- Keep original designations for people/entities
- Quote significant statements verbatim when relevant

3. Organize information hierarchically:
- Main topics as primary headers
- Supporting details as sub-bullets
- Chronological order when relevant
- Group related information logically
</guidelines>

<essential_categories>
1. WHO
- All mentioned individuals/entities
- Their roles and relationships
- Organizations involved

2. WHAT
- Main events/incidents
- Actions taken
- Decisions made
- Outcomes achieved

3. WHEN
- Specific dates and times
- Sequence of events
- Durations
- Deadlines or timeframes

4. WHERE
- Specific locations
- Jurisdictions
- Physical settings
- Relevant geographical context

5. CONTEXT
- Background information
- Underlying circumstances
- Related events
- Environmental factors
</essential_categories>

<quality_controls>
1. Verification Requirements:
- Cross-reference all dates and numbers
- Verify names and designations
- Confirm event sequences
- Check relationships between facts

2. Output Format:
OVERVIEW:
[Document type and main topic]

KEY PARTIES:
[People/entities involved]

TIMELINE:
[Chronological sequence of events]

MAIN POINTS:
[Core information organized by topic]

OUTCOMES:
[Results/conclusions]
</quality_controls>

<warnings>
- Do not include speculative information
- Do not draw conclusions not explicitly stated in the text
- Avoid summarizing irrelevant details
- Do not add interpretive commentary
- If information is ambiguous, note the ambiguity
- Maintain neutral, objective tone throughout
</warnings>

<thinking_process>
Before summarizing, consider:
1. What is the main purpose of this document?
2. Who are the key parties involved?
3. What is the chronological sequence of events?
4. What are the critical pieces of information that must be included?
5. How can the information be organized most effectively?
</thinking_process>

Below is the document you will review:
"""
    return prefix + str(ocr_text)

def get_bert_token_count(text, tokenizer):
    """
    Get exact token count using BERT tokenizer
    """
    return len(tokenizer.encode(str(text)))

def is_valid_row(row, tokenizer):
    # Check if fields are valid and OCR text has between 10,000 and 120,000 tokens
    if not (pd.notna(row['ocr']) and 
            pd.notna(row['summary']) and 
            str(row['ocr']).strip() != '' and 
            str(row['summary']).strip() != ''):
        return False
    
    token_count = get_bert_token_count(row['ocr'], tokenizer)
    return 3000 <= token_count <= 10000

def generate_training_data(df):
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    training_examples = []
    skipped_count = 0
    token_count_too_low = 0
    token_count_too_high = 0
    
    for idx, row in df.iterrows():
        if not pd.notna(row['ocr']) or str(row['ocr']).strip() == '':
            skipped_count += 1
            continue
            
        token_count = get_bert_token_count(row['ocr'], tokenizer)
        
        if token_count <= 3000:
            token_count_too_low += 1
            continue
        elif token_count >= 10000:
            token_count_too_high += 1
            continue
            
        if not is_valid_row(row, tokenizer):
            skipped_count += 1
            continue
            
        example = {
            "messages": [
                {
                    "role": "user",
                    "content": create_prompt(row['ocr'])
                },
                {
                    "role": "assistant",
                    "content": str(row['summary'])
                }
            ]
        }
        training_examples.append(example)
    
    print(f"Processed {len(training_examples)} valid examples")
    print(f"Skipped {skipped_count} invalid rows")
    print(f"Skipped {token_count_too_low} rows due to insufficient tokens (<10,000)")
    print(f"Skipped {token_count_too_high} rows due to excessive tokens (>120,000)")
    
    return training_examples

def save_training_data(examples, output_path):
    """
    Save examples in JSONL format (one JSON object per line)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            # Write each example as a single line of JSON
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

def create_train_val_split(examples, train_ratio=0.8):
    """
    Split examples into training and validation sets
    """
    # Determine split index
    split_idx = int(len(examples) * train_ratio)
    
    # Split examples
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"\nSplit statistics:")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    return train_examples, val_examples

if __name__ == "__main__":
    # Read the data
    df = read_tbl()
    
    # Generate training examples
    examples = generate_training_data(df)
    
    # Split into train and validation sets
    train_examples, val_examples = create_train_val_split(examples)
    
    # Save to files
    save_training_data(train_examples, "../data/output/training_data.jsonl")
    save_training_data(val_examples, "../data/output/validation_data.jsonl")