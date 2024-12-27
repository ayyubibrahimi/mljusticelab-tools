import pandas as pd
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 
import numpy as np
from dotenv import find_dotenv, load_dotenv
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import numpy as np
from rank_bm25 import BM25Okapi

from dateutil import parser
import re
from typing import List,  Dict

load_dotenv(find_dotenv())

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('all-MiniLM-L6-v2')

# model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
model.to(device)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

page_citation_template = """
Your task is to identify which page number contains the source content for the given bulletpoint.

Bulletpoint to find source for:
{current_bulletpoint}

Here are the most likely source pages from the document:

{reference_materials}

Based on the raw text above, which two single page number are most likely to be the source of the bulletpoint?
Return your response in this format: PAGE: [page number_1], [page_number_2], CONFIDENCE: [HIGH/MEDIUM/LOW]
If you cannot determine the source, return: PAGE: UNKNOWN CONFIDENCE: LOW
"""

# page_citation_template = """
# Your task is to identify which page numbers contain the source content for the given bulletpoint.

# Determine the source pages using these criteria:

# SOURCE MATCHING CRITERIA:
# 1. EXACT match: The page contains the exact or nearly exact wording from the bulletpoint
# 2. PARAPHRASE match: The page contains all the key information but with different wording
# 3. PARTIAL match: The page contains some but not all of the key information
# 4. CONTEXTUAL match: The page provides necessary context or background for the bulletpoint

# CONFIDENCE LEVEL CRITERIA:
# - HIGH: At least one page has an EXACT match, or multiple pages have strong PARAPHRASE matches
# - MEDIUM: One or more pages have clear PARAPHRASE matches, or strong PARTIAL matches
# - LOW: Only PARTIAL or CONTEXTUAL matches found, or matches are ambiguous

# Return ALL pages that contain relevant source material, ordered by relevance.
# You must return at least one page number unless no matches are found.
# If multiple pages contain identical information, include all of them.

# Return your response in exactly this format:
# PAGE: [page_number_1], [page_number_2], CONFIDENCE: [HIGH/MEDIUM/LOW]

# If you cannot find any matching source content, return exactly:
# PAGE: UNKNOWN CONFIDENCE: LOW

# EXAMPLE SCENARIOS AND RESPONSES:

# HIGH CONFIDENCE Examples:
# - Single EXACT match: PAGE: [45] CONFIDENCE: HIGH
# - Two EXACT matches: PAGE: [23], [45] CONFIDENCE: HIGH
# - One EXACT + one PARAPHRASE: PAGE: [12], [15] CONFIDENCE: HIGH
# - Multiple strong PARAPHRASE matches: PAGE: [8], [9], [10] CONFIDENCE: HIGH

# MEDIUM CONFIDENCE Examples:
# - Two partial matches that together form complete info: PAGE: [23], [24] CONFIDENCE: MEDIUM
# - One PARAPHRASE match with some missing details: PAGE: [16] CONFIDENCE: MEDIUM
# - Multiple PARTIAL matches with complementary info: PAGE: [7], [12], [15] CONFIDENCE: MEDIUM
# - Strong PARTIAL + weak CONTEXTUAL match: PAGE: [31], [32] CONFIDENCE: MEDIUM

# LOW CONFIDENCE Examples:
# - Only CONTEXTUAL matches found: PAGE: [8], [9] CONFIDENCE: LOW
# - Single weak PARTIAL match: PAGE: [14] CONFIDENCE: LOW
# - Ambiguous matches across many pages: PAGE: [3], [4], [5], [6] CONFIDENCE: LOW
# - Only tangentially related content: PAGE: [21] CONFIDENCE: LOW

# NO MATCH Example:
# - No relevant content found: PAGE: UNKNOWN CONFIDENCE: LOW

# Bulletpoint to find source for:
# {current_bulletpoint}

# Here are the most likely source pages from the document:
# {reference_materials}
# """

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings using PyTorch
    """
    # Ensure tensors are normalized
    embedding1_normalized = F.normalize(embedding1, p=2, dim=0)
    embedding2_normalized = F.normalize(embedding2, p=2, dim=0)
    
    # Calculate cosine similarity
    similarity = torch.dot(embedding1_normalized, embedding2_normalized)
    return similarity.item()

def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text in various formats.
    Handles formats like:
    - December 25, 2024
    - 12/25/2024
    - 25-12-2024
    - 2024-12-25
    - Dec 25th 2024
    """
    dates = []
    
    # Common date patterns
    patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # 12/25/2024, 25-12-2024
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',     # 2024-12-25
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4}',  # December 25, 2024
        r'\d{1,2}(?:st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}'      # 25th December 2024
    ]
    
    # Find all date matches
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                # Parse the date string to ensure it's valid
                date_str = match.group()
                parsed_date = parser.parse(date_str)
                dates.append(parsed_date.strftime('%Y-%m-%d'))  # Standardize format
            except (ValueError, parser.ParserError):
                continue
                
    return list(set(dates))  # Remove duplicates

# def get_candidate_pages(text: str, reference_dict: Dict, n_candidates: int = 10, alpha: float = 0.8, beta: float = 0.7) -> List[Dict]:
#     """
#     Find top candidate pages using BERT embeddings, BM25, and date matching
    
#     Args:
#         text: Query text to match
#         reference_dict: Dictionary containing reference materials
#         n_candidates: Number of top candidates to return
#         alpha: Weight for combining embedding and BM25 scores (0.8 default)
#         beta: Weight for input embeddings vs summary embeddings (0.7 default)
#             Higher beta means input embeddings are weighted more heavily
    
#     Returns:
#         List of top candidate pages with combined scores
#     """
#     # Extract dates from query text
#     query_dates = extract_dates(text)
    
#     # Generate embedding for target text
#     target_embedding = torch.tensor(model.encode(text)).to(device)
#     page_scores = []
    
#     # Pre-compute embeddings for all summaries and inputs
#     summaries = [item['summary'] for item in reference_dict['summaries']]
#     inputs = [item['original_input'] for item in reference_dict['summaries']]
    
#     # Prepare corpus for BM25
#     tokenized_corpus = [doc.lower().split() for doc in inputs]
#     bm25 = BM25Okapi(tokenized_corpus)
    
#     # Get BM25 scores
#     tokenized_query = text.lower().split()
#     bm25_scores = bm25.get_scores(tokenized_query)
    
#     # Normalize BM25 scores to [0,1] range
#     if len(bm25_scores) > 0:
#         bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
    
#     # Calculate embedding similarities
#     summary_embeddings = torch.tensor(model.encode(summaries)).to(device)
#     input_embeddings = torch.tensor(model.encode(inputs)).to(device)
    
#     summary_sims = F.cosine_similarity(target_embedding.unsqueeze(0), summary_embeddings)
#     input_sims = F.cosine_similarity(target_embedding.unsqueeze(0), input_embeddings)
    
#     # Move to CPU for numpy operations
#     summary_sims = summary_sims.cpu()
#     input_sims = input_sims.cpu()
    
#     for idx, item in enumerate(reference_dict['summaries']):
#         # If query contains dates, check if any match in the page text
#         if query_dates:
#             page_dates = extract_dates(item['original_input'])
#             date_match = bool(set(query_dates) & set(page_dates))  # Check for any common dates
            
#             # Skip this page if it doesn't contain any matching dates
#             if not date_match:
#                 continue
        
#         # Calculate weighted embedding similarity score
#         # Beta controls the weight between input and summary similarities
#         embedding_score = (beta * input_sims[idx].item() + (1 - beta) * summary_sims[idx].item())
        
#         # Combine embedding and BM25 scores
#         combined_score = alpha * embedding_score + (1 - alpha) * bm25_scores[idx]
        
#         page_scores.append({
#             'page_number': item['page_number'],
#             'original_input': item['original_input'],
#             'combined_score': combined_score,
#             'embedding_score': embedding_score,
#             'bm25_score': bm25_scores[idx],
#             'summary_sim': summary_sims[idx].item(),
#             'input_sim': input_sims[idx].item(),
#             'date_match': True if query_dates else None
#         })
    
#     # If no pages match the date constraints but dates were present,
#     # return empty list to indicate no valid matches
#     if query_dates and not page_scores:
#         return []
    
#     # Sort by combined score and get top n candidates
#     top_candidates = sorted(page_scores, key=lambda x: x['combined_score'], reverse=True)[:n_candidates]
#     return top_candidates

def get_candidate_pages(text: str, reference_dict: Dict, n_candidates: int = 10, alpha: float = 0.8, beta: float = 0.7, temperature: float = 0.5) -> List[Dict]:
    """
    Find top candidate pages using word-level BERT embeddings, BM25, and date matching
    
    Args:
        text: Query text to match
        reference_dict: Dictionary containing reference materials
        n_candidates: Number of top candidates to return
        alpha: Weight for combining embedding and BM25 scores (0.8 default)
        beta: Weight for input embeddings vs summary embeddings (0.7 default)
        temperature: Temperature parameter for softmax (0.5 default)
            Lower values make softmax more selective
    
    Returns:
        List of top candidate pages with combined scores
    """
    # Extract dates from query text
    query_dates = extract_dates(text)
    
    # Split query into words and generate embeddings for each word
    query_words = text.split()
    query_word_embeddings = torch.tensor(model.encode(query_words)).to(device)
    
    page_scores = []
    
    # Pre-compute embeddings for all summaries and inputs
    summaries = [item['summary'] for item in reference_dict['summaries']]
    inputs = [item['original_input'] for item in reference_dict['summaries']]
    
    # Prepare corpus for BM25
    tokenized_corpus = [doc.lower().split() for doc in inputs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Get BM25 scores
    tokenized_query = text.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Normalize BM25 scores to [0,1] range
    if len(bm25_scores) > 0:
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
    
    # Calculate embedding similarities at document level
    summary_embeddings = torch.tensor(model.encode(summaries)).to(device)
    input_embeddings = torch.tensor(model.encode(inputs)).to(device)
    
    for idx, item in enumerate(reference_dict['summaries']):
        # Handle date matching
        if query_dates:
            page_dates = extract_dates(item['original_input'])
            date_match = bool(set(query_dates) & set(page_dates))
            if not date_match:
                continue
        
        # Calculate word-level similarities
        page_words = item['original_input'].split()
        page_word_embeddings = torch.tensor(model.encode(page_words)).to(device)
        
        # Calculate similarity matrix between all query words and page words
        # Shape: (num_query_words, num_page_words)
        similarity_matrix = torch.mm(query_word_embeddings, page_word_embeddings.t())
        
        # Apply softmax to similarity scores for each query word
        word_importance = F.softmax(similarity_matrix / temperature, dim=1)
        
        # Get maximum similarity for each query word
        max_similarities, _ = torch.max(similarity_matrix, dim=1)
        
        # Weight the max similarities by word importance
        weighted_similarities = max_similarities * word_importance.max(dim=1)[0]
        
        # Calculate final word-level embedding score
        word_level_score = weighted_similarities.mean().item()
        
        # Calculate document-level similarities
        doc_summary_sim = F.cosine_similarity(
            torch.tensor(model.encode(text)).to(device).unsqueeze(0),
            summary_embeddings[idx].unsqueeze(0)
        ).item()
        
        doc_input_sim = F.cosine_similarity(
            torch.tensor(model.encode(text)).to(device).unsqueeze(0),
            input_embeddings[idx].unsqueeze(0)
        ).item()
        
        # Combine document and word-level scores
        embedding_score = (
            beta * (0.7 * word_level_score + 0.3 * doc_input_sim) + 
            (1 - beta) * doc_summary_sim
        )
        
        # Combine with BM25 score
        combined_score = alpha * embedding_score + (1 - alpha) * bm25_scores[idx]
        
        page_scores.append({
            'page_number': item['page_number'],
            'original_input': item['original_input'],
            'combined_score': combined_score,
            'embedding_score': embedding_score,
            'word_level_score': word_level_score,
            'doc_input_sim': doc_input_sim,
            'doc_summary_sim': doc_summary_sim,
            'bm25_score': bm25_scores[idx],
            'date_match': True if query_dates else None
        })
    
    # Handle empty results for date matching
    if query_dates and not page_scores:
        return []
    
    # Sort by combined score and return top candidates
    top_candidates = sorted(page_scores, key=lambda x: x['combined_score'], reverse=True)[:n_candidates]
    return top_candidates

def format_reference_materials(candidates):
    """
    Format reference materials focusing only on raw input text
    """
    formatted_pages = []
    
    for item in candidates:
        page_text = f"=== PAGE {item['page_number']} ===\n{item['original_input']}\n"
        formatted_pages.append(page_text)
    
    return "\n".join(formatted_pages)


def generate_citation(bulletpoint, candidates):
    """
    Generate citation using LLM after finding top candidate pages
    """
    citation_prompt = ChatPromptTemplate.from_template(page_citation_template)
    citation_chain = citation_prompt | llm | StrOutputParser()
    
    formatted_materials = format_reference_materials(candidates)
    
    response = citation_chain.invoke({
        "current_bulletpoint": bulletpoint,
        "reference_materials": formatted_materials
    })

    print(response)
    
    return response


def clean_page_number(page_str):
    """
    Clean a page number string by removing brackets and extra whitespace
    Returns None if the page number is invalid
    """
    cleaned = str(page_str).strip().strip('[]')
    if not cleaned or cleaned.lower() == 'unknown':
        return None
    return cleaned

def parse_page_numbers(page_info):
    """
    Parse page numbers from various formats
    """
    if not page_info or page_info.lower() == 'unknown':
        return []
        
    if ',' in page_info:
        pages = page_info.split(',')
    else:
        pages = [page_info]
    
    cleaned_pages = []
    for page in pages:
        cleaned = clean_page_number(page)
        if cleaned:
            cleaned_pages.append(cleaned)
            
    return cleaned_pages or ["UNKNOWN"]

def process_citations(df, citations_data):
    """
    Process each row in the DataFrame to add citations
    Handles flexible number of page citations and confidence levels
    
    Args:
        df: DataFrame containing sentences to process
        citations_data: Dictionary containing reference materials
        
    Returns:
        DataFrame with added columns for page citations (as list) and confidence
    """
    page_citations = []
    
    for idx, row in df.iterrows():
        try:
            # Get top candidate pages
            candidates = get_candidate_pages(row['sentence'], citations_data)
            
            # Generate citation using LLM
            citation_result = generate_citation(row['sentence'], candidates)
            
            # Debug print
            print(f"\nRow {idx} raw result: {citation_result}")
            
            # Extract page info and confidence
            if 'PAGE:' in citation_result and 'CONFIDENCE:' in citation_result:
                # Split out the page numbers and confidence
                page_info = citation_result.split('PAGE:')[1].split('CONFIDENCE:')[0].strip()
                
                # Parse page numbers into a list
                pages = parse_page_numbers(page_info)
                
            else:
                pages = ["UNKNOWN"]
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            pages = ["UNKNOWN"]
            confidence = "LOW"
            
        page_citations.append(pages)
    
        # Print progress
        if idx % 10 == 0:
            print(f"Processed {idx} rows...")
    
    # Add columns to DataFrame
    df['page_citations'] = page_citations
    return df



def extract_sentences(text):
    """
    Extract individual sentences from bullet-pointed text.
    """
    sentences = [s.strip() for s in text.split('•') if s.strip()]
    sentences = [' '.join(s.split()) for s in sentences]
    return sentences

def read_tbl(json_path):
    """
    Read JSON file and convert to DataFrame with one sentence per row.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_sentences = []
    
    for entry in data:
        for file in entry['files']:
            sentences = extract_sentences(file['filename'])
            sentences.extend(extract_sentences(file['sentence']))
            
            for sentence in sentences:
                all_sentences.append({
                    'sentence': sentence,
                    'filename': file['filename'],
                })
    
    return pd.DataFrame(all_sentences)

if __name__ == "__main__":
    # File paths
    json_path = "../../summary/data/output/summary_20241226_144919.json"
    citations_path = "../../summary/data/output/complete/citations.json"
    
    # Read input files
    df = read_tbl(json_path)
    with open(citations_path, 'r') as f:
        citations_data = json.load(f)
    
    # Process citations
    df = process_citations(df, citations_data)
    
    # Save results
    output_path = "../data/output/summary_with_citations.csv"
    df.to_csv(output_path, index=False)
    
    # Display sample results
    print("\nDataFrame Shape:", df.shape)
    print("\nSample rows with citations:")
    print(df.head(5))