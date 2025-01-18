import asyncio
import json
import pandas as pd
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from aiohttp import ClientTimeout
from aiohttp_retry import RetryClient, ExponentialRetry
import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage
from llm import gpt_4o_mini

structured_llm = gpt_4o_mini.with_structured_output

# Enhanced Models for structured output
class ArticleURL(BaseModel):
    url: str
    title: str
    confidence: float
    reasoning: str

class ArticleContent(BaseModel):
    url: str
    title: str
    abstract: str
    confidence: float
    reasoning: str

class IndexStrategy(BaseModel):
    total_articles: int = Field(description="Total number of articles expected to find")
    url_pattern: str = Field(description="Pattern description for article URLs")
    chunk_size: int = Field(description="Recommended chunk size for processing")
    extraction_strategy: str = Field(description="Strategy for extracting URLs from content")

class ChunkAnalysis(BaseModel):
    urls: List[ArticleURL]
    next_chunk_start: Optional[int] = Field(description="Token index to start next chunk")
    is_complete: bool = Field(description="Whether this chunk completes the article list")
    urls_found_so_far: int = Field(description="Running count of unique URLs found")

class ProgressState(BaseModel):
    total_found: int
    total_expected: int
    current_position: int
    is_complete: bool
    next_action: str
    reasoning: str

# Enhanced Prompts with Structured Output Format

ABSTRACT_EXTRACTION_PROMPT = """You are an expert at extracting article abstracts from markdown content.
Given the markdown content for an article page, identify and extract the abstract.

Focus on:
1. Finding the main abstract/description
2. Excluding navigation elements, headers, etc.
3. Capturing the complete abstract text
4. Assessing confidence in extraction

Content to analyze:
{content}

Return your response in this exact format:
{{
    "url": "string: the URL of the article",
    "title": "string: the extracted title",
    "abstract": "string: the extracted abstract text",
    "confidence": float between 0 and 1,
    "reasoning": "string: detailed explanation of extraction process"
}}"""

STRATEGY_ANALYSIS_PROMPT = """You are an expert at analyzing article index pages to develop extraction strategies.

Your task is to analyze the initial content chunk to determine:
1. Total number of articles to find (look for counts, pagination info)
2. URL pattern for articles
3. Optimal chunk size for processing
4. Strategy for systematic extraction

Initial content to analyze:
{content}

Return your response in this exact format:
{{
    "total_articles": integer number of expected articles,
    "url_pattern": "string: detailed description of URL pattern",
    "chunk_size": integer number of tokens per chunk,
    "extraction_strategy": "string: detailed explanation of extraction approach"
}}"""

URL_EXTRACTION_PROMPT = """You are an expert at extracting article URLs from content chunks.
Current progress: Found {urls_so_far}/{total_expected} articles

URL pattern identified: {url_pattern}
Extraction strategy: {strategy}

Content chunk to analyze:
{content}

Return your response in this exact format:
{{
    "urls": [
        {{
            "url": "string: complete article URL",
            "title": "string: article title",
            "confidence": float between 0 and 1,
            "reasoning": "string: explanation for this URL"
        }},
        // ... additional URLs in same format
    ],
    "next_chunk_start": integer token index or null,
    "is_complete": boolean indicating if chunk completes article list,
    "urls_found_so_far": integer count of unique URLs found
}}"""

PROGRESS_CHECKER_PROMPT = """You are a progress tracking expert monitoring URL extraction.
Total articles expected: {total_expected}
Current unique URLs found: {current_count}
Position in content: {position}

Analyze progress and decide:
1. Whether extraction is complete
2. Next action to take
3. Any strategy adjustments needed

Return your response in this exact format:
{{
    "total_found": integer number of URLs found,
    "total_expected": integer total expected articles,
    "current_position": integer current position in content,
    "is_complete": boolean indicating if extraction is complete,
    "next_action": "string: one of [continue, refine, complete]",
    "reasoning": "string: detailed explanation of decision"
}}"""


class ArticleScraper:
    def __init__(self):
        self.timeout = ClientTimeout(total=300)
        self.retry_options = ExponentialRetry(attempts=3, start_timeout=1, max_timeout=30, factor=2.0)
        
        # Load existing data if available
        try:
            self.url_df = pd.read_csv('article_urls.csv')
            self.results_df = pd.read_csv('article_abstracts.csv')
            print(f"Loaded {len(self.url_df)} existing URLs and {len(self.results_df)} abstracts")
        except FileNotFoundError:
            self.url_df = pd.DataFrame()
            self.results_df = pd.DataFrame()
            print("Starting fresh scrape")
        
        self.found_urls = set()
        self.index_strategy = None
        self.current_position = 0
        self.processed_urls = set(self.results_df['url'].unique() if not self.results_df.empty else [])

    async def analyze_initial_chunk(self, content: str) -> IndexStrategy:
        """Analyze first chunk to determine strategy"""
        system_instructions = STRATEGY_ANALYSIS_PROMPT.format(content=content)
        strategy_llm = structured_llm(IndexStrategy)
        return strategy_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Analyze this content and develop an extraction strategy.")
        ])

    async def extract_urls_from_chunk(self, content: str, urls_so_far: int) -> ChunkAnalysis:
        """Extract URLs from a content chunk"""
        system_instructions = URL_EXTRACTION_PROMPT.format(
            urls_so_far=urls_so_far,
            total_expected=self.index_strategy.total_articles,
            url_pattern=self.index_strategy.url_pattern,
            strategy=self.index_strategy.extraction_strategy,
            content=content
        )
        chunk_llm = structured_llm(ChunkAnalysis)
        return chunk_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Extract article URLs from this chunk.")
        ])

    def check_progress(self) -> ProgressState:
        """Deterministically check extraction progress"""
        unique_urls = len(set(self.url_df['url'].unique()))
        
        progress_state = ProgressState(
            total_found=unique_urls,
            total_expected=self.index_strategy.total_articles,
            current_position=self.current_position,
            is_complete=unique_urls >= self.index_strategy.total_articles,
            next_action="complete" if unique_urls >= self.index_strategy.total_articles else "continue",
            reasoning=f"Found {unique_urls}/{self.index_strategy.total_articles} unique URLs"
        )
        
        return progress_state

    async def process_articles(self, json_path: str):
        """Main processing function with iterative extraction"""
        try:
            # Load and fetch initial content
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            index_url = data['successful_url']
            print(f"Processing index URL: {index_url}")
            
            full_content = await self.fetch_content(index_url)
            enc = tiktoken.encoding_for_model("gpt-4")
            tokens = enc.encode(full_content)
            
            # Analyze initial chunk to determine strategy
            initial_chunk = enc.decode(tokens[:1000])
            self.index_strategy = await self.analyze_initial_chunk(initial_chunk)
            print(f"Strategy determined: {self.index_strategy}")

            # Iteratively process chunks until complete
            while True:
                chunk_start = self.current_position
                chunk_end = min(chunk_start + self.index_strategy.chunk_size, len(tokens))
                current_chunk = enc.decode(tokens[chunk_start:chunk_end])
                
                # Extract URLs from current chunk
                chunk_results = await self.extract_urls_from_chunk(
                    current_chunk, 
                    len(self.found_urls)
                )
                
                # Process chunk results
                chunk_df = pd.DataFrame([{
                    'url': url.url,
                    'title': url.title,
                    'confidence': url.confidence,
                    'reasoning': url.reasoning
                } for url in chunk_results.urls])
                
                # Append new URLs to DataFrame, dropping duplicates
                self.url_df = pd.concat([self.url_df, chunk_df]).drop_duplicates(subset='url')
                
                # Save URLs immediately
                self.url_df.to_csv('article_urls.csv', index=False)
                
                # Process new abstracts only for URLs we haven't processed yet
                new_urls = [url for url in chunk_df['url'] 
                        if url not in self.processed_urls]
                
                for url in new_urls:
                    try:
                        content = await self.fetch_content(url)
                        article_content = await self.extract_abstract(content)
                        
                        # Add to results DataFrame
                        new_result = pd.DataFrame([{
                            'url': url,
                            'title': article_content.title,
                            'abstract': article_content.abstract,
                            'confidence': article_content.confidence,
                            'reasoning': article_content.reasoning
                        }])
                        
                        self.results_df = pd.concat([self.results_df, new_result])
                        self.processed_urls.add(url)
                        
                        # Save abstracts immediately
                        self.results_df.to_csv('article_abstracts.csv', index=False)
                        
                        # Update state files
                        self.save_results()
                        
                        print(f"Processed article: {article_content.title}")
                        
                    except Exception as e:
                        print(f"Error processing article {url}: {str(e)}")
                
                self.current_position = chunk_results.next_chunk_start or chunk_end
                
                # Check progress deterministically
                progress = self.check_progress()
                print(f"Progress: {progress.total_found}/{progress.total_expected} unique articles found")
                
                if progress.is_complete or self.current_position >= len(tokens):
                    break

            # Save final results
            self.save_results()
            
        except Exception as e:
            print(f"Error in main processing: {str(e)}")

    async def fetch_content(self, url: str) -> str:
        """Fetch content from URL with retry logic"""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        async with RetryClient(retry_options=self.retry_options, timeout=self.timeout) as session:
            fixed_url = url.replace('https//', 'https://')
            jina_url = f"https://r.jina.ai/{fixed_url}"
            async with session.get(jina_url, headers=headers) as response:
                return await response.text()

    async def extract_abstract(self, content: str) -> ArticleContent:
        """Extract abstract from article content (reused from original)"""
        system_instructions = ABSTRACT_EXTRACTION_PROMPT.format(content=content)
        abstract_llm = structured_llm(ArticleContent)
        return abstract_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Extract the abstract from this content.")
        ])

    def save_results(self):
        """Save results to files (reused from original)"""
        self.url_df.to_csv('article_urls.csv', index=False)
        self.results_df.to_csv('article_abstracts.csv', index=False)
        with open('article_data.json', 'w', encoding='utf-8') as f:
            json.dump({
                'total_articles': len(self.results_df),
                'articles': self.results_df.to_dict('records')
            }, f, indent=4, ensure_ascii=False)

async def main():
    scraper = ArticleScraper()
    await scraper.process_articles('exploration_results.json')

if __name__ == "__main__":
    asyncio.run(main())