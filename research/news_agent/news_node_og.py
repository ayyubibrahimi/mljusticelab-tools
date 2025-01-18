import asyncio
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
import aiohttp
from aiohttp import ClientTimeout
from aiohttp_retry import RetryClient, ExponentialRetry
import json
from llm import claude_3_5_sonnet
import datetime

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Models for structured output
class ExplorationLink(BaseModel):
    url: str = Field(description="The URL to explore")
    rationale: str = Field(description="Explanation of why this link would provide valuable market intelligence")

class ExplorationLinks(BaseModel):
    selected_links: List[ExplorationLink] = Field(description="List of selected links to explore")

# Initialize LLM
structured_llm = claude_3_5_sonnet.with_structured_output(ExplorationLinks)
LINK_SELECTOR_PROMPT = """You are a strategic intelligence analyst tasked with identifying the most significant current events and developments across all domains. Your goal is to select Yahoo sections that will help discover today's most important stories and developments, whether they're financial, political, social, technological, or from other spheres.

Focus on selecting links that will lead to:
1. Breaking news and major headlines from any domain
2. Significant market and economic developments
3. Important political or policy changes
4. Major social or cultural events
5. Technological breakthroughs or industry shifts

Guidelines for link selection:
- Choose sections likely to contain high-impact current events
- Look for both general news and specialized coverage
- Consider stories that have cross-domain impacts (e.g., political decisions affecting markets)
- Focus on sections that aggregate important developments
- Prioritize feeds that surface major breaking news or announcements

For each selected link provide:
- url: The complete URL path
- rationale: A detailed explanation of what types of significant events and developments we expect to find in this section and why they're important for understanding today's key stories

Think like an intelligence analyst - we're looking for the most impactful stories regardless of their domain. These links will help us discover the events shaping today's world across multiple spheres of influence.

Remember: This is the discovery phase. We're looking for sources rich in significant current events that we can analyze in detail later to understand their broader implications.

Markdown content to analyze:
{markdown_content}"""

# Modified fetch functions with increased timeouts and retry logic
async def fetch_yahoo_news():
    """Fetch yahoo news navigation content through markdown conversion service"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Set longer timeout
    timeout = ClientTimeout(total=300)  # 5 minutes total timeout
    
    # Configure retry strategy
    retry_options = ExponentialRetry(
        attempts=3,
        start_timeout=1,
        max_timeout=30,
        factor=2.0
    )
    
    async with RetryClient(retry_options=retry_options, timeout=timeout) as session:
        async with session.get('https://r.jina.ai/https://finance.yahoo.com/', headers=headers) as response:
            logger.info(f"Yahoo fetch status: {response.status}")
            content = await response.text()
            logger.info(f"Fetched content length: {len(content)}")
            return content

async def fetch_link_content(links: ExplorationLinks):
    """Fetch content from selected links with retry logic"""
    logger.info("Fetching content from selected links")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Set longer timeout
    timeout = ClientTimeout(total=300)  # 5 minutes total timeout
    
    # Configure retry strategy
    retry_options = ExponentialRetry(
        attempts=3,
        start_timeout=1,
        max_timeout=30,
        factor=2.0
    )
    
    explored_content = []
    async with RetryClient(retry_options=retry_options, timeout=timeout) as session:
        for link in links.selected_links:
            try:
                # Fix the URL formatting
                fixed_url = link.url.replace('https//', 'https://')  # Fix any double slashes
                jina_url = f"https://r.jina.ai/{fixed_url}"
                logger.info(f"Fetching: {jina_url}")
                
                async with session.get(jina_url, headers=headers) as response:
                    content = await response.text()
                    explored_content.append({
                        "url": link.url,
                        "rationale": link.rationale,
                        "content": content
                    })
                    logger.info(f"Successfully fetched content from {link.url}")
                    
            except Exception as e:
                logger.error(f"Error fetching {link.url}: {str(e)}")
    
    return explored_content

async def select_valuable_links(markdown_content: str) -> ExplorationLinks:
    """Use LLM to select valuable links for exploration"""
    logger.info("Selecting valuable links for exploration")
    
    try:
        # Format system instructions
        system_instructions = LINK_SELECTOR_PROMPT.format(
            markdown_content=markdown_content
        )
        
        # Select links using LLM
        result = structured_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Select 5 valuable links to explore from this markdown content.")
        ])
        
        logger.info("Selected links:")
        for link in result.selected_links:
            logger.info(f"\nURL: {link.url}")
            logger.info(f"Rationale: {link.rationale}")
            
        return result
        
    except Exception as e:
        logger.error(f"Error during link selection: {str(e)}")
        raise

async def save_explored_content(explored_content):
    """Save the explored content to both text and JSON files"""
    logger.info("Saving explored content to files")
    
    try:
        # Save to text file
        with open('explored_content.txt', 'w', encoding='utf-8') as f:
            for idx, content in enumerate(explored_content, 1):
                f.write(f"\n{'='*50} Link {idx} {'='*50}\n")
                f.write(f"URL: {content['url']}\n")
                f.write(f"Rationale: {content['rationale']}\n")
                f.write(f"Content:\n{content['content']}\n")
        
        # Save to JSON file
        json_output = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_links": len(explored_content),
            "explored_content": [
                {
                    "url": content["url"],
                    "rationale": content["rationale"],
                    "content": content["content"]
                }
                for content in explored_content
            ]
        }
        
        with open('explored_content.json', 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)
        
        logger.info("Successfully saved content to explored_content.txt and explored_content.json")
        
    except Exception as e:
        logger.error(f"Error saving content to files: {str(e)}")
        raise

async def main():
    try:
        # 1. Fetch yahoo news navigation
        logger.info("Starting news exploration process")
        markdown_content = await fetch_yahoo_news()
        
        # 2. Select valuable links
        selected_links = await select_valuable_links(markdown_content)
        
        # 3. Fetch content from selected links
        explored_content = await fetch_link_content(selected_links)
        await save_explored_content(explored_content)
        
        # 4. Display results
        logger.info("\nExploration Results:")
        for idx, content in enumerate(explored_content, 1):
            logger.info(f"\n{idx}. {content['url']}")
            logger.info(f"Rationale: {content['rationale']}")
            logger.info(f"Content length: {len(content['content'])} characters")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())