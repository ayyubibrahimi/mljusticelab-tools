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
from crawl4ai import *

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
class TopicAnalysis(BaseModel):
    recommended_topic: str = Field(description="The most significant topic identified from news")
    significance_score: float = Field(description="Score indicating topic importance (0-1)")
    rationale: str = Field(description="Explanation of why this topic is significant")


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



# Add this constant with your other prompts
TOPIC_ANALYZER_PROMPT = """You are a strategic research advisor analyzing current news to identify the most significant topic for in-depth research.

Today's News Content:
{news_content}

Your task is to:
1. Identify the single most significant developing story that warrants deep research
2. Consider topics that:
   - Have major current developments
   - Would benefit from historical analysis
   - Impact multiple domains (economic, social, political, etc.)
   - Have ongoing relevance
   - Offer rich opportunities for data collection

Required Output:
- recommended_topic: A clear, focused topic for research (be specific but not too narrow)
- significance_score: A score from 0-1 indicating the topic's current importance
- rationale: Explanation of why this topic deserves priority research

Remember: Choose a topic that would benefit from both historical context and current data analysis."""

class FetchNews:
    def __init__(self):
        self.structured_llm = claude_3_5_sonnet.with_structured_output(ExplorationLinks)
        self.topic_analyzer = claude_3_5_sonnet.with_structured_output(TopicAnalysis)

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.timeout = ClientTimeout(total=300)  # 5 minutes total timeout
        self.retry_options = ExponentialRetry(
            attempts=3,
            start_timeout=1,
            max_timeout=30,
            factor=2.0
        )
        
    # async def fetch_yahoo_news(self) -> str:
    #     """Fetch yahoo news navigation content through markdown conversion service"""
    #     async with RetryClient(retry_options=self.retry_options, timeout=self.timeout) as session:
    #         async with session.get('https://r.jina.ai/https://finance.yahoo.com/', 
    #                              headers=self.headers) as response:
    #             logger.info(f"Yahoo fetch status: {response.status}")
    #             content = await response.text()
    #             logger.info(f"Fetched content length: {len(content)}")
    #             return content

    async def fetch_yahoo_news(self) -> str:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url="https://finance.yahoo.com",
            )
            print(result.markdown)
            return result.markdown


    async def select_valuable_links(self, markdown_content: str) -> ExplorationLinks:
        """Use LLM to select valuable links for exploration"""
        logger.info("Selecting valuable links for exploration")
        
        try:
            system_instructions = LINK_SELECTOR_PROMPT.format(
                markdown_content=markdown_content
            )
            
            result = self.structured_llm.invoke([
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

    async def fetch_link_content(self, links: ExplorationLinks) -> List[dict]:
        """Fetch content from selected links with retry logic"""
        logger.info("Fetching content from selected links")
        
        explored_content = []
        async with RetryClient(retry_options=self.retry_options, timeout=self.timeout) as session:
            for link in links.selected_links:
                try:
                    fixed_url = link.url.replace('https//', 'https://')
                    jina_url = f"https://r.jina.ai/{fixed_url}"
                    logger.info(f"Fetching: {jina_url}")
                    
                    async with session.get(jina_url, headers=self.headers) as response:
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
    
    async def analyze_news_content(self, explored_content: List[dict]) -> str:
        """Analyze collected news to identify the most significant topic for research"""
        logger.info("Analyzing news content to identify research topic")
        
        try:
            # Format the news content for analysis
            formatted_content = "\n\n".join([
                f"Source: {item['url']}\n"
                f"Summary: {item['rationale']}\n"
                f"Content Excerpt: {item['content'][:1000]}..."  # Truncate long content
                for item in explored_content
            ])

            # Get topic analysis
            analysis = self.topic_analyzer.invoke([
                SystemMessage(content=TOPIC_ANALYZER_PROMPT.format(
                    news_content=formatted_content
                )),
                HumanMessage(content="Analyze this news content to identify the most significant topic for research.")
            ])

            logger.info(f"Identified topic: {analysis.recommended_topic}")
            logger.info(f"Significance score: {analysis.significance_score}")
            logger.info(f"Rationale: {analysis.rationale}")

            return analysis.recommended_topic

        except Exception as e:
            logger.error(f"Error during topic analysis: {str(e)}")
            raise

    async def run(self) -> List[dict]:
        """Main method to collect and process news"""
        try:
            markdown_content = await self.fetch_yahoo_news()
            selected_links = await self.select_valuable_links(markdown_content)
            explored_content = await self.fetch_link_content(selected_links)
            recommended_topic = await self.analyze_news_content(explored_content)
            
            logger.info("\nExploration Results:")
            for idx, content in enumerate(explored_content, 1):
                logger.info(f"\n{idx}. {content['url']}")
                logger.info(f"Rationale: {content['rationale']}")
                logger.info(f"Content length: {len(content['content'])} characters")
            
            return recommended_topic
            
        except Exception as e:
            logger.error(f"Error in news collection: {str(e)}")
            raise

# Update main execution
if __name__ == "__main__":
    collector = FetchNews()
    asyncio.run(collector.run())
    