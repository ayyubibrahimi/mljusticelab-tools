
import asyncio
import aiohttp
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langsmith import traceable
from llm import claude_3_5_sonnet

import logging


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


# Schema
class HeadlineData(BaseModel):
    title: str = Field(description="The headline text")
    url: str = Field(description="URL to the full article")

class Headlines(BaseModel):
    headlines: List[HeadlineData] = Field(description="List of extracted headlines")

class Headline(BaseModel):
    title: str = Field(description="The headline text")
    url: str = Field(description="URL to the full article")
    importance_score: float = Field(description="Score indicating headline importance (0-1)")
    summary: str = Field(description="Brief summary of why this headline is significant")

class HeadlineStateInput(TypedDict, total=False):
    markdown_content: str  # Raw markdown from Yahoo Finance

class HeadlineStateOutput(TypedDict):
    top_headlines: List[Headline]  # Final top 5 headlines

class HeadlineState(TypedDict):
    markdown_content: str  # Raw markdown from Yahoo Finance
    extracted_headlines: List[Headline]  # All headlines found
    analyzed_headlines: List[Headline]  # Headlines after importance analysis
    top_headlines: List[Headline]  # Final top 5 headlines

# Prompts
headline_extractor_instructions = """You are an expert at processing financial news content. Extract all news headlines from the provided Yahoo Finance markdown content.

Guidelines for extraction:
1. Look for headlines that are actual news items (not navigation elements or ads)
2. Focus only on headlines from news articles and market updates
3. Maintain the original headline text exactly as written
4. Include the complete URL for each headline

Your output must be a list of headlines where each headline has:
- title: The exact headline text
- url: The complete URL to the article

Example output structure:
{
    "headlines": [
        {
            "title": "Example Financial Headline",
            "url": "https://finance.yahoo.com/example"
        }
    ]
}

Markdown content to analyze:
{markdown_content}"""

headline_analyzer_instructions = """You are an expert financial news analyst. Analyze each headline and determine its significance in the current market context.

For each headline, consider:
1. Market Impact (40% weight): Potential effect on stock prices, market sectors, or overall market movement
2. Economic Significance (30% weight): Broader economic implications and policy impacts
3. Time Sensitivity (20% weight): Urgency and immediate relevance
4. Global Relevance (10% weight): International market implications

Score Guidelines:
- 0.8-1.0: Critical market-moving news
- 0.6-0.7: Significant market impact
- 0.4-0.5: Moderate importance
- 0.0-0.3: Limited market impact

For each headline, provide:
- importance_score: A score from 0-1
- summary: A 1-2 sentence explanation of the score

Analyze this headline:
{headline_text}"""

# Graph nodes
async def fetch_yahoo_finance(state: HeadlineState):
    """Fetch Yahoo Finance content through Markdown conversion service"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    async with aiohttp.ClientSession() as session:
        async with session.get('https://r.jina.ai/https://finance.yahoo.com/') as response:
            logger.info(f"Response headers: {dict(response.headers)}")
            logger.info(f"Response status: {response.status}")
            
            markdown_content = await response.text()
            logger.info(f"Content length: {len(markdown_content)}")
            
            # Log the first and last 100 characters to see where truncation occurs
            # logger.info(f"Start of content: {markdown_content}")
            logger.info(f"End of content: {markdown_content[-500:]}")
            
            return {"markdown_content": markdown_content}

def extract_headlines(state: HeadlineState) -> dict:
    """Extract headlines and their URLs from markdown content using LLM analysis"""
    logger.info("Starting headline extraction")
    markdown_content = state["markdown_content"]
    
    try:
        # Create structured LLM for headline extraction
        structured_llm = claude_3_5_sonnet.with_structured_output(Headlines)
        
        # Format system instructions
        system_instructions = headline_extractor_instructions.format(
            markdown_content=markdown_content
        )
        
        logger.info("Sending content to LLM for headline extraction")
        # Extract headlines using LLM
        result = structured_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Extract all news headlines from this markdown content. Return them in the specified JSON structure with 'headlines' as the root key.")
        ])
        
        # Initialize headlines with empty scores and summaries
        headlines = [
            Headline(
                title=h.title,
                url=h.url,
                importance_score=0,
                summary=""
            ) for h in result.headlines
        ]
        
        logger.info(f"Extracted {len(headlines)} headlines")
        for idx, headline in enumerate(headlines, 1):
            logger.info(f"Headline {idx}: {headline.title}")
            logger.info(f"URL {idx}: {headline.url}")
        
        return {"extracted_headlines": headlines}
    except Exception as e:
        logger.error(f"Error during headline extraction: {str(e)}")
        logger.error(f"LLM Response: {result if 'result' in locals() else 'No response'}")
        raise

async def analyze_headlines(state: HeadlineState) -> dict:
    """Analyze headlines for importance and generate summaries using LLM"""
    logger.info("Starting headline analysis")
    headlines = state["extracted_headlines"]
    
    try:
        # Define a schema for headline analysis
        class HeadlineAnalysis(BaseModel):
            importance_score: float = Field(
                description="Importance score from 0-1",
                ge=0,
                le=1
            )
            summary: str = Field(
                description="Brief explanation of the headline's significance"
            )
        
        # Create structured LLM for analysis
        structured_llm = claude_3_5_sonnet.with_structured_output(HeadlineAnalysis)
        
        analyzed_headlines = []
        logger.info(f"Analyzing {len(headlines)} headlines")
        
        for idx, headline in enumerate(headlines, 1):
            logger.info(f"Analyzing headline {idx}/{len(headlines)}: {headline.title}")
            
            # Format system instructions for this headline
            system_instructions = headline_analyzer_instructions.format(
                headline_text=f"Title: {headline.title}\nURL: {headline.url}"
            )
            
            # Analyze headline using LLM
            result = structured_llm.invoke([
                SystemMessage(content=system_instructions),
                HumanMessage(content="Analyze this headline's importance and provide a summary.")
            ])
            
            analyzed_headline = Headline(
                title=headline.title,
                url=headline.url,
                importance_score=result.importance_score,
                summary=result.summary
            )
            
            logger.info(f"Headline score: {result.importance_score:.2f}")
            logger.info(f"Summary: {result.summary}")
            
            analyzed_headlines.append(analyzed_headline)
        
        # Sort by importance score
        analyzed_headlines.sort(key=lambda x: x.importance_score, reverse=True)
        logger.info("Completed headline analysis and sorting")
        
        return {"analyzed_headlines": analyzed_headlines}
    except Exception as e:
        logger.error(f"Error during headline analysis: {str(e)}")
        raise

def select_top_headlines(state: HeadlineState) -> dict:
    """Select the top 5 most important headlines"""
    logger.info("Selecting top headlines")
    analyzed_headlines = state["analyzed_headlines"]
    top_headlines = analyzed_headlines[:5]
    
    logger.info("Top 5 headlines selected:")
    for idx, headline in enumerate(top_headlines, 1):
        logger.info(f"\n{idx}. {headline.title}")
        logger.info(f"Importance: {headline.importance_score}")
        logger.info(f"Summary: {headline.summary}")
        logger.info(f"URL: {headline.url}")
    
    return {"top_headlines": top_headlines}

# Build the graph
builder = StateGraph(HeadlineState, input=HeadlineStateInput, output=HeadlineStateOutput)

builder.add_node("fetch_yahoo_finance", fetch_yahoo_finance)
builder.add_node("extract_headlines", extract_headlines)
builder.add_node("analyze_headlines", analyze_headlines)
builder.add_node("select_top_headlines", select_top_headlines)

# Add edges
builder.add_edge(START, "fetch_yahoo_finance")
builder.add_edge("fetch_yahoo_finance", "extract_headlines")
builder.add_edge("extract_headlines", "analyze_headlines")
builder.add_edge("analyze_headlines", "select_top_headlines")
builder.add_edge("select_top_headlines", END)

graph = builder.compile()

async def main():
    logger.info("Starting Yahoo Finance Headlines Analysis")
    try:
        # Initial state requires markdown_content
        initial_state = HeadlineStateInput(
            markdown_content=""  # Empty string as placeholder, will be populated by fetch_yahoo_finance
        )
        
        logger.info("Executing analysis pipeline")
        result = await graph.ainvoke(initial_state)
        
        logger.info("\nFinal Results:")
        for idx, headline in enumerate(result["top_headlines"], 1):
            logger.info(f"\n{idx}. {headline.title}")
            logger.info(f"Importance: {headline.importance_score}")
            logger.info(f"Summary: {headline.summary}")
            logger.info(f"URL: {headline.url}")
        
        logger.info("Analysis complete")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())