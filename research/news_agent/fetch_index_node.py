import asyncio
import logging
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
import aiohttp
from aiohttp import ClientTimeout
from aiohttp_retry import RetryClient, ExponentialRetry
import json
from llm import gpt_4o_mini, gpt_4o
import datetime
from typing import List, Optional, Dict
from urllib.parse import urlparse, urljoin

from dotenv import load_dotenv
load_dotenv()

import tiktoken 
def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))

### if page is over x amount of tokens, truncate it?

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models for structured output
class ExplorationLink(BaseModel):
    url: str = Field(description="The URL to explore")
    rationale: str = Field(description="Explanation of why this link contains the news index")

class ExplorationLinks(BaseModel):
    selected_links: List[ExplorationLink] = Field(description="List of selected links to explore")

class ExtractedLink(BaseModel):
    url: str = Field(description="The URL of the extracted link")
    context: str = Field(description="Surrounding text/context of the link")
    depth_value: float = Field(description="Calculated promise/depth value between 0-1")
    parent_url: str = Field(description="URL of the page where this link was found")

class ExtractedLinks(BaseModel):
    links: List[ExtractedLink] = Field(description="Collection of extracted links with metadata")

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the page is the correct index page")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Detailed explanation of the validation decision")
    recommendations: Optional[List[str]] = Field(description="Suggestions for alternative links if validation fails")

class OrchestratorDecision(BaseModel):
    action: str = Field(description="Next action to take: 'continue', 'retry', or 'terminate'")
    feedback: str = Field(description="Reasoning behind the decision")
    alternative_strategy: Optional[str] = Field(description="Suggested strategy for finding the correct page")

class NextLinkSelection(BaseModel):
    selected_url: str = Field(description="URL of the next link to explore")
    rationale: str = Field(description="Detailed explanation of why this link was chosen")
    confidence: float = Field(description="Confidence score for this selection (0-1)")


# Initialize LLMs with structured output
structured_llm = gpt_4o_mini.with_structured_output
structured_llm_large = gpt_4o_mini.with_structured_output

# Prompts
LINK_SELECTOR_PROMPT = """You are a strategic intelligence analyst tasked with identifying the most relevant link. 
Your goal is to select the section url that will provide us with the entire index of the website.

Example successful analysis:
Input markdown:
[Latest News](/news)
[Articles](/articles/all)
[Categories](/categories)
[About Us](/about)

Analysis:
url: /articles/all
rationale: This link appears to lead to a comprehensive all section that would contain all published content. The 'all' in the URL path suggests it's a complete listing rather than a filtered or categorized view.

Example unsuccessful analysis:
Input markdown:
[Today's Headlines](/today)
[Search](/search)
[Contact](/contact)
[Latest](/latest)

Analysis:
url: /latest
rationale: While this might contain recent content, it's likely only showing the most recent articles rather than a complete index. This was a suboptimal choice as it wouldn't provide historical content.

Previous attempts (if any):
{previous_attempts}

Markdown content to analyze:
{markdown_content}

Expected Output Format:
url: (complete URL path)
rationale: (detailed explanation of selection)"""

LINK_EXTRACTOR_PROMPT = """You are a deep link analyzer tasked with finding ALL potential paths to index/all pages within the content.

Your goal is to identify any links that might lead to:
- Archived sections
- A complete catalogue 
- Complete article listings
- Historical content indexes
- Site maps
- Content directories

Analyze both direct links and contextual clues that suggest deeper navigation paths.

Example successful extraction:
Input content:
Welcome to our site! Browse [recent posts](/news) or check our [featured content](/featured).
Footer: [Site Map](/sitemap) | [alls by Year](/alls/yearly) | [Categories](/browse/categories)
Sidebar: Looking for older articles? Visit our [complete database](/db/articles) or use the [advanced search](/search/advanced)

Analysis:
links:
- url: /alls/yearly
  context: "alls by Year link in footer navigation"
  depth_value: 0.95
  parent_url: current_page_url
  reasoning: Direct link to yearly alls suggests comprehensive historical content
- url: /db/all_content
  context: "complete database link in sidebar"
  depth_value: 0.90
  parent_url: current_page_url
  reasoning: Database reference implies complete article collection
- url: /sitemap
  context: "Site Map link in footer"
  depth_value: 0.85
  parent_url: current_page_url
  reasoning: Sitemaps often provide access to complete content structure

Example unsuccessful extraction:
Input content:
Check out [today's headlines](/today) and [trending stories](/trending).
[Contact us](/contact) | [About](/about)

Analysis:
links:
- url: /trending
  context: "trending stories in main content"
  depth_value: 0.3
  parent_url: current_page_url
  reasoning: Only shows current popular content, unlikely to lead to complete index

Current page URL: {url}
Previously visited: {visited_urls}
Page content to analyze:
{content}

For each link found, provide:
1. Complete URL
2. Surrounding context
3. Depth value (0-1) based on likelihood of leading to index
4. Parent URL (current page)
5. Reasoning for depth value assignment"""


# Updated VALIDATOR_PROMPT to handle recursive exploration context
VALIDATOR_PROMPT = """
You are a validation specialist analyzing webpages to identify article index pages. Your goal is to determine if a page serves as an article index that provides access to the site's content archive.

An article index page must meet at least ONE of these criteria:
1. Contains a chronological listing of articles with clear organization
2. Provides a categorized view of all articles
3. Shows a searchable/filterable interface for accessing the full article archive
4. Contains pagination or infinite scroll showing ordered article entries

Positive indicators (increase confidence):
- Clear indication of all articles or complete content access
- Clear content organization (by date, category, author, etc.)
- Article counts or metadata

Negative indicators (decrease confidence):
- Mixed content types without clear article focus
- Mixed content types with specific article focus
- No clear organizational structure
- Limited temporal range
- Missing navigation elements

Example Valid Index Page:
[Input]
All articles available: 500
- Article Title 1 | Category: Tech | Published on Date: 2024-01-15
- Article Title 2 | Category: Science | Published on Date: 2024-01-14
- Article Title 3 | Category: Karate | Published on Date: 2024-01-1

[List of articles]

Analysis:
is_valid: true
confidence: .95
reasoning: Page shows paginated article listings with clear metadata, navigation controls, and filtering options. Structure indicates access to full article archive.
next_action: TERMINATE - Valid index page found with high confidence

Example Invalid Page:
[Input]
Featured Stories
- Breaking News: Latest Update
- Editor's Picks
- Trending Now
Recent Posts (Last 7 days)

Analysis:
is_valid: false
confidence: 0.75
reasoning: Page focuses on recent and featured content only. No indication of archive access or complete article organization.
next_action: CONTINUE - Look for links to archive/all articles

Current Context:
URL: {url}
Parent URL: {parent_url}
Depth: {depth}
Content:
{content}

Expected Output Format:
is_valid: true/false
confidence: 0.0-1.0
reasoning: detailed explanation
recommendations: list of suggestions if invalid, or None if valid
"""

# Updated ORCHESTRATOR_PROMPT for recursive exploration
ORCHESTRATOR_PROMPT = """

You are the orchestrator managing a recursive web crawling process to find the content index / page where all of the published articles live.

Decision Criteria:
1. TERMINATE (confidence >= 0.95):
   - Page contains or directly links to ALL content
   - Clear organization system (chronological/categorical)
   - Shows total content counts or clear completeness indicators
   - No better alternatives in unexplored links
   
2. CONTINUE (confidence 0.8-0.89):
   - Page is a valid partial index or navigation hub
   - Unexplored links suggest better/more complete pages
   - Within reasonable depth limit
   - Clear path to improvement visible
   
3. RETRY (confidence < 0.8):
   - Current path unlikely to lead to complete index
   - Better alternative strategies available
   - Current depth approaching limit

Example TERMINATE Decision:
Input state:
- Current depth: 2
- Latest validation: 
  * URL: /all/complete, 
  * Valid: true
  * Confidence: 0.98
- Unexplored links: None above 0.8 confidence

Analysis:
action: terminate
feedback: Found definitive index page with total counts, multiple organization schemes, and no better alternatives.
alternative_strategy: None needed - objective achieved

Example CONTINUE Decision:
Input state:
- Current depth: 2 
- Latest validation:
  * URL: /all/categories
  * Valid: true
  * Confidence: 0.85
  * Shows partial content organization
- Unexplored links: 2 promising paths
  * /all/complete (0.95 confidence)
  * /all/yearly (0.90 confidence)

Analysis:
action: continue
feedback: While current page is valid, unexplored links suggest more complete index pages available.
alternative_strategy: Prioritize /all/complete as it suggests comprehensiveness

Example RETRY Decision:
Input state:
- Current depth: 3
- Latest validation:
  * URL: /news/recent
  * Valid: false
  * Confidence: 0.3
  * Only shows recent articles
- Better unexplored alternatives available

Analysis:
action: retry
feedback: Current path leading to recent content only. Need to pivot to all section.
alternative_strategy: Look for links containing 'all', 'all', or 'complete'

Current state:
- Exploration depth: {current_depth}
- Maximum depth: {max_depth}
- Attempts made: {num_attempts}
- Latest validation: {validation_result}
- Visited URLs: {visited_urls}
- Unexplored high-value links: {unexplored_links}

Decision Process:
1. Check validation confidence against decision criteria
2. Evaluate completeness indicators
3. Assess quality of unexplored alternatives
4. Consider depth and attempt limits
5. Look for clear improvement opportunities

Expected Output Format:
action: continue/terminate/retry
feedback: detailed analysis of decision factors
alternative_strategy: specific next steps if not complete"""

class IndexCrawler:
    def __init__(self):
        self.previous_attempts = []
        self.max_attempts = 5
        self.visited_urls = set()
        self.current_depth = 0
        self.max_depth = 3
        self.exploration_links = []

    def normalize_url(self, base_url, url):
        """
        Normalize a URL by handling relative paths and ensuring consistent format
        
        Args:
            base_url (str): The base URL to resolve relative paths against
            url (str): The URL to normalize
            
        Returns:
            str: The normalized URL
        """
        try:
            # Handle empty or None URLs
            if not url:
                return None
                
            # Remove whitespace
            url = url.strip()
            
            # Handle relative URLs
            if url.startswith('/'):
                # Parse the base URL to get the scheme and netloc
                parsed_base = urlparse(base_url)
                return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
            elif not url.startswith(('http://', 'https://')):
                # For URLs without scheme, join with base URL
                return urljoin(base_url, url)
                
            # Return as-is if it's already an absolute URL
            return url
            
        except Exception as e:
            self.logger.error(f"Error normalizing URL {url}: {str(e)}")
            return None
        
    async def fetch_content(self, url: str) -> str:
        """Fetch content from URL with retry logic"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        timeout = ClientTimeout(total=300)
        retry_options = ExponentialRetry(
            attempts=3,
            start_timeout=1,
            max_timeout=30,
            factor=2.0
        )
        
        async with RetryClient(retry_options=retry_options, timeout=timeout) as session:
            fixed_url = url.replace('https//', 'https://')
            jina_url = f"https://r.jina.ai/{fixed_url}"
            async with session.get(jina_url, headers=headers) as response:
                return await response.text()

    async def select_links(self, markdown_content: str) -> ExplorationLinks:
        """Use LLM to select valuable links for exploration"""
        previous_attempts_str = "\n".join([
            f"- {attempt['url']}: {attempt.get('validation_result', {}).get('reasoning', 'No validation')}"
            for attempt in self.previous_attempts
        ])
        
        system_instructions = LINK_SELECTOR_PROMPT.format(
            previous_attempts=previous_attempts_str,
            markdown_content=markdown_content
        )
        
        selector_llm = structured_llm(ExplorationLinks)
        result = selector_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Select one valuable link to explore from this markdown content.")
        ])
        
        return result

    async def extract_links(self, url: str, content: str) -> ExtractedLinks:
        """Extract and analyze all potential index-related links from content"""
        system_instructions = LINK_EXTRACTOR_PROMPT.format(
            url=url,
            visited_urls=list(self.visited_urls),
            content=content
        )
        
        extractor_llm = structured_llm(ExtractedLinks)
        result = extractor_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Extract and analyze all potential index-related links from this content.")
        ])
        
        # Normalize URLs and add to tracking
        for link in result.links:
            link.url = self.normalize_url(url, link.url)
            link.parent_url = url
            
        return result   
    
    async def validate_page(self, url: str, content: str, depth: int, parent_url: str) -> ValidationResult:
        """Enhanced validation with depth, parent context, and content truncation"""
        
        # Count tokens in content
        content_tokens = count_tokens(content)
        truncated = False
        
        # If content exceeds threshold, truncate and mark as truncated
        if content_tokens > 15000:
            # Get encoding for truncation
            enc = tiktoken.encoding_for_model("gpt-4")
            tokens = enc.encode(content)
            # Truncate to 5000 tokens
            truncated_tokens = tokens[:1000]
            content = enc.decode(truncated_tokens)
            print(content)
            truncated = True

        # Prepare system instructions with truncation notice if needed
        truncation_notice = "\n## NOTE: The content was truncated from {original_tokens} to 5000 tokens due to length. This may indicate a comprehensive index page as the input continues for {original_tokens} more tokens. ##".format(
            original_tokens=content_tokens
        ) if truncated else ""
        
        system_instructions = VALIDATOR_PROMPT.format(
            url=url,
            content=content + truncation_notice,
            depth=depth,
            parent_url=parent_url
        )

        # Initialize validator with structured output
        validator_llm = structured_llm_large(ValidationResult)
        
        # Run validation
        result = validator_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Validate if this page is index where all of the articles live.")
        ])

        # If content was truncated and confidence is borderline, bump it up slightly
        if truncated and 0.8 <= result.confidence <= 0.85:
            result.confidence == .95
            result.reasoning += "\nNOTE: Confidence adjusted upward due to page length requiring truncation, which suggests comprehensive content."

        return result

    async def get_orchestrator_decision(
        self, 
        validation_result: ValidationResult,
        current_depth: int,
        unexplored_links: List[ExtractedLink]
    ) -> OrchestratorDecision:
        """Enhanced orchestration with recursive exploration context"""
        high_value_links = [link for link in unexplored_links if link.depth_value > 0.8]
        
        system_instructions = ORCHESTRATOR_PROMPT.format(
            current_depth=current_depth,
            max_depth=self.max_depth,
            num_attempts=len(self.previous_attempts),
            validation_result=validation_result.dict(),
            visited_urls=list(self.visited_urls),
            unexplored_links=len(high_value_links)
        )
        
        orchestrator_llm = structured_llm(OrchestratorDecision)
        result = orchestrator_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Determine next action based on recursive exploration state.")
        ])
        
        return result
    
    async def select_next_link(self, all_links: List[ExtractedLink], validation_history: List[dict]) -> NextLinkSelection:
        """Use LLM to intelligently select the next link to explore based on all available context"""
        
        NEXT_LINK_PROMPT = """You are an intelligent web crawler tasked with selecting the next most promising link to explore.

        Context:
        - We are looking for the main index/all page that contains or leads to all content
        - Current exploration depth: {depth}
        - Previously explored URLs and their results: {history}

        Available links to choose from:
        {links}

        Select the most promising unexplored link based on:
        1. Link text and context
        2. URL structure
        3. Depth value from previous analysis
        4. Parent page relationship
        5. Patterns from previous exploration attempts

        Previous validation results and what we learned from them should heavily influence the next choice.

        Example good selection:
        When seeing: 
        - /news (explored, led to recent articles only)
        - /all/complete (unexplored, looks promising)
        - /about (unexplored, unlikely to be useful)
        Selection: /all/complete
        Rationale: URL structure suggests comprehensive all, unlike /news which only showed recent content
        Confidence: 0.9

        Example poor selection:
        When seeing:
        - /latest (explored, showed recent posts)
        - /news/today (unexplored but similar to /latest)
        - /all (unexplored)
        Selection: /news/today
        Rationale: Might show different recent content than /latest
        Confidence: 0.3

        Provide:
        1. Selected URL
        2. Detailed rationale
        3. Confidence score (0-1)"""

        # Format the link information
        links_info = []
        for link in all_links:
            explored = link.url in self.visited_urls
            status = "explored" if explored else "unexplored"
            links_info.append(f"URL: {link.url}\nStatus: {status}\nContext: {link.context}\nDepth Value: {link.depth_value}\nParent: {link.parent_url}")
        
        # Format validation history
        history_info = []
        for attempt in validation_history:
            validation = attempt['validation_result']
            history_info.append(
                f"URL: {attempt['url']}\n"
                f"Valid: {validation['is_valid']}\n"
                f"Confidence: {validation['confidence']}\n"
                f"Reasoning: {validation['reasoning']}"
            )
        
        # Prepare the prompt
        system_instructions = NEXT_LINK_PROMPT.format(
            depth=self.current_depth,
            history="\n---\n".join(history_info),
            links="\n===\n".join(links_info)
        )
        
        selector_llm = structured_llm(NextLinkSelection)
        result = selector_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Select the next most promising link to explore based on all available context.")
        ])
        
        return result

    async def save_results(self):
        """Save exploration results to files only if a successful index page was found"""
        # Check if any attempt was successful by looking at the last attempt's validation
        if not self.previous_attempts:
            return
            
        last_attempt = self.previous_attempts[-1]
        validation_result = last_attempt.get('validation_result', {})
        
        # Only save if the last attempt was valid with high confidence
        # (This would be the case that caused termination)
        if validation_result.get('is_valid', False) and validation_result.get('confidence', 0) >= 0.8:
            output = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_attempts": len(self.previous_attempts),
                "successful_url": last_attempt['url'],
                "attempts": self.previous_attempts
            }
            
            # Save JSON output (excluding content to keep file size manageable)
            json_output = {
                "timestamp": output["timestamp"],
                "total_attempts": output["total_attempts"],
                "successful_url": output["successful_url"],
                "attempts": [{
                    "url": attempt["url"],
                    "parent_url": attempt["parent_url"],
                    "depth": attempt["depth"],
                    "depth_value": attempt["depth_value"],
                    "validation_result": attempt["validation_result"]
                } for attempt in self.previous_attempts]
            }
            
            with open('exploration_results.json', 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=4, ensure_ascii=False)
                
            # Save detailed text output including content
            with open('exploration_results.txt', 'w', encoding='utf-8') as f:
                f.write(f"=== SUCCESSFUL INDEX PAGE FOUND ===\n")
                f.write(f"URL: {last_attempt['url']}\n")
                f.write(f"Validation Confidence: {validation_result['confidence']}\n")
                f.write(f"Total Attempts: {len(self.previous_attempts)}\n\n")
                
                # Write details of the successful attempt
                f.write(f"{'='*50} Successful Attempt {'='*50}\n")
                f.write(f"URL: {last_attempt['url']}\n")
                f.write(f"Parent URL: {last_attempt['parent_url']}\n")
                f.write(f"Depth: {last_attempt['depth']}\n")
                f.write(f"Depth Value: {last_attempt['depth_value']}\n")
                
                f.write(f"\nValidation Results:\n")
                f.write(f"- Valid: {validation_result['is_valid']}\n")
                f.write(f"- Confidence: {validation_result['confidence']}\n")
                f.write(f"- Reasoning: {validation_result['reasoning']}\n")
                if validation_result.get('recommendations'):
                    f.write("- Recommendations:\n")
                    for rec in validation_result['recommendations']:
                        f.write(f"  * {rec}\n")
                
                if 'content' in last_attempt:
                    f.write(f"\nContent Length: {len(last_attempt['content'])} characters\n")
                    f.write("\nContent Preview (first 500 chars):\n")
                    f.write(f"{last_attempt['content']}...\n")
                
                f.write(f"\n{'='*100}\n")

    async def run(self):
        """Enhanced main execution loop with intelligent link selection"""
        try:
            logger.info("Starting intelligent web crawling")
            
            # Initial content fetch
            start_url = 'https://ddosecrets.com/'
            markdown_content = await self.fetch_content(start_url)
            
            # Initial link extraction
            extracted_links = await self.extract_links(start_url, markdown_content)
            print(extracted_links)
            
            # Initialize available links
            self.exploration_links = extracted_links.links
            
            while self.exploration_links and len(self.previous_attempts) < self.max_attempts:
                logger.info(f"Current exploration state:")
                logger.info(f"- Available links: {len(self.exploration_links)}")
                logger.info(f"- Current depth: {self.current_depth}")
                logger.info(f"- Visited URLs: {len(self.visited_urls)}")
                
                # Use LLM to select next link
                next_link_decision = await self.select_next_link(
                    all_links=self.exploration_links,
                    validation_history=self.previous_attempts
                )
                
                logger.info(f"Selected next link: {next_link_decision.selected_url}")
                logger.info(f"Selection rationale: {next_link_decision.rationale}")
                
                # Find the selected link in our list
                selected_link = next((link for link in self.exploration_links 
                                    if link.url == next_link_decision.selected_url), None)
                
                if not selected_link or selected_link.url in self.visited_urls:
                    logger.warning("Selected link invalid or already visited")
                    continue
                    
                self.visited_urls.add(selected_link.url)
                logger.info(f"Exploring: {selected_link.url} (depth: {self.current_depth})")
                
                # Fetch and validate content
                content = await self.fetch_content(selected_link.url)
                validation_result = await self.validate_page(
                    url=selected_link.url,
                    content=content,
                    depth=self.current_depth,
                    parent_url=selected_link.parent_url
                )
                
                # Record attempt
                self.previous_attempts.append({
                    "url": selected_link.url,
                    "parent_url": selected_link.parent_url,
                    "depth": self.current_depth,
                    "depth_value": selected_link.depth_value,
                    "content": content,
                    "validation_result": validation_result.model_dump()  # Using model_dump() instead of dict()
                })
                
                # Extract more links if needed
                if not validation_result.is_valid or validation_result.confidence < 0.8:
                    new_links = await self.extract_links(selected_link.url, content)
                    # Add new links to exploration pool
                    for new_link in new_links.links:
                        if new_link.url not in self.visited_urls:
                            self.exploration_links.append(new_link)
                
                # Get orchestrator decision
                decision = await self.get_orchestrator_decision(
                    validation_result=validation_result,
                    current_depth=self.current_depth,
                    unexplored_links=[l for l in self.exploration_links if l.url not in self.visited_urls]
                )
                
                logger.info(f"Attempt {len(self.previous_attempts)} - URL: {selected_link.url}")
                logger.info(f"Validation: {validation_result.is_valid} ({validation_result.confidence})")
                logger.info(f"Decision: {decision.action}")
                
                if decision.action == 'terminate':
                    logger.info("Successfully found the correct index page!")
                    break
                elif decision.action == 'continue':
                    self.current_depth += 1
                    continue
                else:  # retry with new strategy
                    logger.info(f"Retrying with new strategy: {decision.alternative_strategy}")
                    # Could potentially modify exploration strategy based on decision.alternative_strategy
            
            # Save results
            await self.save_results()
            
            if len(self.previous_attempts) >= self.max_attempts:
                logger.warning("Reached maximum attempts without finding correct page")
            
        except Exception as e:
            logger.error(f"Error in crawler execution: {str(e)}")
            raise

if __name__ == "__main__":
    crawler = IndexCrawler()
    asyncio.run(crawler.run())
