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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import tiktoken 
def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))

### if page is over x amount of tokens, truncate it?


class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the page is the correct index page")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Detailed explanation of the validation decision")
    recommendations: Optional[List[str]] = Field(description="Suggestions for alternative links if validation fails")

class AgentState(BaseModel):
    """Represents the current state of an individual exploration agent"""
    agent_id: str = Field(description="Unique identifier for this agent")
    current_url: str = Field(description="Current URL being explored")
    current_depth: int = Field(description="Current exploration depth")
    visited_urls: List[str] = Field(description="URLs visited by this agent")
    validation_result: Optional[ValidationResult] = Field(description="Latest validation result")
    parent_url: Optional[str] = Field(description="Parent URL of current page")
    depth_value: Optional[float] = Field(description="Depth value of current page")

class MultiAgentDecision(BaseModel):
    """High-level orchestrator decision based on all agents' results"""
    action: str = Field(description="Global action: 'terminate', 'explore_new', 'explore_deeper'")
    target_agent_ids: List[str] = Field(description="Agents that should take action")
    target_urls: Optional[Dict[str, str]] = Field(description="New URLs for each agent to explore")
    rationale: str = Field(description="Detailed explanation of the decision")
    confidence: float = Field(description="Confidence in the decision (0-1)")
    winner_agent_id: Optional[str] = Field(description="Agent ID that found the index page")

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
  depth_value: 0.90
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
    def __init__(self, num_agents: int = 5):
        self.num_agents = num_agents
        self.agents = {}  # Dict[str, AgentState]
        self.previous_attempts = []
        self.max_attempts = 15  # Increased for multiple agents
        self.visited_urls = set()
        self.current_depth = 0
        self.max_depth = 3
        self.exploration_links = []

    def initialize_agents(self):
        """Create initial agent states"""
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                current_url="",
                current_depth=0,
                visited_urls=[],
                validation_result=None,
                parent_url=None,
                depth_value=None
            )

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

        print(f"Links: {result}")
        
        # Normalize URLs and add to tracking
        for link in result.links:
            link.url = self.normalize_url(url, link.url)
            link.parent_url = url
            
        return result

    async def validate_page(self, url: str, content: str, depth: int, parent_url: str) -> ValidationResult:
        """Enhanced validation with depth, parent context, and content truncation"""
        content_tokens = count_tokens(content)
        truncated = False
        
        if content_tokens > 15000:
            enc = tiktoken.encoding_for_model("gpt-4")
            tokens = enc.encode(content)
            truncated_tokens = tokens[:1000]
            content = enc.decode(truncated_tokens)
            truncated = True

        truncation_notice = "\n## NOTE: Content truncated from {original_tokens} to 5000 tokens due to length. This may indicate a comprehensive index page. ##".format(
            original_tokens=content_tokens
        ) if truncated else ""
        
        system_instructions = VALIDATOR_PROMPT.format(
            url=url,
            content=content + truncation_notice,
            depth=depth,
            parent_url=parent_url
        )

        validator_llm = structured_llm_large(ValidationResult)
        result = validator_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Validate if this page is index where all of the articles live.")
        ])

        if truncated and 0.8 <= result.confidence <= 0.85:
            result.confidence = 0.95
            result.reasoning += "\nNOTE: Confidence adjusted upward due to page length requiring truncation."

        return result

    async def get_multi_agent_decision(
        self,
        agent_states: Dict[str, AgentState],
        available_links: List[ExtractedLink]
    ) -> MultiAgentDecision:
        """Orchestrate decisions across all agents based on their current states"""
        MULTI_AGENT_ORCHESTRATOR_PROMPT = """You are a high-level orchestrator managing multiple web crawling agents searching for an article index page.

        Current Agent States:
        {agent_states}

        Available Unexplored Links:
        {available_links}

        Your task is to:
        1. Analyze the validation results from all agents
        2. Determine if any agent has found the index page (confidence >= 0.95)
        3. If multiple agents have a high-confidence result, review the validation key for each agent to determine the best choice. For example, if one agent states that the index contains 10 articles while the other has identified 1000 articles, the latter is likely the correct choice.
        4. If not found, decide optimal next actions:
            - Which agents should explore deeper from their current position
            - Which agents should explore new links from the initial set
            - How to distribute agents across promising paths
        
        Previous exploration history:
        {exploration_history}

        Expected output:
        - action: 'terminate' if index found, 'explore_new' or 'explore_deeper'
        - target_agent_ids: list of agents that should take action
        - target_urls: map of agent_id to next unique URL (if applicable).
        - rationale: detailed explanation, if there is a winner include the rationale for the winner agent 
        - confidence: 0-1 score 
        - winner_agent_id: agent_id of the agent that has found the index
        """

        # Format agent states for prompt
        agent_states_str = "\n".join([
            f"Agent {state.agent_id}:"
            f"\n- Current URL: {state.current_url}"
            f"\n- Depth: {state.current_depth}"
            f"\n- Validation: {state.validation_result.dict() if state.validation_result else 'None'}"
            for state in agent_states.values()
        ])

        # Format available links
        links_str = "\n".join([
            f"- {link.url} (depth_value: {link.depth_value})"
            for link in available_links
            if link.url not in self.visited_urls
        ])

        # Format exploration history
        history_str = "\n".join([
            f"Attempt {i}: {attempt['url']} - Valid: {attempt['validation_result']['is_valid']}"
            f" (confidence: {attempt['validation_result']['confidence']})"
            for i, attempt in enumerate(self.previous_attempts)
        ])

        # print(f"Agent states: {agent_states_str}")

        # print(f"Available links: {links_str}")

        # print(f"Exploration history: {history_str}")

        system_instructions = MULTI_AGENT_ORCHESTRATOR_PROMPT.format(
            agent_states=agent_states_str,
            available_links=links_str,
            exploration_history=history_str
        )

        orchestrator_llm = structured_llm_large(MultiAgentDecision)
        result = orchestrator_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content="Determine next actions for all agents based on their current states.")
        ])

        return result

    async def run_agent(self, agent_id: str, url: str):
        """Run a single agent's exploration of a URL"""
        agent = self.agents[agent_id]
        
        # Update agent state
        agent.current_url = url
        agent.visited_urls.append(url)
        self.visited_urls.add(url)

        # Fetch and validate content
        content = await self.fetch_content(url)
        validation_result = await self.validate_page(
            url=url,
            content=content,
            depth=agent.current_depth,
            parent_url=agent.parent_url
        )

        # Update agent state
        agent.validation_result = validation_result

        # Record attempt
        self.previous_attempts.append({
            "agent_id": agent_id,
            "url": url,
            "parent_url": agent.parent_url,
            "depth": agent.current_depth,
            "depth_value": agent.depth_value,
            "content": content,
            "validation_result": validation_result.model_dump()
        })

        # Extract new links if needed
        if not validation_result.is_valid or validation_result.confidence < 0.8:
            new_links = await self.extract_links(url, content)
            for new_link in new_links.links:
                if new_link.url not in self.visited_urls:
                    self.exploration_links.append(new_link)

    async def save_results(self, winner_agent_id: Optional[str] = None):
        """Save exploration results to files with winner agent information"""
        if not self.previous_attempts or not winner_agent_id:
            logger.info("No successful exploration to save")
            return
            
        # Find the winning attempt
        winning_attempt = next(
            (attempt for attempt in self.previous_attempts 
             if attempt.get('agent_id') == winner_agent_id),
            None
        )
        
        if not winning_attempt:
            logger.error(f"Could not find attempt for winning agent {winner_agent_id}")
            return

        validation_result = winning_attempt.get('validation_result', {})
        
        output = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_attempts": len(self.previous_attempts),
            "successful_url": winning_attempt['url'],
            "winner_agent_id": winner_agent_id,
            "attempts": self.previous_attempts
        }
        
        # Save JSON output
        json_output = {
            "timestamp": output["timestamp"],
            "total_attempts": output["total_attempts"],
            "successful_url": output["successful_url"],
            "winner_agent_id": output["winner_agent_id"],
            "attempts": [{
                "url": attempt["url"],
                "parent_url": attempt["parent_url"],
                "depth": attempt["depth"],
                "depth_value": attempt["depth_value"],
                "validation_result": attempt["validation_result"],
                "agent_id": attempt.get("agent_id")
            } for attempt in self.previous_attempts]
        }
        
        with open('exploration_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)
            
        # Save detailed text output
        with open('exploration_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"=== SUCCESSFUL INDEX PAGE FOUND ===\n")
            f.write(f"URL: {winning_attempt['url']}\n")
            f.write(f"Winning Agent ID: {winner_agent_id}\n")
            f.write(f"Validation Confidence: {validation_result['confidence']}\n")
            f.write(f"Total Attempts: {len(self.previous_attempts)}\n\n")
            
            f.write(f"{'='*50} Winning Attempt Details {'='*50}\n")
            f.write(f"URL: {winning_attempt['url']}\n")
            f.write(f"Parent URL: {winning_attempt['parent_url']}\n")
            f.write(f"Depth: {winning_attempt['depth']}\n")
            f.write(f"Depth Value: {winning_attempt['depth_value']}\n")
            
            f.write(f"\nValidation Results:\n")
            f.write(f"- Valid: {validation_result['is_valid']}\n")
            f.write(f"- Confidence: {validation_result['confidence']}\n")
            f.write(f"- Reasoning: {validation_result['reasoning']}\n")
            if validation_result.get('recommendations'):
                f.write("- Recommendations:\n")
                for rec in validation_result['recommendations']:
                    f.write(f"  * {rec}\n")
            
            if 'content' in winning_attempt:
                f.write(f"\nContent Length: {len(winning_attempt['content'])} characters\n")
                f.write("\nContent Preview (first 500 chars):\n")
                f.write(f"{winning_attempt['content'][:500]}...\n")
            
            # Add a summary of all attempts
            f.write(f"\n{'='*50} All Attempts Summary {'='*50}\n")
            for i, attempt in enumerate(self.previous_attempts, 1):
                f.write(f"\nAttempt {i}:\n")
                f.write(f"Agent: {attempt.get('agent_id', 'Unknown')}\n")
                f.write(f"URL: {attempt['url']}\n")
                f.write(f"Valid: {attempt['validation_result']['is_valid']}\n")
                f.write(f"Confidence: {attempt['validation_result']['confidence']}\n")
            
            f.write(f"\n{'='*100}\n")

    async def run(self):
        """Main execution loop"""
        try:
            logger.info("Starting multi-agent web crawling")
            
            # Initialize agents
            self.initialize_agents()
            
            # Initial content fetch and link extraction
            start_url = 'https://ddosecrets.com/'
            markdown_content = await self.fetch_content(start_url)

            # TODO i think extract links shoudl be replaced with a func that parses links from the markdown content
            initial_extracted = await self.extract_links(start_url, markdown_content)
            self.exploration_links = initial_extracted.links

            while len(self.previous_attempts) < self.max_attempts:
                # Get orchestrator decision based on all agent states
                decision = await self.get_multi_agent_decision(
                    agent_states=self.agents,
                    available_links=self.exploration_links
                )

                logger.info(f"Multi-agent decision: {decision.action}")
                logger.info(f"Rationale: {decision.rationale}")

                if decision.action == 'terminate':
                    logger.info(f"Successfully found the correct index page with agent {decision.winner_agent_id}!")
                    self.winner_agent_id = decision.winner_agent_id
                    break

                # Execute decisions for each targeted agent
                tasks = []
                for agent_id in decision.target_agent_ids:
                    if agent_id in decision.target_urls:
                        tasks.append(self.run_agent(agent_id, decision.target_urls[agent_id]))

                # Run agent tasks concurrently
                await asyncio.gather(*tasks)

            # Save results with winner agent id
            await self.save_results(winner_agent_id=getattr(self, 'winner_agent_id', None))

            if len(self.previous_attempts) >= self.max_attempts:
                logger.warning("Reached maximum attempts without finding correct page")
            
        except Exception as e:
            logger.error(f"Error in crawler execution: {str(e)}")
            raise

if __name__ == "__main__":
    crawler = IndexCrawler(num_agents=5)
    asyncio.run(crawler.run())