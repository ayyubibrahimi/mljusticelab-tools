import json
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import asyncio
from concurrent.futures import ThreadPoolExecutor
from llm import gpt_4o_mini, gpt_4o
from langchain_core.messages import HumanMessage, SystemMessage
from news_node import FetchNews

from dotenv import load_dotenv
load_dotenv()


structured_llm = gpt_4o_mini.with_structured_output
structured_llm_large = gpt_4o_mini.with_structured_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThemeAnalysis(BaseModel):
    primary_data_sources: List[str]
    where_to_collect_data: List[str]
    url_citation: List[str]

class ThemeAnalysisCollection(BaseModel):
    themes: List[ThemeAnalysis] = Field(description="Collection of discovered themes")

class ArticleBatch(BaseModel):
    batch_id: str
    articles: List[Dict]
    discovered_themes: List[ThemeAnalysis]
    
class ResearchSummary(BaseModel):
    consolidated_data_sources: List[str]
    high_priority_collection_targets: List[str]
    source_citations: List[str]

class ArticleOutline(BaseModel):
    priority_data_types: str
    where_to_collect_data: List[str]

class GeneratedArticle(BaseModel):
    title: str
    content: str


class ArticleGenerator:
    def __init__(self, archive_path: str, num_parallel_agents: int = 5):
        self.archive_path = archive_path
        self.num_parallel_agents = num_parallel_agents
        self.articles_data = None
        self.research_summary = None
        self.article_outline = None
        self.generated_article = None
        self.executor = ThreadPoolExecutor(max_workers=num_parallel_agents)
        self.news_collector = FetchNews()
        self.current_topic = None

    async def update_current_topic(self):
        """Update the current topic based on news analysis"""
        try:
            self.current_topic = await self.news_collector.run()
            # self.current_topic = "panedemic"
            logger.info(f"Updated current topic to: {self.current_topic}")
            
        except Exception as e:
            logger.error(f"Error updating current topic: {str(e)}")
            self.current_topic = "current global events"

    async def load_archive(self):
        """Load and validate the article archive"""
        try:
            with open(self.archive_path, 'r') as f:
                self.articles_data = json.load(f)
            logger.info(f"Loaded {self.articles_data['total_articles']} articles from archive")
        except Exception as e:
            logger.error(f"Error loading archive: {str(e)}")
            raise

    async def create_article_batches(self) -> List[ArticleBatch]:
        """Split articles into batches for parallel processing"""
        articles = self.articles_data['articles']
        batch_size = len(articles) // self.num_parallel_agents
        print(batch_size)
        
        batches = []
        for i in range(0, len(articles), batch_size):
            batch = ArticleBatch(
                batch_id=f"batch_{len(batches)}",
                articles=articles[i:i + batch_size],
                discovered_themes=[]
            )
            batches.append(batch)
    
        
        return batches

    async def analyze_data_collected(self, batch: ArticleBatch) -> ArticleBatch:
        """Analyze themes in a batch of articles"""
        THEME_ANALYSIS_PROMPT = """
You are a data collection specialist analyzing successful information gathering strategies from previous investigations.

Context:
You are reviewing summaries of previously collected datasets to identify valuable data sources and where similar data can be found. Your analysis will help teams understand what types of data to prioritize and where to look for it.

Input:
{articles}  # List of dataset summaries with URLs

Task:
Analyze these summaries to identify:
1. High-value data source types and their typical characteristics
2. Specific organizations or entity types where similar data might be found
3. Support your recommendations with citations to the source articles


Guidelines:
1. Only include data sources and locations that are directly supported by the input articles
2. Be specific about data characteristics and volumes when available
3. Each recommendation should reference at least one source article
4. Focus on practical, actionable collection targets

Remember: The goal is to provide clear, evidence-based guidance on what data to collect and where to find it, backed by real examples from the source articles.

You must return each of the following components:
- primary_data_sources:
  [List format: "Data type | Typical volume | Key characteristics"]
  Example: "Email servers | 100k-500k emails | Often contains internal communications and attachments"
  Example: "Mobile device backups | 1-10GB | Contains chat logs, photos, and cached data"

- where_to_collect_data:
  [List specific organizations, agencies, or entity types]
  Example: "Maritime shipping companies with international operations"
  Example: "Regional environmental protection agencies"

- url_citation:
  [List the citation to the source article]

Return your response in this format:

- primary_data_sources:
- where_to_collect_data:
- url_citation:

"""

        formatted_articles = "\n".join([
            f"Title: {article['title']}\nAbstract: {article['abstract']}\nUrl: {article['url']}"
            for article in batch.articles
        ])

        # print(f"Formatted articles: {formatted_articles}")

        system_instructions = THEME_ANALYSIS_PROMPT.format(
            articles=formatted_articles
        )

        # Change this to use ThemeAnalysisCollection instead
        analyzer_llm = structured_llm(ThemeAnalysisCollection)
        themes_collection = analyzer_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content=f"Analyze these articles"),
            # HumanMessage(content=f"Analyze these articles related to this topic {self.current_topic}")
        ])

        # print(f"Themes collection for batch {batch.batch_id}: {themes_collection}")

        # themes_collection will have a .themes property that contains List[ThemeAnalysis]
        batch.discovered_themes = themes_collection
        return batch

    async def synthesize_research(self, analyzed_batches: List[ArticleBatch]) -> ResearchSummary:
        """Synthesize themes from all batches into research summary"""
        RESEARCH_SYNTHESIS_PROMPT = """You are a data collection specialist synthesizing findings from multiple analyses to create a comprehensive data collection strategy.

Context:
You are consolidating multiple batches of analysis about valuable data sources and collection targets. Each batch contains:
- Lists of primary data types that were collected
- Specific organizations/entities where data was found
- URLs citing these findings

Input:
{batch_analyses}  # Collection of analyses from multiple batches

Task:
Synthesize these analyses to:
1. Identify the most valuable and recurring data sources across all batches
2. Consolidate recommendations for where to find similar data
3. Maintain traceability to source articles via URLs

Required Output Format:
- consolidated_data_sources:
  [List combining similar data types with their characteristics and frequency across batches]
  Example: "Email servers (found in 8/10 batches) | Typical volume 100k-1M | Most valuable for internal communications"

- high_priority_collection_targets:
  [List of most promising organizations/entities, ordered by frequency of mention]
  Example: "Maritime shipping companies | Mentioned in 5 sources | Consistently yield large email datasets"

- source_citations:
  [List of supporting URLs grouped by finding]
  Example: "Large email datasets in shipping sector: url1, url2, url3"

Guidelines:
1. Prioritize data sources and locations mentioned across multiple batches
2. Group similar data types and collection targets together
3. Maintain links between findings and source articles
4. Focus on practical, actionable insights

Remember: The goal is to create a consolidated view of the most promising data types and collection targets, supported by evidence from source articles."""

        # TODO this could be converted to pairwise comparison where we synthesize and condense after consolidation 
        batch_analyses = "\n".join([
            f"Batch {batch.batch_id} Themes:\n" +
            "\n".join([f"- {theme})"
                      for theme in batch.discovered_themes])
            for batch in analyzed_batches
        ])

        # print(batch_analyses)

        system_instructions = RESEARCH_SYNTHESIS_PROMPT.format(
            topic=self.current_topic,
            batch_analyses=batch_analyses
        )

        synthesis_llm = structured_llm(ResearchSummary)
        research_summary = synthesis_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content=f"Synthesize the research about {self.current_topic}")
        ])

        # print(f"Research summary: {research_summary}")

        return research_summary

    async def plan_article(self, research_summary: ResearchSummary) -> ArticleOutline:
        """Plan article structure based on research summary"""
        ARTICLE_PLANNER_PROMPT = """You are a senior data collection strategist creating a detailed collection plan for your team. This plan will serve as the definitive guide for what data to collect and where to find it.

        Context:
        You have analyzed patterns across multiple data collection efforts and identified high-value data sources and collection targets.

        Input:
        Research Summary: {research_summary}

        Task:
        Create a comprehensive data collection plan that maps data types to specific collection targets. The plan should be organized by data type and provide specific guidance on where and how to find each type of data.
        
        Theme summary:
        {research_summary}.

        Your task is to create comprehensive data collection plan.

        Your data collection plan should include two components: 
        
        1. "Priority Data Types" should list each data type with:
        - Typical volume ranges
        - Value proposition
        - Historical success rate

        2. "Where to collect data" should map each data type to:
        - Specific organizations/entities
        - Industry sectors
        - Geographic regions
        - Supporting evidence (URLs)

        Format your outline as follows:
        priority_data_types: 
        where_to_collect_data: 
        """

        system_instructions = ARTICLE_PLANNER_PROMPT.format(
            research_summary=research_summary.model_dump_json()
        )

        planner_llm = structured_llm(ArticleOutline)
        outline = planner_llm.invoke([
            SystemMessage(content=system_instructions),
            HumanMessage(content=f"Create an article outline about")
        ])

        # print(f"Outline: {outline}")

        return outline

    async def generate_article(self, outline: ArticleOutline, research_summary: ResearchSummary) -> GeneratedArticle:
        """Generate article based on outline and research"""

        # cite articles from ddos
        # give examples of agencies, orgs, individuals that might house these data that we're interested in
        ARTICLE_GENERATOR_PROMPT = """You are a data collection strategist creating a structured collection plan for investigating: {topic}

        Input:
        Topic: {topic}
        Historical Collection Insights: {research}
        Outline: {outline}

        Guidelines:
        1. Focus on concrete specifications, not narrative
        2. All criteria must be measurable
        3. Each recommendation needs supporting evidence
        4. Cite URLs for similar successful collections
        5. Specify exact requirements, not ranges where possible

        Required components:

        1. Priority Data Types:
        [For each data type provide]:
        - Name: [specific data type]
        - Description: [what this data contains]
        - Minimum Volume: [baseline amount needed]
        - Preferred Volume: [ideal amount]
        - Format: [file types, structure]
        - Priority Level: [High/Medium/Low]
        - Historical Value: [cite previous successes]

        2. Collection Targets:
        [For each data type list]:
        - Primary Organizations: [specific entities]
        - Government Agencies: [relevant agencies]
        - Industry Sources: [sector-specific sources]
        - Geographic Focus: [regions of interest]
        - Access Methods: [how to collect]
        - URLs: [supporting evidence]

        3. Success Criteria:
        [For each data type specify]:
        - Volume Requirements:
        * Minimum acceptable size
        * File count ranges
        * Date range requirements
        - Quality Metrics:
        * Required fields/columns
        * Data completeness thresholds
        * Format requirements
        - Validation Steps:
        * How to verify authenticity
        * How to verify completeness
        * How to verify relevance
        - Rejection Criteria:
        * When to reject incomplete data
        * When to reject irrelevant data
        * When to reject poor quality data

        Return your response in the following format:
        title:
        content:

        Remember: This plan will be used to validate whether collected data meets investigation requirements for {topic}."""

        system_instructions = ARTICLE_GENERATOR_PROMPT.format(
            topic=self.current_topic,
            outline=outline.model_dump_json(),
            research=research_summary.model_dump_json()
        )

        generator_llm = structured_llm(GeneratedArticle)
        article = generator_llm.invoke([
            SystemMessage(content=system_instructions),
            # HumanMessage(content=f"Write a data collection plan {self.current_topic}")
        ])

        return article

    async def run(self):
        """Main execution pipeline"""
        try:
            # collect current topic
            logger.info("Fetching current events")
            await self.update_current_topic()

            # 1. Planning Phase
            logger.info("Starting planning phase")
            await self.load_archive()
            batches = await self.create_article_batches()

            # 2. Research Phase
            logger.info("Starting parallel research phase")
            analysis_tasks = [self.analyze_data_collected(batch) for batch in batches]
            analyzed_batches = await asyncio.gather(*analysis_tasks)

            # print(f"Output of batch absrtract analysis {analyzed_batches}")
            
            # 3. Synthesis Phase
            logger.info("Synthesizing research")
            self.research_summary = await self.synthesize_research(analyzed_batches)
            
            # 4. Planning Phase
            logger.info("Planning article")
            self.article_outline = await self.plan_article(self.research_summary)
            
            # 5. Writing Phase
            logger.info("Generating article")
            self.generated_article = await self.generate_article(
                self.article_outline,
                self.research_summary
            )

            # 6. Save Results
            self.save_results()
            
            logger.info("Article generation complete")

        except Exception as e:
            logger.error(f"Error in article generation: {str(e)}")
            raise

    def save_results(self):
        """Save generated article and supporting materials"""
        output = {
            "topic": self.current_topic,
            "research_summary": self.research_summary.dict(),
            "outline": self.article_outline.dict(),
            "article": self.generated_article.dict()
        }

        with open('generated_article.json', 'w') as f:
            json.dump(output, f, indent=2)

        with open('generated_article.md', 'w') as f:
            f.write(f"# {self.generated_article.title}\n\n")
            f.write(self.generated_article.content)

if __name__ == "__main__":
    generator = ArticleGenerator(
        archive_path="article_data.json",
        num_parallel_agents=3
    )
    asyncio.run(generator.run())