"""
Report Service for generating AI-powered commentary from news data.

This service takes structured news data (from the topic search service)
and generates:
1. Executive briefs (one bullet point per topic)
2. Wall Street-style desk note (cohesive narrative)

Using Google's Gemini AI with structured output.
"""

import asyncio
import os
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

from pydantic import BaseModel

from .llm_service import LLMService
from .llm_factory import LLMServiceFactory

logger = logging.getLogger(__name__)


class TopicBrief(BaseModel):
    """A single topic brief for a company."""
    company_name: str
    topic_name: str
    bullet_point: str


class DeskNote(BaseModel):
    """Wall Street-style desk note report."""
    report: str


class Commentary(BaseModel):
    """Complete commentary with briefs and desk note."""
    ticker: str
    company_name: str
    generated_at: str
    briefs: List[TopicBrief]
    desk_note: str


class ReportService:
    """
    Service for generating AI-powered commentary from news data.
    
    This service takes the output from the topic search service
    and generates executive briefs and Wall Street desk notes.
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        prompts_path: str = "config/prompts.yaml"
    ):
        """
        Initialize the report service.
        
        Args:
            llm_service: Optional LLMService instance. If not provided,
                        will create one using LLMServiceFactory with auto-detection.
            prompts_path: Path to prompts.yaml file
        """
        self.llm_service = llm_service or self._create_llm_service()
        self.prompts_path = prompts_path
        self.prompts = self._load_prompts()
    
    def _create_llm_service(self) -> LLMService:
        """Create an LLMService instance from environment variables using factory."""
        try:
            provider = os.getenv('LLM_PROVIDER', 'auto')
            logger.info(f"Creating LLM service with provider: {provider}")
            llm_service = LLMServiceFactory.create(provider=provider)
            logger.info(
                f"Created LLM service for commentary generation "
                f"(provider: {llm_service.provider_name}, model: {llm_service.model})"
            )
            return llm_service
        except ValueError as e:
            raise ValueError(
                "No LLM provider credentials configured. Set one of:\n"
                "  - OPENAI_API_KEY for OpenAI\n"
                "  - GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS for Gemini\n"
                "Or set LLM_PROVIDER env var to 'openai' or 'gemini'."
            ) from e
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompt templates from YAML file."""
        from pathlib import Path
        
        # Try relative path first, then absolute
        prompts_file = Path(self.prompts_path)
        if not prompts_file.is_absolute():
            # Try relative to current file
            base_dir = Path(__file__).parent.parent
            prompts_file = base_dir / "config" / "prompts.yaml"
        
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Prompts file not found: {prompts_file}")
            raise
    
    def _format_context_from_news_response(self, news_response: Dict[str, Any]) -> str:
        """
        Format context from news API response for Gemini processing.
        
        Args:
            news_response: Response from /api/news endpoint with topic_results
            
        Returns:
            Formatted context string
        """
        company_name = news_response.get('company_name', 'Unknown Company')
        topic_results = news_response.get('topic_results', [])
        
        # Group articles by topic_name (the actual topic category, not the query)
        topics_dict = {}
        for article in topic_results:
            topic_name = article.get('topic_name', 'Unknown Topic')
            if topic_name not in topics_dict:
                topics_dict[topic_name] = []
            topics_dict[topic_name].append(article)
        
        # Format context for each topic
        all_contexts = []
        for topic_name, articles in topics_dict.items():
            # Get the formatted query from the first article (all articles for a topic should have the same query)
            formatted_query = articles[0].get('topic', topic_name) if articles else topic_name
            
            context_parts = [
                "<company_name>",
                f"{company_name}",
                "<topic>",
                f"{topic_name}",  # Use topic_name (e.g., "Financial Metrics")
                "<query>",
                f"{formatted_query}",  # Use the formatted query text
            ]
            
            # Add all article full_text as answer chunks
            for article in articles:
                full_text = article.get('full_text', '')
                document_url = article.get('document_url', '')
                if full_text:
                    context_parts.append("<answer_chunk>")
                    if document_url:
                        context_parts.append(f"<source_url>{document_url}</source_url>")
                    context_parts.append(full_text)
                    context_parts.append("</answer_chunk>")
            
            all_contexts.append("\n".join(context_parts))
        
        return "\n\n".join(all_contexts)
    
    def _render_prompt(
        self,
        template: str,
        **variables
    ) -> str:
        """
        Render a prompt template with variables.
        
        Args:
            template: Template string with {{variable}} placeholders
            **variables: Variables to substitute
            
        Returns:
            Rendered prompt string
        """
        rendered = template
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"  # {{key}}
            rendered = rendered.replace(placeholder, str(value))
        return rendered
    
    async def generate_topic_briefs(
        self,
        news_response: Dict[str, Any],
        save_prompt_path: Optional[str] = None
    ) -> List[TopicBrief]:
        """
        Generate executive briefs (one per topic) from news data.
        
        Args:
            news_response: Response from /api/news endpoint
            save_prompt_path: Optional path to save the prompt (e.g., "output/job-xxx/briefing_request_prompt.md")
            
        Returns:
            List of TopicBrief objects
        """
        logger.info(f"Generating topic briefs for {news_response.get('ticker', 'unknown')}")
        
        # Format context
        context = self._format_context_from_news_response(news_response)
        
        # Load prompt template
        prompt_config = self.prompts['executive_brief']
        system_prompt = prompt_config['system_prompt']
        user_template = prompt_config['user_template']
        
        # Render prompt
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_prompt = self._render_prompt(
            user_template,
            current_datetime=current_datetime,
            report=context,
            response_format="See above for expected JSON schema"
        )
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        # Prompt logging removed - prompts are saved to files instead

        # Save prompt if path provided
        if save_prompt_path:
            try:
                from pathlib import Path
                prompt_path = Path(save_prompt_path)
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Executive Briefing Prompt\n\n")
                    f.write(f"**Generated:** {current_datetime}\n\n")
                    f.write(f"**Ticker:** {news_response.get('ticker', 'unknown')}\n")
                    f.write(f"**Company:** {news_response.get('company_name', 'unknown')}\n\n")
                    f.write("---\n\n")
                    f.write("## System Prompt\n\n")
                    f.write(f"{system_prompt}\n\n")
                    f.write("---\n\n")
                    f.write("## User Prompt\n\n")
                    f.write(f"{user_prompt}\n\n")
                    f.write("---\n\n")
                    f.write("## Full Prompt\n\n")
                    f.write(f"{full_prompt}\n")
                logger.info(f"Saved briefing prompt to {prompt_path}")
            except Exception as e:
                logger.warning(f"Failed to save briefing prompt to {save_prompt_path}: {e}")

        # Generate briefs
        briefs = await self.llm_service.generate_content_list(
            prompt=full_prompt,
            response_schema=TopicBrief
        )
        
        # Post-process: Fix topic names if LLM returned generic names like "Topic 0", "Topic 1", etc.
        # Map them back to actual topic names from the search results
        topic_results = news_response.get('topic_results', [])
        
        # Build a mapping of topic names from search results
        # Group by topic_name to get all unique topic names that were actually searched
        topic_names_from_results = {}
        for article in topic_results:
            topic_name = article.get('topic_name', 'Unknown Topic')
            if topic_name and topic_name not in topic_names_from_results:
                # Store the topic name and the query text for reference
                topic_names_from_results[topic_name] = article.get('topic', topic_name)
        
        # Get unique topic names in the order they appear in the context (for fallback matching)
        topic_names_ordered = list(topic_names_from_results.keys())
        
        # If we have briefs with generic topic names, try to map them to actual topic names
        if topic_names_ordered:
            for i, brief in enumerate(briefs):
                # Check if topic_name is generic (starts with "Topic " followed by a number)
                if brief.topic_name and brief.topic_name.startswith("Topic "):
                    topic_num_str = brief.topic_name[6:].strip()
                    if topic_num_str.isdigit():
                        topic_index = int(topic_num_str)
                        # Try to map by index first
                        if 0 <= topic_index < len(topic_names_ordered):
                            actual_topic_name = topic_names_ordered[topic_index]
                            logger.warning(f"Fixing generic topic name '{brief.topic_name}' to '{actual_topic_name}' (by index) for {news_response.get('ticker', 'unknown')}")
                            brief.topic_name = actual_topic_name
                        elif i < len(topic_names_ordered):
                            # Fallback: match by position in briefs list
                            actual_topic_name = topic_names_ordered[i]
                            logger.warning(f"Fixing generic topic name '{brief.topic_name}' to '{actual_topic_name}' (by position) for {news_response.get('ticker', 'unknown')}")
                            brief.topic_name = actual_topic_name
                        else:
                            # Last resort: use first available topic name
                            if topic_names_ordered:
                                logger.warning(f"Fixing generic topic name '{brief.topic_name}' to '{topic_names_ordered[0]}' (fallback) for {news_response.get('ticker', 'unknown')}")
                                brief.topic_name = topic_names_ordered[0]
                # Also check if the brief's topic_name is in our known topic names but might be slightly different
                elif brief.topic_name and brief.topic_name not in topic_names_from_results:
                    # Try to find a close match (case-insensitive)
                    for known_topic in topic_names_ordered:
                        if brief.topic_name.lower() == known_topic.lower():
                            logger.info(f"Normalizing topic name '{brief.topic_name}' to '{known_topic}' for {news_response.get('ticker', 'unknown')}")
                            brief.topic_name = known_topic
                            break
        
        logger.info(f"Generated {len(briefs)} topic briefs")
        return briefs
    
    async def generate_desk_note(
        self,
        briefs: List[TopicBrief],
        save_prompt_path: Optional[str] = None
    ) -> str:
        """
        Generate Wall Street-style desk note from topic briefs.
        
        Args:
            briefs: List of TopicBrief objects
            save_prompt_path: Optional path to save the prompt (e.g., "output/job-xxx/desk_request_prompt.md")
            
        Returns:
            Desk note text
        """
        logger.info(f"Generating desk note from {len(briefs)} briefs")
        
        # Format briefs for prompt
        briefs_text = "\n\n".join([
            f"Topic: {brief.topic_name}\n"
            f"Company: {brief.company_name}\n"
            f"Brief: {brief.bullet_point}"
            for brief in briefs
        ])
        
        # Load prompt template
        prompt_config = self.prompts['wallstreet_desk_note']
        system_prompt = prompt_config['system_prompt']
        user_template = prompt_config['user_template']
        
        # Render prompt
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_prompt = self._render_prompt(
            user_template,
            current_datetime=current_datetime,
            briefs=briefs_text
        )
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Save prompt if path provided
        if save_prompt_path:
            try:
                from pathlib import Path
                prompt_path = Path(save_prompt_path)
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                company_name = briefs[0].company_name if briefs else "unknown"
                ticker = briefs[0].company_name if briefs else "unknown"  # TopicBrief doesn't have ticker, use company_name
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Desk Note Request Prompt\n\n")
                    f.write(f"**Generated:** {current_datetime}\n\n")
                    f.write(f"**Company:** {company_name}\n")
                    f.write(f"**Number of Briefs:** {len(briefs)}\n\n")
                    f.write("---\n\n")
                    f.write("## System Prompt\n\n")
                    f.write(f"{system_prompt}\n\n")
                    f.write("---\n\n")
                    f.write("## User Prompt\n\n")
                    f.write(f"{user_prompt}\n\n")
                    f.write("---\n\n")
                    f.write("## Full Prompt\n\n")
                    f.write(f"{full_prompt}\n")
                logger.info(f"Saved desk note prompt to {prompt_path}")
            except Exception as e:
                logger.warning(f"Failed to save desk note prompt to {save_prompt_path}: {e}")
        
        # Generate desk note
        logger.info("Generating desk note")
        
        result = await self.llm_service.generate_content(
            prompt=full_prompt,
            response_schema=DeskNote
        )
        
        logger.info("Desk note generated successfully")
        return result.report
    
    async def generate_commentary(
        self,
        news_response: Dict[str, Any]
    ) -> Commentary:
        """
        Generate complete commentary (briefs + desk note) from news data.
        
        This is the main method to call from the API endpoint.
        
        Args:
            news_response: Response from /api/news endpoint containing:
                - ticker: Stock ticker
                - company_name: Company name
                - topic_results: List of articles grouped by topic
            
        Returns:
            Commentary object with briefs and desk_note
        """
        ticker = news_response.get('ticker', 'UNKNOWN')
        company_name = news_response.get('company_name', 'Unknown Company')
        
        logger.info(f"Generating commentary for {ticker} ({company_name})")
        
        # Generate topic briefs
        briefs = await self.generate_topic_briefs(news_response)
        
        # Generate desk note from briefs
        desk_note = await self.generate_desk_note(briefs)
        
        # Create commentary object
        commentary = Commentary(
            ticker=ticker,
            company_name=company_name,
            generated_at=datetime.now().isoformat(),
            briefs=briefs,
            desk_note=desk_note
        )
        
        logger.info(f"Commentary generated successfully for {ticker}")
        return commentary


# Convenience function for one-off commentary generation
async def generate_commentary_from_news(
    news_response: Dict[str, Any],
    llm_service: Optional[LLMService] = None
) -> Commentary:
    """
    Convenience function to generate commentary without instantiating a service.
    
    Args:
        news_response: Response from /api/news endpoint
        llm_service: Optional LLMService instance
        
    Returns:
        Commentary object
    """
    service = ReportService(llm_service=llm_service)
    return await service.generate_commentary(news_response)

