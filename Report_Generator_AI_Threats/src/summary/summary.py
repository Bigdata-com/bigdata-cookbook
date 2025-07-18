import logging
import os
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Tuple, Any

from src.summary.token_manager import TokenManager
from src.summary.summary_prompts import topic_summary_by_company


## Company-Specific Issues

class SummarizerCompany:
    """
    Class to process company topics by asynchronously summarizing texts,
    assessing risks and uncertainties, and returning results as a DataFrame.
    """
    def __init__(self, model: str, api_key: str, logger: Any, max_tokens: int = 100000, verbose: bool = False) -> None:
        """
        Initialize the TopicSummarizerCompany.

        Args:
            model (str): The model identifier.
            api_key (str): The API key.
            max_tokens (int, optional): Maximum tokens for processing. Defaults to 100000.
            verbose (bool, optional): Flag to enable verbose logging. Defaults to False.
        """
        self.model = model
        self.api_key = api_key
        self.logger = logger
        self.verbose = verbose
        self.token_manager = TokenManager(model_name=self.model, max_tokens=max_tokens, verbose=verbose)


    async def async_topic_summary(self, text: str, n: int, topic: str, entity_name: str, focus: str) -> str:
        """
        Asynchronously produce a topic summary.

        Args:
            text (str): The text input.
            n (int): The number of documents/reports.
            topic (str): The topic label.
            entity_name (str): The name of the company/entity.
            focus (str): The focus of the summary

        Returns:
            str: The summary text.
        """
        loop = asyncio.get_event_loop()
        self.logger.debug("Running async_topic_summary for entity '%s' and topic '%s'", entity_name, topic)
        return await loop.run_in_executor(
            None, topic_summary_by_company, text, self.model, n, self.api_key, topic, entity_name, focus
        )


    async def run_summaries(self, text: str, n_reports: int, topic: str, entity_name: str, focus: str) -> Tuple[str]:
        """
        Run the summary, risk, and uncertainty tasks in parallel on one text chunk.

        Args:
            text (str): The text chunk.
            n_reports (int): The number of reports.
            topic (str): The topic label.
            entity_name (str): The entity name.
            focus (str): The focus of the summary

        Returns:
            Tuple[str]: Formatted report strings for topic.
        """
        self.logger.debug("Running summaries for entity '%s' and topic '%s'", entity_name, topic)
        topic_summary = await self.async_topic_summary(text, n_reports, topic, entity_name, focus)

        formatted_topic = (
            f"--- Report Start ---\n\n Headline: \n  Text: {topic_summary}\n --- Report End ---"
        )

        return formatted_topic


    async def process_entity_topic(self, entity: Any, topic: str, df_topics_companies: pd.DataFrame, focus: str) -> Optional[pd.DataFrame]:
        """
        Process a single entity for a single topic by generating summaries and packaging them in a DataFrame.

        Args:
            entity (Any): An object with attributes id and name.
            topic (str): The topic label.
            df_topics_companies (pd.DataFrame): DataFrame containing topic/company info.
            focus (str): The focus of the summary

        Returns:
            Optional[pd.DataFrame]: A single-row DataFrame with summary results or None if no documents found.
        """
        self.logger.debug("Processing topic '%s' for entity '%s'", topic, entity.name)
        df_to_summarize = df_topics_companies.loc[
            (df_topics_companies.entity_id == entity.id)
        ]
        n_documents = df_to_summarize.document_id.nunique()
        # avg_sentiment = df_to_summarize.sentiment.mean()

        if df_to_summarize.empty:
            self.logger.warning("No documents found for entity '%s' and topic '%s'", entity.name, topic)
            return None

        # Prepare the text to be summarized.
        df_text_to_summarize = df_to_summarize[['headline', 'text']].drop_duplicates()
        df_text_to_summarize['text_to_summarize'] = (
            '--- Report Start ---\n'
            ' Headline: ' + df_text_to_summarize.headline.astype(str) +
            '\n Text: ' + df_text_to_summarize.text.astype(str) +
            '\n --- Report End ---'
        )
        n_reports = df_text_to_summarize.shape[0]
        text_to_summarize_full = "\n\n".join(df_text_to_summarize['text_to_summarize'])

        # Use the token manager to break the text into chunks if necessary.
        list_text_to_summarize = self.token_manager.split_text_by_chunks(df_text_to_summarize)
        if len(list_text_to_summarize) > 1:
            tasks = [
                asyncio.create_task(self.run_summaries(chunk, n_reports, topic, entity.name))
                for chunk in list_text_to_summarize
            ]
            self.logger.debug("Processing %d text chunks for entity '%s' and topic '%s'",
                              len(list_text_to_summarize), entity.name, topic)
            results = await asyncio.gather(*tasks)
            topic_chunks = zip(*results)
            text_to_summarize_topic = "\n\n".join(topic_chunks)
        else:
            text_to_summarize_topic = text_to_summarize_full

        # Generate summaries; use try/except to log any errors.
        try:
            summary = await self.async_topic_summary(text_to_summarize_topic, n_reports, topic, entity.name, focus)
        except Exception as e:
            self.logger.error("Error during topic summary for entity '%s' and topic '%s': %s", entity.name, topic, e)
            summary = ""

        df_by_company = pd.DataFrame({
            'entity_id': [entity.id],
            'entity_name': [entity.name],
            'topic': [topic],
            'topic_summary': [summary],
            'n_documents': [n_documents]
            # 'avg_sentiment': [avg_sentiment]
        })

        return df_by_company


    async def process_by_company(
        self,
        df_labeled: pd.DataFrame,
        list_entities: List[Any],
        theme: str,
        focus: str,
        import_from_path: Optional[str] = None,
        export_to_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process all entities and topics; optionally import/export results from/to a pickle file.

        Args:
            df_labeled (pd.DataFrame): The labeled DataFrame.
            list_entities (List[Any]): A list of entity objects.
            theme (str): The theme of the summary
            focus (str): The focus of the summary
            import_from_path (Optional[str], optional): Path to a pickle file to import previous results.
            export_to_path (Optional[str], optional): Path to export the results as a pickle file.

        Returns:
            pd.DataFrame: Combined DataFrame with processed results.
        """
        # Import previous results if the file exists.
        if import_from_path and os.path.isfile(import_from_path):
            self.logger.info("Importing summaries by company from pickle file")
            return pd.read_pickle(import_from_path)

        self.logger.info("Preparing topics from the labeled DataFrame")
        df_topics_companies = df_labeled.loc[~df_labeled.label.isin(['', 'unassigned', 'unclear'])].copy()

        tasks = []
        for entity in list_entities:
            tasks.append(self.process_entity_topic(entity, theme, df_topics_companies, focus))

        self.logger.info("Starting processing for %d tasks...", len(tasks))
        results = await asyncio.gather(*tasks)

        # Filter out any None responses.
        df_list = [df for df in results if df is not None]
        if not df_list:
            self.logger.warning("No data was processed; returning an empty DataFrame.")
            return pd.DataFrame()

        df_by_company = pd.concat(df_list, ignore_index=True)

        if export_to_path:
            self.logger.info("Exporting processed data to %s", export_to_path)
            df_by_company.to_pickle(export_to_path)

        return df_by_company
        