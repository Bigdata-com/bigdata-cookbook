import logging
import os
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Tuple, Any

from src.summary.token_manager import TokenManager
from src.response.response_prompts import topic_condense_by_company, response_summary_by_company

class CompanyResponseProcessor:
    """
    Class to process condensed topics and response summaries for companies.
    
    This class contains asynchronous methods for:
      • Condensing company topic summaries (i.e. masking the company name)
      • Generating a response summary based on text chunks
      
    The routines call out to external helper functions via an executor.
    """
    def __init__(self, model: str, api_key: str, logger: Any, max_tokens: int = 100000, verbose: bool = False) -> None:
        """
        Initialize the CompanyResponseProcessor.
        
        Args:
            model (str): The model identifier.
            api_key (str): API key for the external service.
            max_tokens (int, optional): Maximum tokens for processing. Defaults to 100000.
            verbose (bool, optional): Enables logging verbosity.
        """
        self.model = model
        self.api_key = api_key
        self.verbose = verbose
        self.logger = logger
        # Uncomment and adjust the basicConfig as needed:
        # logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        # Initialize a TokenManager instance to help split text into chunks.
        self.token_manager = TokenManager(model_name=self.model, max_tokens=max_tokens, verbose=verbose)
    
    #
    # CONDENSE ROUTINES
    #
    
    async def async_condense_topic(self, text: str, company_name: str) -> str:
        """
        Asynchronously condense a topic using an external function.
        
        Args:
            text (str): The input text.
            company_name (str): The name of the company.
        
        Returns:
            str: The condensed topic summary.
        """
        self.logger.debug("Executing async_condense_topic for company '%s'", company_name)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, topic_condense_by_company, text, self.model, self.api_key, company_name
        )
    
    async def condense_entity_topic(self, entity: Any, topic: str, df_topics_companies: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process a single entity and topic:
          - Build a text string from the topic summaries,
          - Mask the company name,
          - Condense the resulting text.
        
        Args:
            entity (Any): An object having at least attributes `id` and `name`.
            topic (str): The topic label.
            df_topics_companies (pd.DataFrame): DataFrame with company topic summaries.
        
        Returns:
            Optional[pd.DataFrame]: A DataFrame with an added column 'company_response_question' or None if no text.
        """
        self.logger.info("Condensing topic '%s' for entity '%s'", topic, entity.name)
        df_to_summarize = df_topics_companies.loc[
            (df_topics_companies.entity_id == entity.id) & (df_topics_companies.topic == topic)
        ].copy()
        
        # Combine topic summaries and mask the company name.
        text_to_summarize = "\n\n".join(df_to_summarize['topic_summary'].astype(str))
        text_to_summarize = text_to_summarize.replace(entity.name, 'Target Company')
        
        if not text_to_summarize:
            self.logger.warning("No text found to condense for entity '%s' and topic '%s'", entity.name, topic)
            return None
        
        try:
            summary = await self.async_condense_topic(text_to_summarize, entity.name)
        except Exception as e:
            self.logger.error("Error condensing topic for '%s' -- %s", entity.name, e)
            return None

        df_to_summarize['company_response_question'] = summary
        return df_to_summarize

    async def condense_topic_by_company(self, df_by_company: pd.DataFrame, list_entities: List[Any]) -> pd.DataFrame:
        """
        Process all companies and topics repeatedly to generate condensed topic summaries.
        
        Args:
            df_by_company (pd.DataFrame): Input DataFrame that contains a column "topic".
            list_entities (List[Any]): List of entity objects.
        
        Returns:
            pd.DataFrame: Combined DataFrame with updated company condensed summaries.
        """
        self.logger.info("Starting condense_topic_by_company processing for %d entities", len(list_entities))
        
        tasks = []
        # For each entity, process each unique topic present in df_by_company.
        for entity in list_entities:
            entity_topics = list(df_by_company.loc[df_by_company.entity_id == entity.id].topic.unique())
            for topic in entity_topics:
                tasks.append(self.condense_entity_topic(entity, topic, df_by_company))
        
        results = await asyncio.gather(*tasks)
        # Filter out tasks that returned None
        valid_results = [res for res in results if res is not None]
        if not valid_results:
            self.logger.warning("No condensed topics were generated; returning empty DataFrame.")
            return pd.DataFrame()
        
        new_df_by_company = pd.concat(valid_results, ignore_index=True)
        return new_df_by_company
    
    async def async_response_summary(self, text: str, n: int, topic: str, topic_summary: str, entity_name: str) -> str:
        """
        Asynchronously obtain a response summary based on the given text.
        
        Args:
            text (str): The input text.
            n (int): The number of reports/documents.
            topic (str): The topic label.
            topic_summary (str): The topic summary.
            entity_name (str): The name of the company.
        
        Returns:
            str: The response summary.
        """
        self.logger.debug("Executing async_response_summary for company '%s' and topic '%s'", entity_name, topic)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, response_summary_by_company, text, self.model, n, self.api_key, topic, topic_summary, entity_name
        )
    
    async def run_response_summaries(self, text: str, n_reports: int, topic: str, topic_summary: str, entity_name: str) -> Tuple[str]:
        """
        Run a response summary on one text chunk and return a formatted result.
        
        Args:
            text (str): A text chunk.
            n_reports (int): The number of reports.
            topic (str): The topic label.
            topic_summary (str): The topic summary.
            entity_name (str): The name of the company.
        
        Returns:
            Tuple[str]: A tuple containing a formatted response summary.
        """
        self.logger.debug("Running response summary chunk for company '%s' and topic '%s'", entity_name, topic)
        try:
            response = await self.async_response_summary(text, n_reports, topic, topic_summary, entity_name)
        except Exception as e:
            self.logger.error("Error in async_response_summary for company '%s' and topic '%s': %s", entity_name, topic, e)
            response = ""
        
        formatted_response = (
            f"--- Report Start ---\n\n Headline: \n  Text: {response}\n --- Report End ---"
        )
        return (formatted_response,)
    
    async def process_entity_response(
        self, 
        entity: Any, 
        topic: str, 
        df_topics_companies: pd.DataFrame, 
        df_by_company: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Process the response summary for a single entity and topic.
        
        Args:
            entity (Any): An object with attributes `id` and `name`.
            topic (str): The topic label.
            df_topics_companies (pd.DataFrame): DataFrame with topic information.
            df_by_company (pd.DataFrame): DataFrame with pre-generated topic summaries.
        
        Returns:
            Optional[pd.DataFrame]: A DataFrame row with the response summary or None if data is missing.
        """
        self.logger.debug("Processing response summary for entity '%s' and topic '%s'", entity.name, topic)
        
        df_to_summarize = df_topics_companies.loc[
            (df_topics_companies.entity_id == entity.id) & (df_topics_companies.label_by_company == topic)
        ]
        
        # Ensure there is a pre-existing topic summary from df_by_company.
        company_rows = df_by_company.loc[
            (df_by_company.entity_id == entity.id) & (df_by_company.topic == topic)
        ]
        if company_rows.shape[0] == 0:
            self.logger.warning("No topic summary present for entity '%s' and topic '%s'", entity.name, topic)
            return None
        else:
            topic_summary = company_rows.topic_summary.iloc[0]
        
        n_documents = df_to_summarize.document_id.nunique()
        
        # Prepare text to be summarized by combining headline and text.
        df_text_to_summarize = df_to_summarize[['headline', 'text']].drop_duplicates()
        df_text_to_summarize['text_to_summarize'] = (
            '--- Report Start ---\n'
            ' Headline: ' + df_text_to_summarize.headline.astype(str) +
            '\n Text: ' + df_text_to_summarize.text.astype(str) +
            '\n --- Report End ---'
        )
        n_reports = df_text_to_summarize.shape[0]
        text_to_summarize_full = "\n\n".join(df_text_to_summarize['text_to_summarize'])
        
        if not text_to_summarize_full:
            self.logger.warning("No text available for response summary for entity '%s' and topic '%s'", entity.name, topic)
            return None
        
        # Use a token manager (a new instance) to break the text into chunks if needed.
        token_manager = TokenManager(model_name=self.model, max_tokens=100000, verbose=self.verbose)
        list_text_to_summarize = token_manager.split_text_by_chunks(df_text_to_summarize)
        
        if len(list_text_to_summarize) > 1:
            self.logger.debug("Splitting text into %d chunks for entity '%s' and topic '%s'",
                              len(list_text_to_summarize), entity.name, topic)
            result_futures = [
                asyncio.create_task(
                    self.run_response_summaries(chunk, n_reports, topic, topic_summary, entity.name)
                )
                for chunk in list_text_to_summarize
            ]
            results = await asyncio.gather(*result_futures)
            # Each result is a tuple with one element; zip them together and join.
            # For example, if results = [(A,), (B,), (C,)], then we obtain (A, B, C)
            formatted_chunks = list(zip(*results))[0]
            text_to_summarize_response = "\n\n".join(formatted_chunks)
        else:
            text_to_summarize_response = list_text_to_summarize[0]
        
        try:
            summary = await self.async_response_summary(
                text_to_summarize_response, n_reports, topic, topic_summary, entity.name
            )
        except Exception as e:
            self.logger.error("Error generating response summary for entity '%s' and topic '%s': %s", entity.name, topic, e)
            summary = ""
        
        df_output = pd.DataFrame({
            'entity_id': [entity.id],
            'entity_name': [entity.name],
            'topic': [topic],
            'response_summary': [summary],
            'n_response_documents': [n_documents]
        })
        return df_output
    
    async def process_response_by_company(
        self,
        df_labeled: pd.DataFrame,
        df_by_company: pd.DataFrame,
        list_entities: List[Any],
        import_from_path: Optional[str] = None,
        export_to_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process response summaries for all companies and topics.
        
        Optionally imports pre-processed data from a pickle file and exports the finalized DataFrame.
        
        Args:
            df_labeled (pd.DataFrame): The original labeled DataFrame.
            df_by_company (pd.DataFrame): The DataFrame containing pre-generated topic summaries.
            list_entities (List[Any]): A list of entity objects.
            import_from_path (Optional[str], optional): File path to import prior results.
            export_to_path (Optional[str], optional): File path for exporting the final DataFrame.
        
        Returns:
            pd.DataFrame: Combined DataFrame with response summaries.
        """
        if import_from_path and os.path.isfile(import_from_path):
            self.logger.info("Importing response data from %s", import_from_path)
            return pd.read_pickle(import_from_path)
        
        # Create a DataFrame subset with topics that are valid.
        df_topics_companies = df_labeled.loc[
            ~df_labeled.label.isin(['', 'unassigned', 'unclear'])
        ].copy()
        # Create a composite label (or key) for the company-topic combination.
        df_topics_companies['label_by_company'] = df_topics_companies.apply(
            lambda x: f"{x.theme} - {x.label}", axis=1
        )
        
        tasks = []
        for entity in list_entities:
            topics = list(
                df_topics_companies.loc[df_topics_companies.entity_id == entity.id].label_by_company.unique()
            )
            for topic in topics:
                tasks.append(
                    self.process_entity_response(entity, topic, df_topics_companies, df_by_company)
                )
        self.logger.info("Starting process_response_by_company with %d tasks...", len(tasks))
        results = await asyncio.gather(*tasks)
        # Filter out None responses.
        valid_results = [res for res in results if res is not None]
        if not valid_results:
            self.logger.warning("No response data generated; returning an empty DataFrame.")
            return pd.DataFrame({'entity_id': [], 'entity_name': [], 'topic': [], 'response_summary': [],'n_response_documents': []})
        
        final_df = pd.concat(valid_results, ignore_index=True)
        
        if export_to_path:
            self.logger.info("Exporting response summary to %s", export_to_path)
            final_df.to_pickle(export_to_path)
        
        return final_df
