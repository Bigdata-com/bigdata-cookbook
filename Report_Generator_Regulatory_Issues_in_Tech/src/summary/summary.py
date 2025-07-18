import logging
import os
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Tuple, Any

from src.summary.token_manager import TokenManager
from src.summary.summary_prompts import topic_summary, topic_summary_by_company, topic_risk_by_company, topic_uncertainty_by_company

### Sector-Wide Issues

class TopicSummarizerSector:
    def __init__(self, model: str, api_key: str, df_labeled: pd.DataFrame, general_theme: str,
                 list_specific_focus: List[str], themes_tree_dict: dict, logger: Any):
        """
        Initializes the TopicSummarizer.

        :param df_labeled: DataFrame containing labeled sentences and topics.
        :param list_specific_themes: A list of specific themes to process for summarization.
        :param themes_tree_dict: Dictionary that maps themes to their structures.
        """
        self.model = model
        self.api_key = api_key
        self.df_labeled = df_labeled
        self.general_theme = general_theme
        self.list_specific_focus = list_specific_focus
        self.themes_tree_dict = themes_tree_dict
        self.logger = logger

    @staticmethod
    def get_most_granular_elements(tree, element):
        """
        Extracts the elements (labels or summaries) of the most granular (leaf) nodes from the taxonomy tree
        and formats them as a string list.

        Args:
            tree (dict): The taxonomy tree structure with 'Label' and 'Children'.
            element (str): The element of the tree, either 'Label' or 'Summary'

        Returns:
            str: A formatted string with each granular label prefixed by a dash.
        """
        granular_labels = []

        def traverse(node):
            # If the node has no children, it's a leaf node
            if not node.get('Children'):
                sentence = f"{node.get(element, '')}"
                granular_labels.append(sentence)
            else:
                for child in node['Children']:
                    traverse(child)

        traverse(tree)

        # Format the labels as a string list
        formatted_labels = [label for label in granular_labels]
        return formatted_labels

    def summarize(self, import_from_path: Optional[str] = None, 
                  export_to_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Summarizes topics by theme and optionally imports from or exports to a pickle file.

        :param import_from_path: Optional path to import a DataFrame from a pickle file.
        :param export_to_path: Optional path to export the DataFrame of summaries to a pickle file.
        :return: A DataFrame containing the summaries by theme and topic or None if no summaries were generated.
        """
        # Try to import existing summaries from a pickle file
        if import_from_path and os.path.isfile(import_from_path):
            self.logger.info("Importing summaries by theme from pickle file.")
            return pd.read_pickle(import_from_path)

        # Filter DataFrame to remove unassigned or unclear labels
        df_topics_themes = self.df_labeled.loc[
            ~self.df_labeled.label.isin(['', 'unassigned', 'unclear'])
        ].copy()

        # Prepare for summarization
        list_df_by_theme = []

        for focus in self.list_specific_focus:
            list_topics = list(self.themes_tree_dict[focus].get_terminal_label_summaries().keys())

            for topic in list_topics:
                df_to_summarize = df_topics_themes.loc[
                    (df_topics_themes.theme == self.general_theme + ' in ' + focus) &
                    (df_topics_themes.label == topic)
                ]

                n_documents = df_to_summarize.document_id.nunique()
                df_text_to_summarize = df_to_summarize[['headline', 'text']].drop_duplicates()
                df_text_to_summarize['text_to_summarize'] = self.prepare_text_to_summarize(df_text_to_summarize)

                n_reports = df_text_to_summarize.shape[0]
                if n_reports == 0:
                    continue

                token_manager = TokenManager(
                    model_name=self.model, # fix for new model format / Warning: will work only with openAI models
                    max_tokens=100000,
                    verbose=False
                )

                list_text_to_summarize = token_manager.split_text_by_chunks(df_text_to_summarize)

                # Summarization logic
                text_to_summarize = self.generate_summary(list_text_to_summarize, self.model, n_reports, self.api_key, self.general_theme + ' in ' + focus, topic)

                summary = topic_summary(text_to_summarize, self.model, n_reports, self.api_key, self.general_theme + ' in ' + focus, topic)

                df_by_theme = pd.DataFrame({
                    'theme': [self.general_theme + ' in ' + focus],
                    'topic': [topic],
                    'topic_summary': [summary],
                    'n_documents': [n_documents]
                })
                list_df_by_theme.append(df_by_theme)

        # Concatenate results and reset index
        if list_df_by_theme:
            df_by_theme = pd.concat(list_df_by_theme).reset_index(drop=True)

            # Export the DataFrame to a pickle file if specified
            if export_to_path:
                df_by_theme.to_pickle(export_to_path)
                self.logger.info("Exported summaries to pickle file.")

            return df_by_theme

        self.logger.warning("No summaries were produced; returning None.")
        return None

    def prepare_text_to_summarize(self, df_text: pd.DataFrame) -> pd.Series:
        """
        Prepares the text for summarization by formatting the headline and text.

        :param df_text: DataFrame containing headlines and texts to summarize.
        :return: A Series containing formatted texts ready for summarization.
        """
        return (
            '--- Report Start ---\n'
            ' Headline: ' + df_text.headline.astype(str) + 
            '\n Text: ' + df_text.text.astype(str) +
            '\n --- Report End ---'
        )

    def generate_summary(self, list_text_to_summarize: List[str], model: str,
                         n_reports: int, api_key: str, main_theme: str, topic: str) -> str:
        """
        Generates a single summary from a list of text chunks.

        :param list_text_to_summarize: List of text chunks to summarize.
        :param model: The model to use for summarization.
        :param n_reports: The number of reports being summarized.
        :param api_key: API key to access the summarization service.
        :param main_theme: The main theme for summary context.
        :param topic: The specific topic for which the summary is generated.
        :return: The concatenated summary of the provided text chunks.
        """
        if len(list_text_to_summarize) > 1:
            list_summary = []
            for this_text in list_text_to_summarize:
                this_summary = topic_summary(this_text, model, n_reports, api_key, main_theme, topic)
                list_summary.append(
                    '--- Report Start ---\n'
                    ' Headline: ' + '' +
                    '\n Text: ' + this_summary +
                    '\n --- Report End ---'
                )
            return "\n\n".join(list_summary)
        else:
            return list_text_to_summarize[0]


## Company-Specific Issues

class TopicSummarizerCompany:
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

    async def async_topic_summary(self, text: str, n: int, topic: str, entity_name: str) -> str:
        """
        Asynchronously produce a topic summary.

        Args:
            text (str): The text input.
            n (int): The number of documents/reports.
            topic (str): The topic label.
            entity_name (str): The name of the company/entity.

        Returns:
            str: The summary text.
        """
        loop = asyncio.get_event_loop()
        self.logger.debug("Running async_topic_summary for entity '%s' and topic '%s'", entity_name, topic)
        return await loop.run_in_executor(
            None, topic_summary_by_company, text, self.model, n, self.api_key, topic, entity_name
        )

    async def async_topic_risk(self, text: str, n: int, topic: str, entity_name: str) -> dict:
        """
        Asynchronously assess the risk for a topic.

        Args:
            text (str): The text input.
            n (int): The number of documents/reports.
            topic (str): The topic label.
            entity_name (str): The name of the company/entity.

        Returns:
            dict: A dictionary containing risk summary and magnitude.
        """
        loop = asyncio.get_event_loop()
        self.logger.debug("Running async_topic_risk for entity '%s' and topic '%s'", entity_name, topic)
        return await loop.run_in_executor(
            None, topic_risk_by_company, text, self.model, n, self.api_key, topic, entity_name
        )

    async def async_topic_uncertainty(self, text: str, n: int, topic: str, entity_name: str) -> dict:
        """
        Asynchronously assess the uncertainty for a topic.

        Args:
            text (str): The text input.
            n (int): The number of documents/reports.
            topic (str): The topic label.
            entity_name (str): The name of the company/entity.

        Returns:
            dict: A dictionary containing the uncertainty explanation.
        """
        loop = asyncio.get_event_loop()
        self.logger.debug("Running async_topic_uncertainty for entity '%s' and topic '%s'", entity_name, topic)
        return await loop.run_in_executor(
            None, topic_uncertainty_by_company, text, self.model, n, self.api_key, topic, entity_name
        )

    async def run_summaries(self, text: str, n_reports: int, topic: str, entity_name: str) -> Tuple[str, str, str]:
        """
        Run the summary, risk, and uncertainty tasks in parallel on one text chunk.

        Args:
            text (str): The text chunk.
            n_reports (int): The number of reports.
            topic (str): The topic label.
            entity_name (str): The entity name.

        Returns:
            Tuple[str, str, str]: Formatted report strings for topic, risk, and uncertainty.
        """
        self.logger.debug("Running summaries for entity '%s' and topic '%s'", entity_name, topic)
        topic_summary = await self.async_topic_summary(text, n_reports, topic, entity_name)
        risk_response = await self.async_topic_risk(text, n_reports, topic, entity_name)
        uncertainty_response = await self.async_topic_uncertainty(text, n_reports, topic, entity_name)

        formatted_topic = (
            f"--- Report Start ---\n\n Headline: \n  Text: {topic_summary}\n --- Report End ---"
        )
        formatted_risk = (
            f"--- Report Start ---\n\n Headline: \n  Text: {risk_response.get('summary', '')}\n --- Report End ---"
        )
        formatted_uncertainty = (
            f"--- Report Start ---\n\n Headline: \n  Text: {uncertainty_response.get('explanation', '')}\n --- Report End ---"
        )

        return formatted_topic, formatted_risk, formatted_uncertainty

    async def process_entity_topic(self, entity: Any, topic: str, df_topics_companies: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process a single entity for a single topic by generating summaries and packaging them in a DataFrame.

        Args:
            entity (Any): An object with attributes id and name.
            topic (str): The topic label.
            df_topics_companies (pd.DataFrame): DataFrame containing topic/company info.

        Returns:
            Optional[pd.DataFrame]: A single-row DataFrame with summary results or None if no documents found.
        """
        self.logger.debug("Processing topic '%s' for entity '%s'", topic, entity.name)
        df_to_summarize = df_topics_companies.loc[
            (df_topics_companies.entity_id == entity.id) & (df_topics_companies.label_by_company == topic)
        ]
        n_documents = df_to_summarize.document_id.nunique()

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
            topic_chunks, risk_chunks, uncertainty_chunks = zip(*results)
            text_to_summarize_topic = "\n\n".join(topic_chunks)
            text_to_summarize_risk = "\n\n".join(risk_chunks)
            text_to_summarize_uncertainty = "\n\n".join(uncertainty_chunks)
        else:
            text_to_summarize_topic = text_to_summarize_full
            text_to_summarize_risk = text_to_summarize_full
            text_to_summarize_uncertainty = text_to_summarize_full

        # Generate summaries; use try/except to log any errors.
        try:
            summary = await self.async_topic_summary(text_to_summarize_topic, n_reports, topic, entity.name)
        except Exception as e:
            self.logger.error("Error during topic summary for entity '%s' and topic '%s': %s", entity.name, topic, e)
            summary = ""

        try:
            risk_response = await self.async_topic_risk(text_to_summarize_risk, n_reports, topic, entity.name)
            risk_magnitude = risk_response.get('magnitude', '')
            risk_summary = risk_response.get('summary', '').replace('Target Company', entity.name)
        except Exception as e:
            self.logger.error("Error during risk summary for entity '%s' and topic '%s': %s", entity.name, topic, e)
            risk_magnitude = ""
            risk_summary = ""

        try:
            uncertainty_response = await self.async_topic_uncertainty(text_to_summarize_uncertainty, n_reports, topic, entity.name)
            uncertainty = uncertainty_response.get('uncertainty', '')
            uncertainty_explanation = uncertainty_response.get('explanation', '').replace('Target Company', entity.name)
        except Exception as e:
            self.logger.error("Error during uncertainty summary for entity '%s' and topic '%s': %s", entity.name, topic, e)
            uncertainty = ""
            uncertainty_explanation = ""

        df_by_company = pd.DataFrame({
            'entity_id': [entity.id],
            'entity_name': [entity.name],
            'topic': [topic],
            'topic_summary': [summary],
            'risk_magnitude': [risk_magnitude],
            'risk_summary': [risk_summary],
            'uncertainty': [uncertainty],
            'uncertainty_explanation': [uncertainty_explanation],
            'n_documents': [n_documents]
        })

        return df_by_company

    async def process_topic_by_company(
        self,
        df_labeled: pd.DataFrame,
        list_entities: List[Any],
        import_from_path: Optional[str] = None,
        export_to_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process all entities and topics; optionally import/export results from/to a pickle file.

        Args:
            df_labeled (pd.DataFrame): The labeled DataFrame.
            list_entities (List[Any]): A list of entity objects.
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
        df_topics_companies['label_by_company'] = df_topics_companies.apply(
            lambda x: f"{x.theme} - {x.label}", axis=1
        )

        tasks = []
        for entity in list_entities:
            topics = list(df_topics_companies.loc[df_topics_companies.entity_id == entity.id].label_by_company.unique())
            for topic in topics:
                tasks.append(self.process_entity_topic(entity, topic, df_topics_companies))

        self.logger.info("Starting processing for %d tasks...", len(tasks))
        results = await asyncio.gather(*tasks)

        # Filter out any None responses.
        df_list = [df for df in results if df is not None]
        if not df_list:
            self.logger.warning("No data was processed; returning an empty DataFrame.")
            return pd.DataFrame()

        df_by_company = pd.concat(df_list, ignore_index=True)

        # Map risk and uncertainty levels to numeric scores.
        convert_risk_score_dict = {'High': 3, 'Medium': 2, 'Low': 1, 'Neutral': 0}
        convert_uncertainty_score_dict = {'High': 3, 'Medium': 2, 'Low': 1, 'Past': 0}

        df_by_company['risk_score'] = df_by_company['risk_magnitude'].apply(lambda x: convert_risk_score_dict.get(x, 0))
        df_by_company['uncertainty_score'] = df_by_company['uncertainty'].apply(lambda x: convert_uncertainty_score_dict.get(x, 0))

        if export_to_path:
            self.logger.info("Exporting processed data to %s", export_to_path)
            df_by_company.to_pickle(export_to_path)

        return df_by_company
        