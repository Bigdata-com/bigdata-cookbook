"""Content retrieval using REST API and smart-batching (no SDK dependencies).

MIGRATION NOTE: Rewritten to use run_universe_search from search_helper.
Takes company_ids + id_to_name instead of SDK client + entity objects.
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional

from src.search_helper import run_universe_search
from src.mindmap.generate_trees import get_most_granular_elements


class DataRetriever:
    def __init__(
        self,
        company_ids: List[str],
        id_to_name: Dict[str, str],
        document_limit: int,
        sortby: str,
        search_freq: str,
        start_date_query: str,
        end_date_query: str,
    ):
        """
        Initialize the DataRetriever.

        :param company_ids: List of RP_ENTITY_ID values.
        :param id_to_name: Mapping from entity ID to company name.
        :param document_limit: The maximum number of documents to retrieve per query.
        :param sortby: The criteria by which results should be sorted (ignored for REST).
        :param search_freq: Frequency of searches (ignored for REST).
        :param start_date_query: The start date for the search query.
        :param end_date_query: The end date for the search query.
        """
        self.company_ids = company_ids
        self.id_to_name = id_to_name
        self.document_limit = document_limit
        self.sortby = sortby
        self.search_freq = search_freq
        self.start_date_query = start_date_query
        self.end_date_query = end_date_query

    def retrieve_by_sentences_entity_theme(
        self,
        entity_id: str,
        theme_sentences: List[str],
        document_type: str,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve sentences based on the entity and theme sentences using the specified document type.

        :param entity_id: The entity ID for which to retrieve sentences.
        :param theme_sentences: A list of sentences representing the theme.
        :param document_type: The type of document to retrieve ('news', 'filings', 'transcripts').
        :return: A DataFrame containing the retrieved sentences or None if no sentences are found.
        """
        try:
            df = run_universe_search(
                company_ids=[entity_id],
                queries=theme_sentences,
                start_date=self.start_date_query,
                end_date=self.end_date_query,
                scope=document_type,
                id_to_name=self.id_to_name,
            )
            if df.empty:
                return None
            return df
        except Exception as e:
            logging.error(f"Error in retrieving sentences for entity {entity_id}: {e}")
            return None

    def export_to_pickle(self, df_sentences: pd.DataFrame, export_to_path: Optional[str]) -> None:
        """
        Export the DataFrame to a pickle file.

        :param df_sentences: The DataFrame to export.
        :param export_to_path: The file path where the DataFrame will be saved.
        """
        if export_to_path:
            try:
                df_sentences.to_pickle(export_to_path)
            except Exception as e:
                logging.error(f"Error exporting DataFrame to pickle: {e}")

    def retrieve(
        self,
        themes_tree_dict: dict,
        list_specific_themes: List[str],
        document_type: str,
        import_from_path: Optional[str] = None,
        export_to_path: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve sentences for specified entities and themes.

        :param themes_tree_dict: A dictionary containing theme trees.
        :param list_specific_themes: A list of specific themes for the search.
        :param document_type: The type of document to retrieve ('news', 'filings', 'transcripts').
        :param import_from_path: Optional path to import DataFrame from a pickle file.
        :param export_to_path: Optional path to export the DataFrame to a pickle file.
        :return: A concatenated DataFrame of retrieved sentences or None if no sentences were found.
        """
        if import_from_path and os.path.isfile(import_from_path):
            logging.info("Importing DataFrame from pickle file.")
            return pd.read_pickle(import_from_path)
        
        list_df_sentences = []
        for spec_theme in list_specific_themes:
            for entity_id in self.company_ids:
                list_sentences = get_most_granular_elements(themes_tree_dict[spec_theme], 'Summary')

                df_sentences = self.retrieve_by_sentences_entity_theme(
                    entity_id, list_sentences, document_type
                )
                if df_sentences is not None:
                    df_sentences['theme'] = spec_theme
                    df_sentences['entity_searched_id'] = entity_id
                    df_sentences['entity_searched_name'] = self.id_to_name.get(entity_id, entity_id)
                    list_df_sentences.append(df_sentences)

        if list_df_sentences:
            df_sentences = pd.concat(list_df_sentences, ignore_index=True)
            self.export_to_pickle(df_sentences, export_to_path)
            return df_sentences
        else:
            logging.warning("No DataFrames were retrieved.")
            return None

    def retrieve_company_response(
        self,
        df_by_company: pd.DataFrame,
        list_specific_themes: List[str],
        document_type: str,
        import_from_path: Optional[str] = None,
        export_to_path: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve company responses based on the provided DataFrame and themes.

        :param df_by_company: DataFrame containing company response questions.
        :param list_specific_themes: A list of specific themes for the search.
        :param document_type: The type of document to retrieve ('news', 'filings', 'transcripts').
        :param import_from_path: Optional path to import DataFrame from a pickle file.
        :param export_to_path: Optional path to export the DataFrame to a pickle file.
        :return: A concatenated DataFrame of retrieved company responses or None if no responses were found.
        """
        if import_from_path and os.path.isfile(import_from_path):
            logging.info("Importing DataFrame for company responses from pickle file.")
            return pd.read_pickle(import_from_path)

        list_df_sentences = []
        for entity_id in self.company_ids:
            for theme in list_specific_themes:
                try:
                    unique_questions = df_by_company.loc[
                        (df_by_company.entity_id == entity_id) & (df_by_company.theme == theme),
                        'company_response_question'
                    ].unique()
                    list_sentences = list(unique_questions)
                    df_sentences = self.retrieve_by_sentences_entity_theme(
                        entity_id, list_sentences, document_type
                    )
                    if df_sentences is not None:
                        df_sentences['entity_searched_id'] = entity_id
                        df_sentences['entity_searched_name'] = self.id_to_name.get(entity_id, entity_id)
                        df_sentences['theme'] = theme
                        list_df_sentences.append(df_sentences)
                except Exception as e:
                    logging.error(f"Error retrieving data for entity {entity_id} and theme {theme}: {e}")

        if list_df_sentences:
            df_sentences = pd.concat(list_df_sentences, ignore_index=True)
            self.export_to_pickle(df_sentences, export_to_path)
            return df_sentences

        logging.warning("No DataFrames retrieved for company responses.")
        return None
