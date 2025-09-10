
import os
import pandas as pd
import logging
from typing import Dict, List, Optional


from src.search.query_tools import build_batched_query, create_date_ranges
from src.search.search import run_search
from src.search.sentences import postprocess_search_results, build_dataframe_entity
from src.mindmap.generate_trees import get_most_granular_elements

### Content Retrieval

class DataRetriever:
    def __init__(self, bigdata: object, document_limit: int, sortby: str,
                 search_freq: str, start_date_query: str, end_date_query: str):
        """
        Initialize the DataRetriever.

        :param bigdata: Bigdata client instance.
        :param document_limit: The maximum number of documents to retrieve per query.
        :param sortby: The criteria by which results should be sorted.
        :param search_freq: Frequency of searches (e.g., daily, weekly).
        :param start_date_query: The start date for the search query.
        :param end_date_query: The end date for the search query.
        """
        self.bigdata = bigdata
        self.document_limit = document_limit
        self.sortby = sortby
        self.search_freq = search_freq
        self.start_date_query = start_date_query
        self.end_date_query = end_date_query

    def retrieve_by_sentences_entity_theme(self, entity: object, theme_sentences: List[str], 
                                             document_type: str) -> Optional[pd.DataFrame]:
        """
        Retrieve sentences based on the entity and theme sentences using the specified document type.

        :param entity: The entity for which to retrieve sentences.
        :param theme_sentences: A list of sentences representing the theme.
        :param document_type: The type of document to retrieve (e.g., 'news', 'reports').
        :return: A DataFrame containing the retrieved sentences or None if no sentences are found.
        """
        try:
            queries_sim = build_batched_query(
                sentences=theme_sentences,
                entity_keys=[entity.id],
                keywords=None,
                control_entities=None,
                batch_size=100,
                scope=document_type  
            )
            print("len(queries_sim)", len(queries_sim))
            print("queries_sim", queries_sim)
            date_ranges = create_date_ranges(self.start_date_query, self.end_date_query, self.search_freq)
            print("date_ranges", date_ranges)
            results_list_sim = run_search(
                bigdata=self.bigdata, 
                queries=queries_sim, 
                date_range=date_ranges,
                sortby=self.sortby,
                scope=document_type,
                limit=self.document_limit
            )
            
            results, entities = postprocess_search_results(results_list_sim)
            if results and entities:
                return build_dataframe_entity(results, entities, [entity])
            return None
        except Exception as e:
            logging.error(f"Error in retrieving sentences for entity {entity.id}: {e}")
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

    def retrieve(self, list_entities: List[object], themes_tree_dict: dict, 
                 list_specific_themes: List[str], document_type: str,
                 import_from_path: Optional[str] = None, 
                 export_to_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve sentences for specified entities and themes.

        :param list_entities: A list of entities to search.
        :param themes_tree_dict: A dictionary containing theme trees.
        :param list_specific_themes: A list of specific themes for the search.
        :param document_type: The type of document to retrieve (e.g., 'news', 'reports').
        :param import_from_path: Optional path to import DataFrame from a pickle file.
        :param export_to_path: Optional path to export the DataFrame to a pickle file.
        :return: A concatenated DataFrame of retrieved sentences or None if no sentences were found.
        """        
        if import_from_path and os.path.isfile(import_from_path):
            logging.info("Importing DataFrame from pickle file.")
            return pd.read_pickle(import_from_path)
        print("test")
        list_df_sentences = []
        for spec_theme in list_specific_themes:
            for entity in list_entities:
                list_sentences = get_most_granular_elements(themes_tree_dict[spec_theme], 'Summary')
                print("entity", entity)
                print("len(list_sentences)", len(list_sentences))
                print("list_sentences", list_sentences)

                df_sentences = self.retrieve_by_sentences_entity_theme(entity, list_sentences, document_type)
                if df_sentences is not None:
                    df_sentences['theme'] = spec_theme
                    df_sentences['entity_searched_id'] = entity.id
                    df_sentences['entity_searched_name'] = entity.name
                    list_df_sentences.append(df_sentences)

        if list_df_sentences:
            df_sentences = pd.concat(list_df_sentences, ignore_index=True)
            self.export_to_pickle(df_sentences, export_to_path)
            return df_sentences
        else:
            logging.warning("No DataFrames were retrieved.")
            return None

    def retrieve_company_response(self, df_by_company: pd.DataFrame, list_entities: List[object], 
                                  list_specific_themes: List[str], document_type: str,
                                  import_from_path: Optional[str] = None, 
                                  export_to_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve company responses based on the provided DataFrame and themes.

        :param df_by_company: DataFrame containing company response questions.
        :param list_entities: A list of entities to search responses for.
        :param list_specific_themes: A list of specific themes for the search.
        :param document_type: The type of document to retrieve (e.g., 'news', 'reports').
        :param import_from_path: Optional path to import DataFrame from a pickle file.
        :param export_to_path: Optional path to export the DataFrame to a pickle file.
        :return: A concatenated DataFrame of retrieved company responses or None if no responses were found.
        """        
        if import_from_path and os.path.isfile(import_from_path):
            logging.info("Importing DataFrame for company responses from pickle file.")
            return pd.read_pickle(import_from_path)

        list_df_sentences = []
        for entity in list_entities:
            for theme in list_specific_themes:
                try:
                    unique_questions = df_by_company.loc[
                        (df_by_company.entity_id == entity.id) & (df_by_company.theme == theme),
                        'company_response_question'
                    ].unique()
                    list_sentences = list(unique_questions)
                    df_sentences = self.retrieve_by_sentences_entity_theme(entity, list_sentences, document_type)
                    if df_sentences is not None:
                        df_sentences['entity_searched_id'] = entity.id
                        df_sentences['entity_searched_name'] = entity.name
                        df_sentences['theme'] = theme
                        list_df_sentences.append(df_sentences)
                except Exception as e:
                    logging.error(f"Error retrieving data for entity {entity.id} and theme {theme}: {e}")

        if list_df_sentences:
            df_sentences = pd.concat(list_df_sentences, ignore_index=True)
            self.export_to_pickle(df_sentences, export_to_path)
            return df_sentences

        logging.warning("No DataFrames retrieved for company responses.")
        return None
