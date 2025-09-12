import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Any

from src.label.labels import process_request, deserialize_responses, get_prompts, get_system_prompt

class LabelProcessor:
    def __init__(self, list_entities: List[Any],
                 themes_tree_dict: Dict[str, Any],
                 list_specific_themes: List[str],
                 api_key: Optional[str] = None):
        """
        Initialize the LabelProcessor with entities, themes, and themes tree.

        :param list_entities: List of entities for whom labeling is performed.
        :param themes_tree_dict: Dictionary containing themes and their structures.
        :param list_specific_themes: List of specific themes to process.
        :param api_key: OpenAI API key for making requests.
        """
        self.list_entities = list_entities
        self.themes_tree_dict = themes_tree_dict
        self.list_specific_themes = list_specific_themes
        self.api_key = api_key

    def run_label_process(self, df_sentences: pd.DataFrame,
                          import_from_path: Optional[str] = None,
                          export_to_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Processes sentences, labels them based on themes and entities, 
        and optionally imports from or exports to a pickle file.

        :param df_sentences: DataFrame containing sentences to be labeled.
        :param import_from_path: Optional path to import a DataFrame from a pickle file.
        :param export_to_path: Optional path to export the labeled DataFrame to a pickle file.
        :return: A DataFrame containing the labeled sentences or None if no sentences were processed.
        """
        # Attempt to import labeled DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path):
            logging.info("Importing labeled DataFrame from pickle file.")
            return pd.read_pickle(import_from_path)

        # List to store labeled DataFrames
        list_df_labeled = []

        for spec_theme in self.list_specific_themes:
            theme_tree = self.themes_tree_dict.get(spec_theme)
            system_prompt = get_system_prompt(theme_tree, spec_theme)

            for entity in self.list_entities:
                # Filter sentences for the specific theme and entity
                df_sentences_theme_entity = df_sentences.loc[
                    (df_sentences.theme == spec_theme) &
                    (df_sentences.entity_searched_id == entity.id)
                ].copy()

                df_sentences_theme_entity.reset_index(drop=True, inplace=True)

                # Prepare prompts for processing
                prompts = get_prompts(df_sentences_theme_entity)

                try:
                    # Process requests and deserialize responses
                    responses = process_request(prompts, system_prompt, self.api_key)
                    df_labeled = deserialize_responses(responses)

                    # Merge the labeled results with the original sentences
                    df_labeled = pd.merge(
                        df_sentences_theme_entity,
                        df_labeled,
                        left_index=True,
                        right_index=True
                    )

                    # Append the labeled DataFrame to the list
                    list_df_labeled.append(df_labeled)
                except Exception as e:
                    logging.error(f"Error processing entity {entity.id} for theme {spec_theme}: {e}")

        # Concatenate all labeled DataFrames
        if list_df_labeled:
            df_labeled = pd.concat(list_df_labeled, ignore_index=True)

            # Export the labeled DataFrame to a pickle file if path provided
            if export_to_path:
                try:
                    df_labeled.to_pickle(export_to_path)
                    logging.info("Exported labeled DataFrame to pickle file.")
                except Exception as e:
                    logging.error(f"Error exporting labeled DataFrame to pickle: {e}")

            return df_labeled

        logging.warning("No sentences were processed; returning None.")
        return None
