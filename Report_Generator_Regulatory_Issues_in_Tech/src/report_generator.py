from logging import Logger, getLogger
from typing import Dict, List, Optional

import os
import logging
import pandas as pd
from pandas import merge
import pickle
import asyncio

from src.bigdata_rest import BigdataRestClient, load_universe
from src.labeling import SimpleLabeler
from src.mindmap_generator import generate_theme_tree
from src.openai_utils import resolve_model
from src.search_helper import run_universe_search

from src.summary.summary import TopicSummarizerSector, TopicSummarizerCompany
from src.response.company_response import CompanyResponseProcessor


class Report:
    """A simple container for the generated report data."""
    def __init__(
        self, 
        watchlist_name: str,
        themes_tree_dict: Dict,
        report_by_theme: pd.DataFrame,
        report_by_company: pd.DataFrame
    ):
        """
        Initialize a Report.

        :param watchlist_name: Name of the watchlist.
        :param themes_tree_dict: Dictionary of the generated themes tree.
        :param report_by_theme: DataFrame with sector-level summaries.
        :param report_by_company: DataFrame with company-level summaries (including mitigation plans).
        """
        self.watchlist_name = watchlist_name
        self.themes_tree_dict = themes_tree_dict
        self.report_by_theme = report_by_theme
        self.report_by_company = report_by_company


class GenerateReport:
    """
    Generate a final report by coordinating document retrieval,
    labeling, summarization, and response extraction.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Configure a class-level logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    def __init__(
        self, 
        universe_df: pd.DataFrame,
        general_theme: str,
        list_specific_focus: List[str],
        llm_model: str,
        api_key: str,
        start_date: str,
        end_date: str,
        fiscal_year: int,
        search_frequency: str,
        document_limit_news: int,
        document_limit_filings: int,
        document_limit_transcripts: int,
        batch_size: int,
        chunk_percentage: float = 0.05,
    ):
        """
        Initialize the GenerateReport class.

        :param universe_df: DataFrame with RP_ENTITY_ID and COMPANY_NAME columns.
        :param general_theme: General theme for the mind map.
        :param list_specific_focus: List of specific focus areas.
        :param llm_model: LLM model identifier (e.g., 'gpt-4o-mini').
        :param api_key: OpenAI API key.
        :param start_date: Start date for document search (YYYY-MM-DD).
        :param end_date: End date for document search (YYYY-MM-DD).
        :param fiscal_year: Fiscal year for document and transcript search.
        :param search_frequency: Frequency (ignored, kept for compatibility).
        :param document_limit_news: Maximum number of news documents to retrieve.
        :param document_limit_filings: Maximum number of filings to retrieve.
        :param document_limit_transcripts: Maximum number of transcripts to retrieve.
        :param batch_size: Batch size (ignored, kept for compatibility).
        :param chunk_percentage: Fraction of matching chunks to sample per smart-batching
            search call (cost control knob passed through to ``run_universe_search``).
        """
        self.logger = GenerateReport.logger
        self.universe_df = universe_df
        self.general_theme = general_theme 
        self.list_specific_focus = list_specific_focus 
        self.llm_model = llm_model
        self.api_key = api_key
        self.start_date = start_date
        self.end_date = end_date 
        self.fiscal_year = fiscal_year
        self.search_frequency = search_frequency
        self.document_limit_news = document_limit_news
        self.document_limit_filings = document_limit_filings
        self.document_limit_transcripts = document_limit_transcripts
        self.batch_size = batch_size
        self.chunk_percentage = chunk_percentage
        self.rest_client = BigdataRestClient()
        self.openai_model = resolve_model(llm_model)


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

    @staticmethod
    def extract_theme_summaries(theme_tree_or_dict):
        """Extract terminal label summaries from theme tree (dict or object)."""
        # If it has get_terminal_label_summaries method, use it
        if hasattr(theme_tree_or_dict, 'get_terminal_label_summaries'):
            return list(theme_tree_or_dict.get_terminal_label_summaries().values())
        # Otherwise, assume it's a dict and walk it
        elif isinstance(theme_tree_or_dict, dict):
            return GenerateReport.get_most_granular_elements(theme_tree_or_dict, 'Summary')
        else:
            raise ValueError(f"Unknown theme tree type: {type(theme_tree_or_dict)}")

    def _document_limit_for_scope(self, scope: str) -> int:
        if scope == "news":
            return self.document_limit_news
        if scope == "filings":
            return self.document_limit_filings
        if scope == "transcripts":
            return self.document_limit_transcripts
        return 0

    def _search_scope(
        self,
        scope: str,
        *,
        import_from_path: str | None,
        export_to_path: str | None,
        pickle_name: str,
    ) -> pd.DataFrame:
        """Run smart-batching search for one document scope, or return empty frame."""
        if import_from_path and os.path.isfile(import_from_path + pickle_name):
            self.logger.info("Importing %s from pickle file.", pickle_name)
            df = pd.read_pickle(import_from_path + pickle_name)
            self.logger.info("%s: %d rows", pickle_name, len(df))
            return df

        if self._document_limit_for_scope(scope) <= 0:
            self.logger.info("Skipping %s search (document_limit=0)", scope)
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for focus in self.list_specific_focus:
            theme_summaries = self.extract_theme_summaries(self.themes_tree_dict[focus])
            df_sentences = run_universe_search(
                company_ids=self.company_ids,
                queries=theme_summaries,
                start_date=self.start_date,
                end_date=self.end_date,
                scope=scope,
                chunk_percentage=self.chunk_percentage,
                id_to_name=self.id_to_name,
            )
            df_sentences["theme"] = self.general_theme + " in " + focus
            df_sentences["rp_entity_id"] = None
            df_sentences["rp_document_id"] = None
            df_sentences["sentence_id"] = df_sentences["document_id"] + "-" + df_sentences.index.astype(str)
            df_sentences["timestamp_utc"] = df_sentences["timestamp"]
            df_sentences["entity_sector"] = None
            df_sentences["entity_industry"] = None
            df_sentences["entity_country"] = None
            df_sentences["entity_ticker"] = None
            df_sentences["other_entities"] = None
            df_sentences["entities"] = None
            frames.append(df_sentences)

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames)
        self.logger.info("%s: %d rows", pickle_name, len(result))
        if export_to_path:
            result.to_pickle(export_to_path + pickle_name)
        return result

    
    def generate_report(self, import_from_path: Optional[str] = None, export_to_path: Optional[str] = None) -> Report:
        """
        Generate the final report.

        This function coordinates the entire process:
          1. Load company universe (CSV / RP_ENTITY_ID list).
          2. Generate the themes tree.
          3. Retrieve news documents.
          4. Label news documents.
          5. Summarize at sector and company levels.
          6. Extract the company's mitigation plans.
          7. Build the final Report object.

        :param import_from_path: Optional directory to import cached data.
        :param export_to_path: Optional directory to export processed data.
        :return: A Report object with the consolidated results.
        """

        # Company universe from caller-supplied CSV / DataFrame (not platform watchlists)
        self.company_ids = self.universe_df["RP_ENTITY_ID"].astype(str).str.strip().tolist()
        self.id_to_name = dict(
            zip(
                self.universe_df["RP_ENTITY_ID"].astype(str).str.strip(),
                self.universe_df["COMPANY_NAME"].astype(str).str.strip(),
            )
        )
        # Build fake entity list for downstream summarizers (they expect objects with .id and .name)
        from types import SimpleNamespace
        self.list_entities = [
            SimpleNamespace(id=entity_id, name=name)
            for entity_id, name in self.id_to_name.items()
        ]
        self.logger.info("universe: %d companies", len(self.company_ids))


        ### Step 1: Mindmap

        # with open(import_from_path+'/themes_tree_dict', 'rb') as handle:
        #     self.themes_tree_dict = pickle.load(handle)

        # Generate the Theme Tree
        # Attempt to import the Theme Tree if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/themes_tree_dict'):
            self.logger.info("Importing themes_tree_dict from pickle file.")
            with open(import_from_path+'/themes_tree_dict', 'rb') as handle:
                self.themes_tree_dict = pickle.load(handle)
        else:
            themes_tree_dict = {}
            for focus in self.list_specific_focus:
                theme_tree = generate_theme_tree(
                    main_theme=self.general_theme,
                    focus=focus,
                    model=self.openai_model,
                )
                themes_tree_dict[focus] = theme_tree
            self.themes_tree_dict = themes_tree_dict
            # Export to Pickle if path provided
            if export_to_path:
                with open(export_to_path+'/themes_tree_dict', 'wb') as handle:
                    pickle.dump(themes_tree_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    


        ### Step 2: Searches

        df_sentences_news = self._search_scope(
            "news",
            import_from_path=import_from_path,
            export_to_path=export_to_path,
            pickle_name="/df_sentences_news",
        )
        df_sentences_filings = self._search_scope(
            "filings",
            import_from_path=import_from_path,
            export_to_path=export_to_path,
            pickle_name="/df_sentences_filings",
        )
        df_sentences_transcripts = self._search_scope(
            "transcripts",
            import_from_path=import_from_path,
            export_to_path=export_to_path,
            pickle_name="/df_sentences_transcripts",
        )

        ### Step 3: Labeling

        # Label the search results with our theme labels
        labeler = SimpleLabeler(model=self.openai_model, api_key=self.api_key)

        # News
        # Attempt to import df_news_labeled DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/df_news_labeled'):
            self.logger.info("Importing df_news_labeled DataFrame from pickle file.")
            df_news_labeled = pd.read_pickle(import_from_path+'/df_news_labeled')
            self.logger.info("df_news_labeled: %d rows", len(df_news_labeled))
        elif df_sentences_news.empty:
            df_news_labeled = pd.DataFrame()
        else:
            df_news_labeled = []
            for focus in self.list_specific_focus:
                df_sentences_news_theme = df_sentences_news.loc[(df_sentences_news.theme == self.general_theme + ' in ' + focus)]
                df_sentences_news_theme.reset_index(drop=True, inplace=True)
                # Extract theme labels from tree (handles both dict and object)
                theme_tree = self.themes_tree_dict[focus]
                if hasattr(theme_tree, 'get_terminal_label_summaries'):
                    theme_labels = list(theme_tree.get_terminal_label_summaries().keys())
                elif isinstance(theme_tree, dict):
                    theme_labels = self.get_most_granular_elements(theme_tree, 'Label')
                else:
                    theme_labels = []
                df_labels = labeler.get_labels(
                    main_theme=self.general_theme + ' in ' + focus, 
                    labels=theme_labels, 
                    texts=df_sentences_news_theme["masked_text"].tolist()        
                )
                df_news_labels = pd.merge(df_sentences_news_theme, df_labels, left_index=True, right_index=True)
                df_news_labeled.append(df_news_labels)
            df_news_labeled = pd.concat(df_news_labeled)
            # Export to Pickle if path provided
            if export_to_path:
                df_news_labeled.to_pickle(export_to_path+'/df_news_labeled')
        self.logger.info("df_news_labeled: %d rows", len(df_news_labeled))

        # Filings
        # Attempt to import df_filings_labeled DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/df_filings_labeled'):
            self.logger.info("Importing df_news_labeled DataFrame from pickle file.")
            df_filings_labeled = pd.read_pickle(import_from_path+'/df_filings_labeled')
            self.logger.info("df_filings_labeled: %d rows", len(df_filings_labeled))
        elif df_sentences_filings.empty:
            df_filings_labeled = pd.DataFrame()
        else:
            df_filings_labeled = []
            for focus in self.list_specific_focus:
                df_sentences_filings_theme = df_sentences_filings.loc[(df_sentences_filings.theme == self.general_theme + ' in ' + focus)]
                df_sentences_filings_theme.reset_index(drop=True, inplace=True)
                # Extract theme labels from tree (handles both dict and object)
                theme_tree = self.themes_tree_dict[focus]
                if hasattr(theme_tree, 'get_terminal_label_summaries'):
                    theme_labels = list(theme_tree.get_terminal_label_summaries().keys())
                elif isinstance(theme_tree, dict):
                    theme_labels = self.get_most_granular_elements(theme_tree, 'Label')
                else:
                    theme_labels = []
                df_labels = labeler.get_labels(
                    main_theme=self.general_theme + ' in ' + focus, 
                    labels=theme_labels,
                    texts=df_sentences_filings_theme["masked_text"].tolist()        
                )
                df_filings_labels = pd.merge(df_sentences_filings_theme, df_labels, left_index=True, right_index=True)
                df_filings_labeled.append(df_filings_labels)
            df_filings_labeled = pd.concat(df_filings_labeled)
            # Export to Pickle if path provided
            if export_to_path:
                df_filings_labeled.to_pickle(export_to_path+'/df_filings_labeled')
        self.logger.info("df_filings_labeled: %d rows", len(df_filings_labeled))

        
        # Transcripts
        # Attempt to import df_transcripts_labeled DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/df_transcripts_labeled'):
            self.logger.info("Importing df_transcripts_labeled DataFrame from pickle file.")
            df_transcripts_labeled = pd.read_pickle(import_from_path+'/df_transcripts_labeled')
            self.logger.info("df_transcripts_labeled: %d rows", len(df_transcripts_labeled))
        elif df_sentences_transcripts.empty:
            df_transcripts_labeled = pd.DataFrame()
        else:
            df_transcripts_labeled = []
            for focus in self.list_specific_focus:
                df_sentences_transcripts_theme = df_sentences_transcripts.loc[(df_sentences_transcripts.theme == self.general_theme + ' in ' + focus)]
                df_sentences_transcripts_theme.reset_index(drop=True, inplace=True)
                # Extract theme labels from tree (handles both dict and object)
                theme_tree = self.themes_tree_dict[focus]
                if hasattr(theme_tree, 'get_terminal_label_summaries'):
                    theme_labels = list(theme_tree.get_terminal_label_summaries().keys())
                elif isinstance(theme_tree, dict):
                    theme_labels = self.get_most_granular_elements(theme_tree, 'Label')
                else:
                    theme_labels = []
                df_labels = labeler.get_labels(
                    main_theme=self.general_theme + ' in ' + focus, 
                    labels=theme_labels, # to adapt
                    texts=df_sentences_transcripts_theme["masked_text"].tolist()        
                )
                df_transcripts_labels = pd.merge(df_sentences_transcripts_theme, df_labels, left_index=True, right_index=True)
                df_transcripts_labeled.append(df_transcripts_labels)
            df_transcripts_labeled = pd.concat(df_transcripts_labeled)
            # Export to Pickle if path provided
            if export_to_path:
                df_transcripts_labeled.to_pickle(export_to_path+'/df_transcripts_labeled')
        self.logger.info("df_transcripts_labeled: %d rows", len(df_transcripts_labeled))

        ### Step 4: Summarize at sector and company levels.

        # Run the process to summarize the documents and compute media attention by topic, sector-wide
        summarizer_sector = TopicSummarizerSector(
            model=self.openai_model,
            api_key=self.api_key, 
            df_labeled=df_news_labeled, 
            general_theme = self.general_theme,
            list_specific_focus = self.list_specific_focus,
            themes_tree_dict=self.themes_tree_dict,
            logger=self.logger
        )


        if import_from_path == None:
            path_import_by_theme = None
        else:
            path_import_by_theme = import_from_path+'/df_by_theme'

        if export_to_path == None:
            path_export_by_theme = None
        else:
            path_export_by_theme = export_to_path+'/df_by_theme'
        
        df_by_theme = summarizer_sector.summarize(import_from_path=path_import_by_theme, export_to_path=path_export_by_theme)
        self.logger.info("df_by_theme: %d rows", len(df_by_theme))

        # Run the process to summarize the documents and score media attention, risk and uncertainty by topic at company level.
        summarizer_company = TopicSummarizerCompany(
            model=self.openai_model,
            api_key=self.api_key,
            logger=self.logger, 
            verbose=True
        )

        if import_from_path == None:
            path_import_by_company = None
        else:
            path_import_by_company = import_from_path+'/df_by_company'

        if export_to_path == None:
            path_export_by_company = None
        else:
            path_export_by_company = export_to_path+'/df_by_company'



        df_by_company = asyncio.run(
            summarizer_company.process_topic_by_company(
                df_labeled=df_news_labeled, 
                list_entities=self.list_entities, 
                import_from_path=path_import_by_company, 
                export_to_path=path_export_by_company
            )
        )
        self.logger.info("df_by_company: %d rows", len(df_by_company))


        ### Step 5: Extract the company's mitigation plan.

        # Concatenate Filings and Transcripts dataframes
        ft_frames = []
        if not df_filings_labeled.empty:
            df_filings_labeled = df_filings_labeled.copy()
            df_filings_labeled["doc_type"] = "Filings"
            ft_frames.append(df_filings_labeled)
        if not df_transcripts_labeled.empty:
            df_transcripts_labeled = df_transcripts_labeled.copy()
            df_transcripts_labeled["doc_type"] = "Transcripts"
            ft_frames.append(df_transcripts_labeled)
        df_ft_labeled = pd.concat(ft_frames, ignore_index=True) if ft_frames else pd.DataFrame()

        # Run the process to extract company's mitigation plan from the documents (filings and transcripts)
        response_processor = CompanyResponseProcessor(
            model=self.openai_model, api_key=self.api_key, logger=self.logger, verbose=True
        )

        if df_ft_labeled.empty:
            nb_labeled_chunks = 0
        else:
            nb_labeled_chunks = len(
                df_ft_labeled.loc[~df_ft_labeled.label.isin(["", "unassigned", "unclear"])]
            )
        self.logger.info("df_ft_labeled: %d kept rows", nb_labeled_chunks)


        if import_from_path:
            path_import_response = os.path.join(import_from_path, "df_response_by_company.pkl")
        else:
            path_import_response = None

        if export_to_path:
            path_export_response = os.path.join(export_to_path, "df_response_by_company.pkl")
        else:
            path_export_response = None

        if df_ft_labeled.empty:
            df_response_by_company = pd.DataFrame(
                {
                    "entity_id": pd.Series(dtype="object"),
                    "entity_name": pd.Series(dtype="object"),
                    "topic": pd.Series(dtype="object"),
                    "response_summary": pd.Series(dtype="object"),
                    "n_response_documents": pd.Series(dtype="int64"),
                }
            )
        else:
            df_response_by_company = asyncio.run(
                response_processor.process_response_by_company(
                    df_labeled=df_ft_labeled,
                    df_by_company=df_by_company,
                    list_entities=self.list_entities,
                    import_from_path=path_import_response,
                    export_to_path=path_export_response,
                )
            )
        self.logger.info("df_response_by_company: %d rows", len(df_response_by_company))

        # Merge the companies responses to the dataframe with issue summaries and scores
        df_by_company_with_responses = pd.merge(df_by_company, df_response_by_company, on=['entity_id', 'entity_name', 'topic'], how='left')
        df_by_company_with_responses['filings_response_summary'] = df_by_company_with_responses['response_summary']

        # Extract the company's mitigation plan for each regulatory issue from the News documents if no relevant information was found in the Filings and Transcripts.
        df_news_response_by_company = asyncio.run(response_processor.process_response_by_company(
            df_labeled=df_news_labeled, 
            df_by_company=df_by_company, 
            list_entities=self.list_entities))


        df_news_response_by_company = df_news_response_by_company.rename(
            columns={'response_summary': 'news_response_summary', 'n_response_documents': 'news_n_response_documents'})
        df_by_company_with_responses = pd.merge(df_by_company_with_responses, df_news_response_by_company, 
                                                on=['entity_id', 'entity_name', 'topic'], how='left')
        df_by_company_with_responses['response_summary'] = df_by_company_with_responses['response_summary'].fillna(
            df_by_company_with_responses['news_response_summary'])


        self.logger.info("df_by_company_with_responses: %d rows", len(df_by_company_with_responses))
        # Export to Pickle if path provided
        if export_to_path:
            df_by_company_with_responses.to_pickle(export_to_path+'/df_by_company_with_responses')


        # Construct the Report
        report = Report(
            watchlist_name="Company Universe",
            themes_tree_dict=self.themes_tree_dict,
            report_by_theme=df_by_theme,
            report_by_company=df_by_company_with_responses
        )
            
        return report
