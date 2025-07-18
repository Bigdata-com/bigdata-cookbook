from logging import Logger, getLogger
from typing import Dict, List, Optional

import os
import logging
import pandas as pd
from pandas import merge
import pickle
import asyncio

from bigdata_client.models.search import DocumentType, SortBy
from bigdata_client import Bigdata

from bigdata_research_tools.search.screener_search import search_by_companies
from bigdata_research_tools.labeler.screener_labeler import ScreenerLabeler
from bigdata_research_tools.themes import generate_theme_tree, stringify_label_summaries

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
        watchlist_id: str,
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
        bigdata: Bigdata
    ):       
        """
        Initialize the GenerateReport class.

        :param watchlist_id: Identifier for the watchlist.
        :param general_focus: General focus for the mind map.
        :param specific_themes: List of specific themes.
        :param llm_model: LLM model identifier to be used for processing.
        :param api_key: LLM API key for external services.
        :param start_date: Start date for document search (YYYY-MM-DD).
        :param end_date: End date for document search (YYYY-MM-DD).
        :param fiscal_year: Fiscal year for document and transcript search.
        :param search_frequency: Frequency of searches (e.g., daily, weekly).
        :param document_limit_news: Maximum number of news documents to retrieve.
        :param document_limit_filings: Maximum number of filings/transcripts to retrieve.
        :param bigdata: Bigdata client instance.
        """
        self.logger = GenerateReport.logger
        self.watchlist_id = watchlist_id
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
        self.bigdata = bigdata
        # self.df_labeled = None 


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

    
    def generate_report(self, import_from_path: Optional[str] = None, export_to_path: Optional[str] = None) -> Report:
        """
        Generate the final report.

        This function coordinates the entire process:
          1. Retrieve watchlist and entities.
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

        # Fetch the watchlist and set up the entity mapping
        watchlist = self.bigdata.watchlists.get(self.watchlist_id)
        self.list_entities = self.bigdata.knowledge_graph.get_entities(watchlist.items) 
        self.logger.info("list_entities: %d entities", len(self.list_entities))
        

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
                    focus=focus
                )
                themes_tree_dict[focus] = theme_tree
            self.themes_tree_dict = themes_tree_dict
            # Export to Pickle if path provided
            if export_to_path:
                with open(export_to_path+'/themes_tree_dict', 'wb') as handle:
                    pickle.dump(themes_tree_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    


        ### Step 2: Searches
        
        # Run a (hybrid semantic) search on News via BigData API with our parameters
        # Attempt to import df_sentences_news DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/df_sentences_news'):
            self.logger.info("Importing df_sentences_news DataFrame from pickle file.")
            df_sentences_news = pd.read_pickle(import_from_path+'/df_sentences_news')
            self.logger.info("df_sentences_news: %d rows", len(df_sentences_news))
        else:
            df_sentences_news = []
            for focus in self.list_specific_focus:
                df_sentences = search_by_companies(
                    companies=self.list_entities,
                    sentences=list(self.themes_tree_dict[focus].get_terminal_label_summaries().values()),
                    fiscal_year=None,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    scope=DocumentType.NEWS, 
                    freq=self.search_frequency,
                    document_limit=self.document_limit_news,
                    batch_size=self.batch_size
                )
                df_sentences['theme'] = self.general_theme + ' in ' + focus
                df_sentences_news.append(df_sentences)
            df_sentences_news = pd.concat(df_sentences_news)
            self.logger.info("df_sentences_news: %d rows", len(df_sentences_news))
            # Export to Pickle if path provided
            if export_to_path:
                df_sentences_news.to_pickle(export_to_path+'/df_sentences_news')

        # Run search on Filings
        # Attempt to import df_sentences_filings DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/df_sentences_filings'):
            self.logger.info("Importing df_sentences_filings DataFrame from pickle file.")
            df_sentences_filings = pd.read_pickle(import_from_path+'/df_sentences_filings')
            self.logger.info("df_sentences_filings: %d rows", len(df_sentences_filings))
        else:
            df_sentences_filings = []
            for focus in self.list_specific_focus:
                df_sentences = search_by_companies(
                    companies=self.list_entities,
                    sentences=list(self.themes_tree_dict[focus].get_terminal_label_summaries().values()),
                    start_date=self.start_date,
                    end_date=self.end_date,
                    fiscal_year=self.fiscal_year,
                    scope=DocumentType.FILINGS, 
                    freq=self.search_frequency,
                    document_limit=self.document_limit_filings,
                    batch_size=self.batch_size
                )
                df_sentences['theme'] = self.general_theme + ' in ' + focus
                df_sentences_filings.append(df_sentences)
            df_sentences_filings = pd.concat(df_sentences_filings)
            self.logger.info("df_sentences_filings: %d rows", len(df_sentences_filings))
            # Export to Pickle if path provided
            if export_to_path:
                df_sentences_filings.to_pickle(export_to_path+'/df_sentences_filings')

        # Run search on Transcripts
        # Attempt to import df_sentences_filings DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/df_sentences_transcripts'):
            self.logger.info("Importing df_sentences_transcripts DataFrame from pickle file.")
            df_sentences_transcripts = pd.read_pickle(import_from_path+'/df_sentences_transcripts')
            self.logger.info("df_sentences_transcripts: %d rows", len(df_sentences_transcripts))
        else:
            df_sentences_transcripts = []
            for focus in self.list_specific_focus:
                df_sentences = search_by_companies(
                    companies=self.list_entities,
                    sentences=list(self.themes_tree_dict[focus].get_terminal_label_summaries().values()),
                    start_date=self.start_date,
                    end_date=self.end_date,
                    fiscal_year=self.fiscal_year,
                    scope=DocumentType.TRANSCRIPTS, 
                    freq=self.search_frequency,
                    document_limit=self.document_limit_transcripts,
                    batch_size=self.batch_size 
                )
                df_sentences['theme'] = self.general_theme + ' in ' + focus
                df_sentences_transcripts.append(df_sentences)
            df_sentences_transcripts = pd.concat(df_sentences_transcripts)
            self.logger.info("df_sentences_transcripts: %d rows", len(df_sentences_transcripts))
            # Export to Pickle if path provided
            if export_to_path:
                df_sentences_transcripts.to_pickle(export_to_path+'/df_sentences_transcripts')



        ### Step 3: Labeling

        # Label the search results with our theme labels
        labeler = ScreenerLabeler(llm_model=self.llm_model)

        # News
        # Attempt to import df_news_labeled DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/df_news_labeled'):
            self.logger.info("Importing df_news_labeled DataFrame from pickle file.")
            df_news_labeled = pd.read_pickle(import_from_path+'/df_news_labeled')
            self.logger.info("df_news_labeled: %d rows", len(df_news_labeled))
        else:
            df_news_labeled = []
            for focus in self.list_specific_focus:
                df_sentences_news_theme = df_sentences_news.loc[(df_sentences_news.theme == self.general_theme + ' in ' + focus)]
                df_sentences_news_theme.reset_index(drop=True, inplace=True)
                df_labels = labeler.get_labels(
                    main_theme=self.general_theme + ' in ' + focus, 
                    labels=list(self.themes_tree_dict[focus].get_terminal_label_summaries().keys()), 
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
        else:
            df_filings_labeled = []
            for focus in self.list_specific_focus:
                df_sentences_filings_theme = df_sentences_filings.loc[(df_sentences_filings.theme == self.general_theme + ' in ' + focus)]
                df_sentences_filings_theme.reset_index(drop=True, inplace=True)
                df_labels = labeler.get_labels(
                    main_theme=self.general_theme + ' in ' + focus, 
                    labels=list(self.themes_tree_dict[focus].get_terminal_label_summaries().keys()),
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
        else:
            df_transcripts_labeled = []
            for focus in self.list_specific_focus:
                df_sentences_transcripts_theme = df_sentences_transcripts.loc[(df_sentences_transcripts.theme == self.general_theme + ' in ' + focus)]
                df_sentences_transcripts_theme.reset_index(drop=True, inplace=True)
                df_labels = labeler.get_labels(
                    main_theme=self.general_theme + ' in ' + focus, 
                    labels=list(self.themes_tree_dict[focus].get_terminal_label_summaries().keys()), # to adapt
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
            model=self.llm_model.split('::')[1], 
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
            model=self.llm_model.split('::')[1], 
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
        df_filings_labeled['doc_type'] = 'Filings'
        df_transcripts_labeled['doc_type'] = 'Transcripts'
        df_ft_labeled = pd.concat([df_filings_labeled, df_transcripts_labeled])
        df_ft_labeled = df_ft_labeled.reset_index(drop=True)

        # Run the process to extract company's mitigation plan from the documents (filings and transcripts)
        response_processor = CompanyResponseProcessor(model=self.llm_model.split('::')[1], api_key=self.api_key, logger=self.logger, verbose=True)

        nb_labeled_chunks = len(df_ft_labeled.loc[~df_ft_labeled.label.isin(['', 'unassigned', 'unclear'])])
        self.logger.info("df_ft_labeled: %d kept rows", nb_labeled_chunks)


        if import_from_path:
            path_import_response = os.path.join(import_from_path, "df_response_by_company.pkl")
        else:
            path_import_response = None

        if export_to_path:
            path_export_response = os.path.join(export_to_path, "df_response_by_company.pkl")
        else:
            path_export_response = None

        df_response_by_company = asyncio.run(
            response_processor.process_response_by_company(
                df_labeled=df_ft_labeled, 
                df_by_company=df_by_company, 
                list_entities=self.list_entities,
                import_from_path=path_import_response,
                export_to_path=path_export_response
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
            watchlist_name=watchlist.name,
            themes_tree_dict=self.themes_tree_dict,
            report_by_theme=df_by_theme,
            report_by_company=df_by_company_with_responses
        )
            
        return report
