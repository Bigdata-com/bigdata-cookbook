from logging import Logger, getLogger
from typing import Dict, List, Optional

import os
import pandas as pd
from pandas import merge
import pickle
import asyncio

from bigdata_client.models.search import DocumentType, SortBy
from bigdata_client import Bigdata

from src.mindmap.generate_trees import generate_themes_tree_dict
from src.search.content_retrieval import DataRetriever
from src.label.label_process import LabelProcessor
from src.summary.summary import TopicSummarizerSector, TopicSummarizerCompany
from src.response.company_response import CompanyResponseProcessor
from bigdata_research_tools.workflows.risk_analyzer import RiskAnalyzer

# Tracking functionality - defined before classes to avoid NameError
from importlib.metadata import version
from bigdata_client import tracking_services

_intialization_sent = False



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

def theme_tree_to_dict(theme_tree):
    # Create dictionary for current node
    node_dict = {
        'Node': theme_tree.node,
        'Label': theme_tree.label,
        'Summary': theme_tree.summary
    }
    
    # Process children if they exist
    if theme_tree.children:
        children_list = []
        for child in theme_tree.children:
            children_list.append(theme_tree_to_dict(child))
        node_dict['Children'] = children_list
    
    return node_dict



class GenerateReport:
    """
    Variant of the report generator that receives precomputed RiskAnalyzer outputs
    (risk_tree, risk_summaries, terminal_labels) and executes the remaining workflow:
    - retrieve and label News using the provided taxonomy
    - summarize sector/company
    - extract mitigation plans from Filings/Transcripts with News fallback
    """
    def __init__(
        self,
        watchlist_id: str,
        main_theme: str,
        focus: str,
        llm_model: str,
        api_key: str,
        start_date: str,
        end_date: str,
        search_frequency: str,
        document_limit_news: int,
        document_limit_filings: int,
        bigdata: Bigdata,
        batch_size: int,
        themes_tree_dict: Dict
    ):
        self.logger: Logger = getLogger(__name__)
        self.watchlist_id = watchlist_id
        self.main_theme = main_theme
        self.focus = focus
        
        # Initialize tracking automatically
        self.llm_model = llm_model
        self.api_key = api_key
        self.start_date = start_date
        self.end_date = end_date
        self.search_frequency = search_frequency
        self.document_limit_news = document_limit_news
        self.document_limit_filings = document_limit_filings
        self.bigdata = bigdata
        self.batch_size = batch_size
        self.themes_tree_dict = themes_tree_dict


    def generate_report(
        self,
        df_labeled: pd.DataFrame,
        import_from_path: Optional[str] = None,
        export_to_path: Optional[str] = None,
        news_search_fallback: bool = True
    ) -> Report:
        # Fetch the watchlist and entities
        watchlist = self.bigdata.watchlists.get(self.watchlist_id)
        self.list_entities = self.bigdata.knowledge_graph.get_entities(watchlist.items)

        # Use provided themes_tree_dict and labeled news

        # Sector summaries
        summarizer_sector = TopicSummarizerSector(
            model=self.llm_model,
            api_key=self.api_key,
            df_labeled=df_labeled,
            list_specific_themes=[self.main_theme],
            themes_tree_dict=self.themes_tree_dict
        )
        df_by_theme = summarizer_sector.summarize(
            import_from_path=import_from_path+'/df_by_theme' if import_from_path else None,
            export_to_path=export_to_path+'/df_by_theme' if export_to_path else None
        )

        # Company summaries and scores
        summarizer_company = TopicSummarizerCompany(
            model=self.llm_model,
            api_key=self.api_key,
            verbose=True
        )
        df_by_company = asyncio.run(
            summarizer_company.process_topic_by_company(
                df_labeled=df_labeled,
                list_entities=self.list_entities,
                import_from_path=import_from_path+'/df_by_company' if import_from_path else None,
                export_to_path=export_to_path+'/df_by_company' if export_to_path else None
            )
        )

        # Mitigation plans extraction
        df_by_company_with_responses = self.extract_mitigation_plan_v2(
            df_by_company=df_by_company,
            df_labeled=df_labeled,
            import_from_path=import_from_path,
            export_to_path=export_to_path,
            news_search_fallback=news_search_fallback
        )

        # Construct report
        report = Report(
            watchlist_name=watchlist.name,
            themes_tree_dict=self.themes_tree_dict,
            report_by_theme=df_by_theme,
            report_by_company=df_by_company_with_responses
        )

        return report

    def extract_mitigation_plan_v2(self, df_by_company: pd.DataFrame, df_labeled: pd.DataFrame, import_from_path: Optional[str] = None, export_to_path: Optional[str] = None, news_search_fallback: bool = True) -> List[pd.DataFrame]:

        """
        In this version I don't condense the company's issue but directly query with the sentence from the tree (like with the News)
        """
        
        # Import Pickle if path provided and file exists
        if import_from_path:
            if os.path.isfile(import_from_path+'/df_by_company_with_responses'):
                df_by_company_with_responses = pd.read_pickle(import_from_path+'/df_by_company_with_responses')
                return df_by_company_with_responses
        
        data_retriever = DataRetriever(
            bigdata = self.bigdata, 
            document_limit=self.document_limit_filings,
            sortby=SortBy.RELEVANCE, 
            search_freq=self.search_frequency,
            start_date_query=self.start_date, 
            end_date_query=self.end_date, 
        )

        df_sentences_filings = data_retriever.retrieve(
            list_entities=self.list_entities, 
            themes_tree_dict=self.themes_tree_dict, 
            list_specific_themes=[self.main_theme], 
            document_type=DocumentType.FILINGS,
            import_from_path=import_from_path+'/df_sentences_filings' if import_from_path else None,
            export_to_path=export_to_path+'/df_sentences_filings' if export_to_path else None
        )
        
        df_sentences_transcripts = data_retriever.retrieve(
            list_entities=self.list_entities, 
            themes_tree_dict=self.themes_tree_dict, 
            list_specific_themes=[self.main_theme], 
            document_type=DocumentType.TRANSCRIPTS,
            import_from_path=import_from_path+'/df_sentences_transcripts' if import_from_path else None,
            export_to_path=export_to_path+'/df_sentences_transcripts' if export_to_path else None
        )

        # Run the topic verification and identification for Filings and Transcripts
        label_processor = LabelProcessor(
            list_entities=self.list_entities, 
            themes_tree_dict=self.themes_tree_dict, 
            list_specific_themes=[self.main_theme] ,
        )
        
        df_filings_labeled = label_processor.run_label_process(
            df_sentences=df_sentences_filings, import_from_path=import_from_path+'/df_filings_labeled' if import_from_path else None, export_to_path=export_to_path+'/df_filings_labeled' if export_to_path else None)
        
        df_transcripts_labeled = label_processor.run_label_process(
            df_sentences=df_sentences_transcripts, import_from_path=import_from_path+'/df_transcripts_labeled' if import_from_path else None, export_to_path=export_to_path+'/df_transcripts_labeled' if export_to_path else None)        

        # Concatenate Filings and Transcripts dataframes
        df_filings_labeled['doc_type'] = 'Filings'
        df_transcripts_labeled['doc_type'] = 'Transcripts'
        df_ft_labeled = pd.concat([df_filings_labeled, df_transcripts_labeled])
        df_ft_labeled = df_ft_labeled.reset_index(drop=True)

        # Run the process to extract company's mitigation plan from the documents (filings and transcripts)
        response_processor = CompanyResponseProcessor(model=self.llm_model, api_key=self.api_key, verbose=True)
        
        df_response_by_company = asyncio.run(
            response_processor.process_response_by_company(
                df_labeled=df_ft_labeled, 
                df_by_company=df_by_company, 
                list_entities=self.list_entities,
                import_from_path=import_from_path+'/df_response_by_company' if import_from_path else None,
                export_to_path=export_to_path+'/df_response_by_company' if export_to_path else None
            )
        )

        # Merge the companies responses to the dataframe with issue summaries and scores
        df_by_company_with_responses = pd.merge(df_by_company, df_response_by_company, on=['entity_id', 'entity_name', 'topic'], how='left')
        df_by_company_with_responses['filings_response_summary'] = df_by_company_with_responses['response_summary']
        # Initialize origin flag for response summary
        df_by_company_with_responses['response_from_news'] = False

        # Extract the company's mitigation plan for each regulatory issue from the News documents if enabled
        if news_search_fallback:
            df_news_response_by_company = asyncio.run(
                response_processor.process_response_by_company(
                    df_labeled=df_labeled,
                    df_by_company=df_by_company,
                    list_entities=self.list_entities
                )
            )

            df_news_response_by_company = df_news_response_by_company.rename(
                columns={'response_summary': 'news_response_summary', 'n_response_documents': 'news_n_response_documents'}
            )
            df_by_company_with_responses = pd.merge(
                df_by_company_with_responses,
                df_news_response_by_company,
                on=['entity_id', 'entity_name', 'topic'],
                how='left'
            )
            # Mark rows where we will use News fallback
            fallback_mask = df_by_company_with_responses['response_summary'].isna() & df_by_company_with_responses['news_response_summary'].notna()
            # Apply fallback
            df_by_company_with_responses['response_summary'] = df_by_company_with_responses['response_summary'].fillna(
                df_by_company_with_responses['news_response_summary']
            )
            # Update origin flag
            df_by_company_with_responses.loc[fallback_mask, 'response_from_news'] = True
        
        # Export to Pickle if path provided
        if export_to_path:
            df_by_company_with_responses.to_pickle(export_to_path+'/df_by_company_with_responses')

        return df_by_company_with_responses

def notebook_initialized(bigdata):
    """Call this function after creating your bigdata client to enable usage tracking."""
    global _intialization_sent
    if not _intialization_sent:
            trace = tracking_services.TraceEvent(
                event_name="BigdataCookbookExecution", 
                properties={
                    "bigdataResearchToolsVersion": version("bigdata_research_tools"),
                    "bigdataClientVersion": version("bigdata-client"),
                    "cookbook_name": "ReportGeneratorSpecializedReportTariffs"                    
                }
            )
            tracking_services.send_trace(bigdata_client=bigdata, trace=trace)
            _intialization_sent = True


bigdata = Bigdata()
notebook_initialized(bigdata)
