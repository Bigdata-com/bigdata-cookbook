from logging import Logger, getLogger
from typing import Dict, List, Optional

import os
import logging
import pandas as pd
from pandas import merge
import pickle
import asyncio
import matplotlib.pyplot as plt

from bigdata_client.models.search import DocumentType, SortBy
from bigdata_client import Bigdata

from bigdata_research_tools.search.screener_search import search_by_companies
from bigdata_research_tools.labeler.screener_labeler import ScreenerLabeler

from src.summary.summary import SummarizerCompany


class Report:
    """A simple container for the generated report data."""
    def __init__(
        self, 
        watchlist_name: str,
        report_by_company: pd.DataFrame
    ):
        """
        Initialize a Report.

        :param watchlist_name: Name of the watchlist.
        :param report_by_company: DataFrame with company-level summaries (including mitigation plans).
        """
        self.watchlist_name = watchlist_name
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
        keywords: List[str],
        main_theme_risk: str,
        main_theme_proactivity: str,
        list_sentences_risks: List[str],
        list_sentences_proactivity: List[str],
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
        self.keywords = keywords
        self.main_theme_risk = main_theme_risk
        self.main_theme_proactivity = main_theme_proactivity
        self.list_sentences_risks = list_sentences_risks 
        self.list_sentences_proactivity = list_sentences_proactivity 
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
    def aggregate_verbatim(df, label):
        
        # Select only relevant chunks
        df_agg = df.loc[df.label==label].copy()

        # Aggregate rp_document_ids
        df_agg_id = df_agg.groupby('entity_name', as_index=False).agg(document_ids=('document_id', ' \n\n '.join))
        df_agg_id = df_agg_id.rename(columns={'document_ids': 'document_ids_'+label})

        # Aggregate quotes
        df_agg['text_to_agg'] = (
            '--- Quote Start ---\n\n'
            ' Headline: ' + df_agg.headline.astype(str) +
            '\n\n Text: ' + df_agg.text.astype(str) +
            '\n\n --- Quote End ---'
        )
        df_agg_quotes = df_agg.groupby('entity_name', as_index=False).agg(quotes=('text_to_agg', " \n\n ".join))
        df_agg_quotes = df_agg_quotes.rename(columns={'quotes': 'quotes_'+label})

        df_agg = pd.merge(df_agg_id, df_agg_quotes, how='left', on='entity_name')

        return df_agg


    def retrieve_documents(self, doc_type, doc_limit, doc_type_name, fiscal_year, import_from_path: Optional[str] = None, export_to_path: Optional[str] = None):
        if doc_limit>0:

            # Retrieve content for Risk

            if import_from_path and os.path.isfile(import_from_path+'/df_sentences_semantic_risk_'+doc_type_name):
                
                self.logger.info(f"Importing df_sentences_semantic_risk_{doc_type_name} DataFrame from pickle file.")
                df_sentences_semantic_risk = pd.read_pickle(import_from_path+'/df_sentences_semantic_risk_'+doc_type_name)
                self.logger.info(f"df_sentences_semantic_risk_{doc_type_name}: %d rows", len(df_sentences_semantic_risk))
            else:
                df_sentences_semantic_risk = search_by_companies(
                    companies=self.list_entities,
                    keywords=self.keywords,
                    sentences=self.list_sentences_risks,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    fiscal_year=fiscal_year,
                    scope=doc_type, 
                    freq=self.search_frequency,
                    document_limit=doc_limit,
                    batch_size=self.batch_size
                )
                self.logger.info(f"df_sentences_semantic_risk_{doc_type_name}: %d rows", len(df_sentences_semantic_risk))
                # Export to Pickle if path provided
                if export_to_path:
                    df_sentences_semantic_risk.to_pickle(export_to_path+'/df_sentences_semantic_risk_'+doc_type_name)

            # Retrieve content for Proactivity

            if import_from_path and os.path.isfile(import_from_path+'/df_sentences_semantic_proactivity_'+doc_type_name):
                self.logger.info(f"Importing df_sentences_semantic_proactivity_{doc_type_name} DataFrame from pickle file.")
                df_sentences_semantic_proactivity = pd.read_pickle(import_from_path+'/df_sentences_semantic_proactivity_'+doc_type_name)
                self.logger.info(f"df_sentences_semantic_proactivity_{doc_type_name}: %d rows", len(df_sentences_semantic_proactivity))
            else:
                df_sentences_semantic_proactivity = search_by_companies(
                    companies=self.list_entities,
                    keywords=self.keywords,
                    sentences=self.list_sentences_proactivity,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    fiscal_year=fiscal_year,
                    scope=doc_type, 
                    freq=self.search_frequency,
                    document_limit=doc_limit,
                    batch_size=self.batch_size
                )
                self.logger.info(f"df_sentences_semantic_proactivity_{doc_type_name}: %d rows", len(df_sentences_semantic_proactivity))
                # Export to Pickle if path provided
                if export_to_path:
                    df_sentences_semantic_proactivity.to_pickle(export_to_path+'/df_sentences_semantic_proactivity_'+doc_type_name)

        else:

            df_sentences_semantic_risk = None
            df_sentences_semantic_proactivity = None

        return df_sentences_semantic_risk, df_sentences_semantic_proactivity
    

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

        ### Step 1: Searches
        # Run a (hybrid semantic) search on News via BigData API with our parameters
        df_sentences_semantic_risk_news, df_sentences_semantic_proactivity_news = self.retrieve_documents(
            doc_type=DocumentType.NEWS, doc_limit=self.document_limit_news, doc_type_name='news', fiscal_year=None,
            import_from_path=import_from_path, export_to_path=export_to_path)
        df_sentences_semantic_risk_filings, df_sentences_semantic_proactivity_filings = self.retrieve_documents(
            doc_type=DocumentType.FILINGS, doc_limit=self.document_limit_filings, doc_type_name='filings', fiscal_year=self.fiscal_year,
            import_from_path=import_from_path, export_to_path=export_to_path)
        df_sentences_semantic_risk_transcripts, df_sentences_semantic_proactivity_transcripts = self.retrieve_documents(
            doc_type=DocumentType.TRANSCRIPTS, doc_limit=self.document_limit_transcripts, doc_type_name='transcripts', fiscal_year=self.fiscal_year,
            import_from_path=import_from_path, export_to_path=export_to_path)
        df_sentences_semantic_risk = pd.concat([df_sentences_semantic_risk_news, df_sentences_semantic_risk_filings, df_sentences_semantic_risk_transcripts])
        df_sentences_semantic_proactivity = pd.concat([df_sentences_semantic_proactivity_news, 
                                                       df_sentences_semantic_proactivity_filings, df_sentences_semantic_proactivity_transcripts])


        ### Step 2: Check that the search results are related to the main theme 

        # Label the search results with our theme labels
        labeler = ScreenerLabeler(llm_model=self.llm_model)

        # Risk
        # Attempt to import df_risk_labeled DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/df_risk_labeled'):
            self.logger.info("Importing df_risk_labeled DataFrame from pickle file.")
            df_risk_labeled = pd.read_pickle(import_from_path+'/df_risk_labeled')
        else:
            df_risk_labels = labeler.get_labels(
                main_theme=self.main_theme_risk,
                labels=['risk'], 
                texts=df_sentences_semantic_risk["masked_text"].tolist()        
            )
            df_risk_labeled = pd.merge(df_sentences_semantic_risk, df_risk_labels, left_index=True, right_index=True)
            # Export to Pickle if path provided
            if export_to_path:
                df_risk_labeled.to_pickle(export_to_path+'/df_risk_labeled')
        self.logger.info("df_risk_labeled: %d rows", len(df_risk_labeled))

        # Proactivity
        # Attempt to import df_proactivity_labeled DataFrame if path provided and file exists
        if import_from_path and os.path.isfile(import_from_path+'/df_proactivity_labeled'):
            self.logger.info("Importing df_proactivity_labeled DataFrame from pickle file.")
            df_proactivity_labeled = pd.read_pickle(import_from_path+'/df_proactivity_labeled')
        else:
            df_proactivity_labels = labeler.get_labels(
                main_theme=self.main_theme_proactivity,
                labels=['proactivity'], 
                texts=df_sentences_semantic_proactivity["masked_text"].tolist()        
            )
            df_proactivity_labeled = pd.merge(df_sentences_semantic_proactivity, df_proactivity_labels, left_index=True, right_index=True)
            # Export to Pickle if path provided
            if export_to_path:
                df_proactivity_labeled.to_pickle(export_to_path+'/df_proactivity_labeled')
        self.logger.info("df_proactivity_labeled: %d rows", len(df_proactivity_labeled))


        # Keep only the content labeled as relevant
        df_risk_labeled_relevant = df_risk_labeled.loc[~df_risk_labeled.label.isin(['', 'unassigned', 'unclear'])].copy()
        df_proactivity_labeled_relevant = df_proactivity_labeled.loc[~df_proactivity_labeled.label.isin(['', 'unassigned', 'unclear'])].copy()


        ### Step 3: Summarize at company level.

        # Run the process to summarize the documents and score media attention, risk and uncertainty by topic at company level.
        summarizer_company = SummarizerCompany(
            model=self.llm_model.split('::')[1], 
            api_key=self.api_key,
            logger=self.logger, 
            verbose=True
        )


        if import_from_path == None:
            path_import_risk = None
        else:
            path_import_risk = import_from_path+'/df_risk_by_company'

        if export_to_path == None:
            path_export_risk = None
        else:
            path_export_risk = export_to_path+'/df_risk_by_company'

        df_risk_by_company = asyncio.run(
            summarizer_company.process_by_company(
                df_labeled=df_risk_labeled_relevant, 
                list_entities=self.list_entities, 
                theme=self.main_theme_risk,
                focus='risk',
                import_from_path=path_import_risk, 
                export_to_path=path_export_risk
            )
        )
        self.logger.info("df_risk_by_company: %d rows", len(df_risk_by_company))


        if import_from_path == None:
            path_import_proactivity = None
        else:
            path_import_proactivity = import_from_path+'/df_proactivity_by_company'

        if export_to_path == None:
            path_export_proactivity = None
        else:
            path_export_proactivity = export_to_path+'/df_proactivity_by_company'

        df_proactivity_by_company = asyncio.run(
            summarizer_company.process_by_company(
                df_labeled=df_proactivity_labeled_relevant, 
                list_entities=self.list_entities, 
                theme=self.main_theme_proactivity,
                focus='',
                import_from_path=path_import_proactivity, 
                export_to_path=path_export_proactivity
            )
        )
        self.logger.info("df_proactivity_by_company: %d rows", len(df_proactivity_by_company))

        # Merge risk and proactivity dataframes
        dfr = df_risk_by_company[['entity_id', 'entity_name', 'topic_summary', 'n_documents']].copy()
        dfr = dfr.rename(columns={'topic_summary': 'risk_summary', 'n_documents': 'n_documents_risk'})
        dfp = df_proactivity_by_company[['entity_id', 'entity_name', 'topic_summary', 'n_documents']].copy()
        dfp = dfp.rename(columns={'topic_summary': 'proactivity_summary', 'n_documents': 'n_documents_proactivity'})
        df_by_company = dfr.merge(dfp, on=['entity_id', 'entity_name'], how='outer')

        df_by_company['n_documents_risk'] = df_by_company['n_documents_risk'].fillna(0)
        df_by_company['n_documents_proactivity'] = df_by_company['n_documents_proactivity'].fillna(0)
        df_by_company['ai_disruption_risk_score'] = df_by_company['n_documents_risk']/df_by_company['n_documents_risk'].mean()
        df_by_company['ai_proactivity_score'] = df_by_company['n_documents_proactivity']/df_by_company['n_documents_proactivity'].mean()
        df_by_company['ai_proactivity_minus_disruption_risk_score'] = df_by_company['ai_proactivity_score'] - df_by_company['ai_disruption_risk_score']

        # Add concatanated quotes and document ids
        df_quotes_risk = self.aggregate_verbatim(df_risk_labeled_relevant, 'risk')
        df_by_company = pd.merge(df_by_company, df_quotes_risk, how='left', on='entity_name')
        df_quotes_proactivity = self.aggregate_verbatim(df_proactivity_labeled_relevant, 'proactivity')
        df_by_company = pd.merge(df_by_company, df_quotes_proactivity, how='left', on='entity_name')

        # Construct the Report
        report = Report(
            watchlist_name=watchlist.name,
            report_by_company=df_by_company
        )
            
        return report

    def plot_company_scores(self, df, score_1, score_2, title):
        """
        Plots entity_name on a plane defined by score_1 and score_2.

        Parameters:
        df (pd.DataFrame): DataFrame containing score_1, score_2, and 'entity_name' columns.
        """
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot: each point represents a company
        ax.scatter(df[score_1], df[score_2], color='blue', alpha=0.6)

        # Add the company name as a label on each point
        for _, row in df.iterrows():
            ax.annotate(row['entity_name'],
                        (row[score_1], row[score_2]),
                        textcoords="offset points",  # Use offset to position text more cleanly
                        xytext=(5, 5),               # Offset: 5 points right and 5 points up
                        ha='left',                   # Horizontal alignment of text
                        fontsize=9)                  # Adjust font size as needed

        # Set plot labels and title
        ax.set_xlabel('AI Disruption Risk Score')
        ax.set_ylabel('AI Proactivity Score')
        ax.set_title(title)

        # Optionally, adjust grid or styling
        ax.grid(True)

        # Save the figure
        plt.savefig('company_scores.png', dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()
