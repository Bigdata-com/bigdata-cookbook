from logging import Logger, getLogger
from typing import Dict, List, Optional

import os
import logging
import pandas as pd
from pandas import merge
import pickle
import asyncio
from types import SimpleNamespace
import matplotlib.pyplot as plt

from src.bigdata_rest import BigdataRestClient, load_universe
from src.labeling import SimpleLabeler
from src.summary.summary import SummarizerCompany
from src.search_helper import run_universe_search


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
        universe_df: pd.DataFrame,
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
        chunk_percentage: float = 0.05,
    ):
        """
        Initialize the GenerateReport class.

        :param universe_df: DataFrame with RP_ENTITY_ID and COMPANY_NAME columns.
        :param keywords: Keywords for search.
        :param main_theme_risk: Main risk theme.
        :param main_theme_proactivity: Main proactivity theme.
        :param list_sentences_risks: Semantic search sentences for risk.
        :param list_sentences_proactivity: Semantic search sentences for proactivity.
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
        self.chunk_percentage = chunk_percentage
        self.rest_client = BigdataRestClient()


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


    def retrieve_documents(self, doc_scope, doc_limit, doc_type_name, fiscal_year, import_from_path: Optional[str] = None, export_to_path: Optional[str] = None):
        if doc_limit>0:
            # Build company_ids and id_to_name mapping
            company_ids = self.universe_df["RP_ENTITY_ID"].astype(str).str.strip().tolist()
            id_to_name = dict(
                zip(
                    self.universe_df["RP_ENTITY_ID"].astype(str).str.strip(),
                    self.universe_df["COMPANY_NAME"].astype(str).str.strip(),
                )
            )

            # Retrieve content for Risk

            if import_from_path and os.path.isfile(import_from_path+'/df_sentences_semantic_risk_'+doc_type_name):
                
                self.logger.info(f"Importing df_sentences_semantic_risk_{doc_type_name} DataFrame from pickle file.")
                df_sentences_semantic_risk = pd.read_pickle(import_from_path+'/df_sentences_semantic_risk_'+doc_type_name)
                self.logger.info(f"df_sentences_semantic_risk_{doc_type_name}: %d rows", len(df_sentences_semantic_risk))
            else:
                # Use run_universe_search with risk sentences as queries
                df_sentences_semantic_risk = run_universe_search(
                    company_ids=company_ids,
                    queries=self.list_sentences_risks,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    scope=doc_scope,
                    chunk_percentage=self.chunk_percentage,
                    id_to_name=id_to_name,
                )
                # run_universe_search already provides a "text" column identical to
                # "chunk_text"; drop the duplicate instead of renaming into a collision.
                df_sentences_semantic_risk = df_sentences_semantic_risk.drop(columns=["chunk_text"], errors="ignore")
                df_sentences_semantic_risk["document_type"] = doc_scope
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
                # Use run_universe_search with proactivity sentences as queries
                df_sentences_semantic_proactivity = run_universe_search(
                    company_ids=company_ids,
                    queries=self.list_sentences_proactivity,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    scope=doc_scope,
                    chunk_percentage=self.chunk_percentage,
                    id_to_name=id_to_name,
                )
                # run_universe_search already provides a "text" column identical to
                # "chunk_text"; drop the duplicate instead of renaming into a collision.
                df_sentences_semantic_proactivity = df_sentences_semantic_proactivity.drop(columns=["chunk_text"], errors="ignore")
                df_sentences_semantic_proactivity["document_type"] = doc_scope
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
          1. Retrieve documents via smart-batching search.
          2. Label documents with OpenAI.
          3. Summarize at company level.
          4. Build the final Report object.

        :param import_from_path: Optional directory to import cached data.
        :param export_to_path: Optional directory to export processed data.
        :return: A Report object with the consolidated results.
        """

        ### Step 1: Searches
        # Run a (hybrid semantic) search on News via BigData API with our parameters
        df_sentences_semantic_risk_news, df_sentences_semantic_proactivity_news = self.retrieve_documents(
            doc_scope="news", doc_limit=self.document_limit_news, doc_type_name='news', fiscal_year=None,
            import_from_path=import_from_path, export_to_path=export_to_path)
        df_sentences_semantic_risk_filings, df_sentences_semantic_proactivity_filings = self.retrieve_documents(
            doc_scope="filings", doc_limit=self.document_limit_filings, doc_type_name='filings', fiscal_year=self.fiscal_year,
            import_from_path=import_from_path, export_to_path=export_to_path)
        df_sentences_semantic_risk_transcripts, df_sentences_semantic_proactivity_transcripts = self.retrieve_documents(
            doc_scope="transcripts", doc_limit=self.document_limit_transcripts, doc_type_name='transcripts', fiscal_year=self.fiscal_year,
            import_from_path=import_from_path, export_to_path=export_to_path)
        df_sentences_semantic_risk = pd.concat([df_sentences_semantic_risk_news, df_sentences_semantic_risk_filings, df_sentences_semantic_risk_transcripts])
        df_sentences_semantic_proactivity = pd.concat([df_sentences_semantic_proactivity_news, 
                                                       df_sentences_semantic_proactivity_filings, df_sentences_semantic_proactivity_transcripts])


        ### Step 2: Check that the search results are related to the main theme 

        # Label the search results with our theme labels
        labeler = SimpleLabeler(model=self.llm_model, api_key=self.api_key)

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
            model=self.llm_model,
            api_key=self.api_key,
            logger=self.logger,
            verbose=True
        )

        # SummarizerCompany.process_entity_topic expects objects with .id / .name
        # attributes (matching the old SDK's entity objects); the universe
        # DataFrame only has RP_ENTITY_ID / COMPANY_NAME columns, so adapt it here.
        list_entities = [
            SimpleNamespace(
                id=str(row.RP_ENTITY_ID).strip(),
                name=str(row.COMPANY_NAME).strip(),
            )
            for row in self.universe_df.itertuples()
        ]


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
                list_entities=list_entities,
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
                list_entities=list_entities,
                theme=self.main_theme_proactivity,
                focus='',
                import_from_path=path_import_proactivity, 
                export_to_path=path_export_proactivity
            )
        )
        self.logger.info("df_proactivity_by_company: %d rows", len(df_proactivity_by_company))

        # Merge risk and proactivity dataframes
        def _company_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(
                    {
                        "entity_id": [entity.id for entity in list_entities],
                        "entity_name": [entity.name for entity in list_entities],
                        "topic_summary": [None] * len(list_entities),
                        "n_documents": [0] * len(list_entities),
                    }
                )
            return df[["entity_id", "entity_name", "topic_summary", "n_documents"]].copy()

        dfr = _company_summary_frame(df_risk_by_company)
        dfr = dfr.rename(columns={"topic_summary": "risk_summary", "n_documents": "n_documents_risk"})
        dfp = _company_summary_frame(df_proactivity_by_company)
        dfp = dfp.rename(
            columns={"topic_summary": "proactivity_summary", "n_documents": "n_documents_proactivity"}
        )
        df_by_company = dfr.merge(dfp, on=["entity_id", "entity_name"], how="outer")

        df_by_company["n_documents_risk"] = df_by_company["n_documents_risk"].fillna(0)
        df_by_company["n_documents_proactivity"] = df_by_company["n_documents_proactivity"].fillna(0)
        risk_mean = df_by_company["n_documents_risk"].mean()
        proactivity_mean = df_by_company["n_documents_proactivity"].mean()
        if risk_mean > 0:
            df_by_company["ai_disruption_risk_score"] = (
                df_by_company["n_documents_risk"] / risk_mean
            )
        else:
            df_by_company["ai_disruption_risk_score"] = 0.0
        if proactivity_mean > 0:
            df_by_company["ai_proactivity_score"] = (
                df_by_company["n_documents_proactivity"] / proactivity_mean
            )
        else:
            df_by_company["ai_proactivity_score"] = 0.0
        df_by_company['ai_proactivity_minus_disruption_risk_score'] = df_by_company['ai_proactivity_score'] - df_by_company['ai_disruption_risk_score']

        # Add concatanated quotes and document ids
        df_quotes_risk = self.aggregate_verbatim(df_risk_labeled_relevant, 'risk')
        df_by_company = pd.merge(df_by_company, df_quotes_risk, how='left', on='entity_name')
        df_quotes_proactivity = self.aggregate_verbatim(df_proactivity_labeled_relevant, 'proactivity')
        df_by_company = pd.merge(df_by_company, df_quotes_proactivity, how='left', on='entity_name')

        # Construct the Report
        report = Report(
            watchlist_name="Company Universe",
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
