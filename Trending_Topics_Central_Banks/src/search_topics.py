
from datetime import datetime
from typing import List, Optional, Dict
from bigdata_client.query import Keyword, Source, Any
import pandas as pd
from bigdata_client import Bigdata
from bigdata_client.query import Similarity, Entity, Source, Document, Keyword, Any
from bigdata_client.models.search import DocumentType, SortBy
from bigdata_client.daterange import RollingDateRange, AbsoluteDateRange
from bigdata_research_tools.search.query_builder import create_date_ranges, _expand_queries
from bigdata_research_tools.search import run_search
from collections import defaultdict
from tqdm.notebook import tqdm
from pqdm.threads import pqdm
from operator import itemgetter

from logging import Logger, getLogger
logger: Logger = getLogger(__name__)


def search_by_keywords(keywords: List[str], start_date: str, end_date: str, scope = DocumentType.NEWS, fiscal_year: Optional[int] = None, sources: Optional[List[str]] = None, freq:str='D', sort_by: SortBy = SortBy.RELEVANCE, document_limit:int=100, **kwargs):

    if scope == DocumentType.NEWS:
        if fiscal_year is not None:
            raise ValueError(
                f"`fiscal_year` must be None when `document_scope` is `{scope.value}`"
            )
        
    #print('input type', type(start_date), type(end_date))

    date_ranges = create_date_ranges(start_date, end_date, freq=freq)

    # print('create date range type', type(date_ranges))
    # print('create date range element type', type(date_ranges[0]))

    queries = [Keyword(keyword) for keyword in keywords]

    if sources or fiscal_year:
        sources = [Source(source) for source in sources] if sources else None
        queries = _expand_queries(queries, source_query = sources if sources else None, fiscal_year = fiscal_year if fiscal_year else None)

    no_queries = len(queries)
    no_dates = len(date_ranges)
    total_no = no_dates * no_queries

    print(f"About to run {total_no} queries")
    print("Example Query:", queries[0], "over date range:", date_ranges[0])
    # Run concurrent search
    results = run_search(
        queries=queries,
        date_ranges=date_ranges,
        limit=document_limit,
        scope=scope,
        sortby=sort_by,
        rerank_threshold=None,
        only_results=False,
        **kwargs,
    )

    results_flattened = process_keywords_search_results(results)

    # Concatenate all chunks
    chunks = results_flattened.sort_values('timestamp')
    chunks = chunks.drop_duplicates(subset=['timestamp', 'headline', 'text']).reset_index(drop=True)

    # Convert 'timestamp' to datetime and extract the date
    chunks['date'] = pd.to_datetime(chunks['timestamp']).dt.date

    # Count chunks by date and keyword
    daily_keyword_counts = chunks.groupby(['date', 'keyword']).size().reset_index(name='number_of_chunks')

    return chunks, daily_keyword_counts

def process_keywords_search_results(results:dict):
    
    results_flattened = pd.DataFrame()
    ##post process results and add keyword to df
    for k, v in results.items():
        keyword_searched = k[0].to_dict()['value'][0].replace("&quot", "")
        
        chunk_results = []
        for result in v:
            for chunk in result.chunks:
                chunk_results.append({'timestamp':result.timestamp, 'rp_document_id':result.id,'headline':result.headline,'chunk_number':chunk.chunk, 'sentence_id':f'{result.id}-{chunk.chunk}', 'source_id':result.source.key,'source_name':result.source.name, 'text':chunk.text})
            if result.cluster:
                for cluster_doc in result.cluster:
                    for chunk in cluster_doc.chunks:
                        chunk_results.append({'timestamp':cluster_doc.timestamp, 'rp_document_id':cluster_doc.id,'headline':cluster_doc.headline, 'chunk_number':chunk.chunk,'sentence_id':f'{cluster_doc.id}-{chunk.chunk}', 'source_id':cluster_doc.source.key,'source_name':cluster_doc.source.name, 'text':chunk.text})

        all_chunks = pd.DataFrame(chunk_results)
        all_chunks['keyword'] = keyword_searched
        results_flattened = pd.concat([results_flattened, all_chunks], ignore_index=True)
    return results_flattened



def run_search_keyword_parallel(keywords, start_query, end_query, bigdata_credentials, limit=100, job_number=10, freq='D'):
    start_list, end_list = create_dates_lists(pd.to_datetime(start_query), pd.to_datetime(end_query), freq=freq)
    args = []

    # List to store results with keywords
    chunks_with_keywords = []

    for period in range(1, len(start_list) + 1):
        start_date = start_list[period - 1]
        end_date = end_list[period - 1]
        print(f'{start_date} - {end_date}')
        date_range = AbsoluteDateRange(start_date, end_date)

        for keyword in keywords:
            # Modify the args to include the keyword information
            args.append({'query': Keyword(keyword), 'date_range': date_range, 'bigdata_credentials': bigdata_credentials, 'limit': limit, 'keyword': keyword})

    # Run the search in parallel
    chunks_final = pqdm(args, run_single_search, n_jobs=job_number, argument_type='kwargs')

    # Concatenate all chunks
    chunks = pd.concat(chunks_final).sort_values('timestamp').reset_index(drop=True)
    chunks = chunks.drop_duplicates(subset=['timestamp', 'headline', 'text'])

    # Convert 'timestamp' to datetime and extract the date
    chunks['date'] = pd.to_datetime(chunks['timestamp']).dt.date

    # Count chunks by date and keyword
    daily_keyword_counts = chunks.groupby(['date', 'keyword']).size().reset_index(name='number_of_chunks')

    return chunks, daily_keyword_counts

def run_single_search(query, date_range,bigdata_credentials, limit, keyword = ''):
    chunk_results = []
    bigdata = bigdata_credentials
    search = bigdata.search.new(
                    query=query,
                    date_range= date_range,
                    scope=DocumentType.NEWS,
                    sortby=SortBy.RELEVANCE,
                )
    # Limit the number of documents to retrieve
    results = search.limit_documents(limit)
    for result in results:
        for chunk in result.chunks:
            chunk_results.append({'timestamp':result.timestamp, 'rp_document_id':result.id,'headline':result.headline,'chunk_number':chunk.chunk, 'sentence_id':f'{result.id}-{chunk.chunk}', 'source_id':result.source.key,'source_name':result.source.name, 'text':chunk.text})
        if result.cluster:
            for cluster_doc in result.cluster:
                for chunk in cluster_doc.chunks:
                    chunk_results.append({'timestamp':cluster_doc.timestamp, 'rp_document_id':cluster_doc.id,'headline':cluster_doc.headline, 'chunk_number':chunk.chunk,'sentence_id':f'{cluster_doc.id}-{chunk.chunk}', 'source_id':cluster_doc.source.key,'source_name':cluster_doc.source.name, 'text':chunk.text})

    all_chunks = pd.DataFrame(chunk_results)
    all_chunks['keyword'] = keyword
    return all_chunks

def filter_headline_keywords(headlines, keywords):
    import re
    results = []
    for x in headlines:
        for w in keywords:
            if re.search("\\b{}\\b".format(w), x.lower()):
                #print(x.lower())
                results.append(x)
                break
    return results

def search_keyword(report_headline_keywords, start_query, end_query, report_headlines_list,bigdata_credentials, job_number = 10, limit=750, freq = 'D'):
    
    chunk_text, daily_keyword_counts = run_search_keyword_parallel(keywords = report_headline_keywords, start_query = start_query, end_query= end_query, bigdata_credentials=bigdata_credentials, limit=limit, job_number=job_number, freq = 'D')
    
    #post_processed_text = chunk_text.loc[chunk_text.headline.isin(report_headlines_list)]
    
    monthly_report_ids = chunk_text.rp_document_id.unique()
    
    return chunk_text, daily_keyword_counts 

def create_dates_lists(start_date, end_date, freq='D'):
    print(f"Creating date ranges from {start_date} to {end_date} with frequency '{freq}'")
    from datetime import datetime
    ## Convert start_date and end_date to datetime objects if they are not already
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    # Initialize lists
    start_list = []
    end_list = []
    # Generate date ranges
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    for i in range(len(date_range) - 1):
        start_list.append(date_range[i].strftime('%Y-%m-%d 00:00:00'))
        end_list.append((date_range[i + 1] - pd.Timedelta(days=1)).strftime('%Y-%m-%d 23:59:59'))
    # Handle the last range
    start_list.append(date_range[-1].strftime('%Y-%m-%d 00:00:00'))
    end_list.append(end_date.strftime('%Y-%m-%d 23:59:59'))
    return start_list, end_list

def create_parallel_query_objects(sentences, query_type = 'Similarity', entities = []):
    if not sentences:
        raise ValueError("The list of sentences must not be empty.")
    
    def contains_lists(lst):
        for item in lst:
            if isinstance(item, list):
                return True
        return False
    if contains_lists(sentences):
        final_list = list(set(flatten(sentences)))
    else:
        final_list = sentences

    if query_type == 'Similarity':
        sentences = [Similarity(sample) for sample in final_list]
    elif query_type == 'Keyword':
        sentences = [Keyword(sample) for sample in final_list]
    else:
        raise ValueError("The query type is not specified.")
    
    if entities:
        ents = [Entity(ent) for ent in entities]
        query = [sentence & Any(ents) for sentence in sentences]
        
    else:
        query=sentences
    
    return query