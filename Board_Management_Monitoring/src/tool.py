import os
import time
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

# Bigdata client imports - only what's needed
from bigdata_client.query import Keyword, Entity, Similarity, Source, Any as QueryAny
from bigdata_client.daterange import AbsoluteDateRange
from bigdata_client.models.search import DocumentType, SortBy


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@contextmanager
def timer(operation_name: str):
    """
    Context manager to time a block of code and log the elapsed time.

    Parameters:
        operation_name (str): A descriptive name for the timed operation.
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        print(f"{operation_name} completed in {duration:.2f} seconds")


# =============================================================================
# DATE UTILITIES
# =============================================================================

def parse_date(date_input):
    """Convert various date formats to date object"""
    from datetime import datetime, date
    
    if isinstance(date_input, str):
        return datetime.strptime(date_input, "%Y-%m-%d").date()
    elif isinstance(date_input, datetime):
        return date_input.date()
    elif isinstance(date_input, date):
        return date_input
    else:
        raise ValueError(f"Unsupported date format: {type(date_input)}")


# =============================================================================
# PRE-PROCESSING FUNCTIONS (Query Preparation)
# =============================================================================

def create_source_filter(source_ids: List[str]):
    """
    Create a query filtering component for a list of source IDs.

    Parameters:
        source_ids (list): A list of source ID strings.

    Returns:
        A query component that filters documents by these source IDs.
    """
    return QueryAny([Source(source_id) for source_id in source_ids])


def build_queries_for_monitoring(
    date_periods: List[Tuple],
    persons: Dict[str, Dict[str, Any]],
    company: Dict[str, str],
    management_themes: List[str],
    board_themes: List[str],
    search_mode: str = "strict",
    sources: Optional[Dict[str, str]] = None,
    use_source_filter: bool = True
) -> Tuple[List, List, List]:
    """
    Build base queries for board/management monitoring (without date multiplication).

    Parameters:
        date_periods: List of (start_date, end_date) tuples
        persons: Dictionary of persons with their name variations
        company: Dictionary containing company info with 'id' field
        management_themes: List of management themes
        board_themes: List of board themes
        search_mode: "strict", "relaxed", or "relaxed_post"
        sources: Dictionary of source names to IDs (optional)
        use_source_filter: Whether to apply source filtering

    Returns:
        Tuple of (base_queries, date_ranges_list, query_details_template)
    """
    base_queries = []
    query_details_template = []

    # Generate source filter if requested
    source_filter = None
    if use_source_filter and sources:
        source_filter = create_source_filter(list(sources.values()))
        print(f"Using source filter with {len(sources)} trusted sources")
    else:
        print("Using all available news sources (no source filtering)")

    # Build date ranges list directly from tuples as ISO strings for JSON compatibility
    date_ranges_list = []
    for start, end in date_periods:
        start_date = parse_date(start)
        end_date = parse_date(end)
        # Convert to ISO string tuples to avoid JSON serialization issues in tracing
        start_iso = datetime.combine(start_date, datetime.min.time())
        end_iso = datetime.combine(end_date, datetime.min.time())
        date_ranges_list.append((start_iso, end_iso))

    # Build base queries
    for person_name, data in persons.items():
        variations = data.get("variations", [person_name])
        person_component = QueryAny([Keyword(var) for var in variations])

        # In strict mode, require company entity
        if search_mode == "strict":
            company_component = Entity(company['id'])
            combined_component = company_component & person_component
        else:
            combined_component = person_component

        # Build queries for management themes
        for theme in management_themes:
            if source_filter:
                q = combined_component & Similarity(theme) & source_filter
            else:
                q = combined_component & Similarity(theme)
            base_queries.append(q)
            query_details_template.append({
                "person": person_name,
                "theme": theme,
                "theme_type": "management"
            })

        # Build queries for board themes
        for theme in board_themes:
            if source_filter:
                q = combined_component & Similarity(theme) & source_filter
            else:
                q = combined_component & Similarity(theme)
            base_queries.append(q)
            query_details_template.append({
                "person": person_name,
                "theme": theme,
                "theme_type": "board"
            })

    print(f"Prepared {len(base_queries)} base queries and {len(date_ranges_list)} date ranges")
    return base_queries, date_ranges_list, query_details_template


# =============================================================================
# POST-PROCESSING FUNCTIONS
# =============================================================================

def company_mentioned(annotated: Dict[str, Any], company_name: str) -> bool:
    """
    Check if the main company name appears within the annotated entities of a document.

    Parameters:
        annotated (dict): The annotated dictionary downloaded from a document.
        company_name (str): The standardized company name to search for.

    Returns:
        bool: True if the company name is found in the 'COMP' category, False otherwise.
    """
    analytics = annotated.get("analytics", {})
    entities = analytics.get("entities", [])
    companies_found = set(ent.get("entity_name") for ent in entities if ent.get("entity_type") == "COMP")
    return company_name in companies_found


def process_doc_to_dict(doc) -> Dict[str, Any]:
    """
    Download annotated metadata from a document and extract key fields.

    Parameters:
        doc (Document): A document object returned from the Bigdata API.

    Returns:
        dict: A dictionary with the key fields extracted.
    """
    # Extract text chunks directly from the document object
    text_chunks = "\n\n".join([chunk.text for chunk in doc.chunks]) if doc.chunks else ""

    # Download annotations for entity and topic information
    try:
        annotated = doc.download_annotated_dict()
    except Exception as e:
        print(f"Could not download annotated data for doc {doc.id}: {e}")
        annotated = {}

    doc_info = annotated.get("document", {})
    metadata = doc_info.get("metadata", {})
    url = metadata.get("url", "")

    # Extract analytics information
    analytics = annotated.get("analytics", {})
    entities = analytics.get("entities", [])
    grouped_entities = {}

    # Group entities by their type
    for ent in entities:
        etype = ent.get("entity_type")
        ename = ent.get("entity_name")
        if etype and ename:
            grouped_entities.setdefault(etype, set()).add(ename)

    companies_entities = ", ".join(sorted(grouped_entities.get("COMP", [])))
    people_entities = ", ".join(sorted(grouped_entities.get("PEOP", [])))
    topic_types = ['TOPI', 'BUSI', 'CEPT', 'SECT']
    all_topics = set()
    for t in topic_types:
        if t in grouped_entities:
            all_topics.update(grouped_entities[t])
    topics = ", ".join(sorted(all_topics))

    return {
        "Date": doc.timestamp,
        "SourceID": metadata.get("source_id", getattr(doc.source, "id", getattr(doc.source, "name", ""))),
        "DocumentID": doc.id,
        "Headline": doc.headline,
        "URL": url,
        "textChunks": text_chunks,
        "Companies_entities": companies_entities,
        "People_entities": people_entities,
        "Topics": topics
    }


def filter_documents_by_company(
    search_results, 
    query_details_template: List[Dict[str, str]], 
    company_name: str, 
    search_mode: str
) -> List[Tuple[Any, Dict[str, str]]]:
    """
    Filter documents based on company mention (for relaxed_post mode).

    Parameters:
        search_results: Results from run_search (list format expected)
        query_details_template: Template of query metadata
        company_name: Main company name to filter by
        search_mode: The search mode ("strict", "relaxed", "relaxed_post")

    Returns:
        List of (document, metadata) tuples
    """
    final_results = []

    if isinstance(search_results, list) and search_results:
        num_queries = len(query_details_template)
        num_dates = len(search_results) // num_queries if num_queries > 0 else 0

        print(f"Processing {num_queries} queries × {num_dates} dates = {len(search_results)} results")

        for result_idx, query_results in enumerate(search_results):
            if query_results:  # Skip empty result lists
                # Calculate which query this result belongs to
                query_idx = result_idx // num_dates
                date_idx = result_idx % num_dates

                if query_idx < len(query_details_template):
                    meta = query_details_template[query_idx]

                    for doc in query_results:
                        if search_mode == "relaxed_post":
                            try:
                                annotated = doc.download_annotated_dict()
                            except Exception as e:
                                print(f"Skipping doc {doc.id} due to error: {e}")
                                continue
                            if not company_mentioned(annotated, company_name):
                                continue
                        final_results.append((doc, meta))

    print(f"Filtered results: {len(final_results)} documents after company filtering")
    return final_results


def deduplicate_results(results: List[Tuple[Any, Dict[str, str]]]) -> List[Tuple[Any, Dict[str, str]]]:
    """
    Remove duplicate documents based on document IDs.

    Parameters:
        results: List of (document, metadata) tuples

    Returns:
        List of deduplicated (document, metadata) tuples
    """
    seen_doc_ids = set()
    deduplicated_results = []

    for doc, meta in results:
        if doc.id not in seen_doc_ids:
            seen_doc_ids.add(doc.id)
            deduplicated_results.append((doc, meta))

    print(f"Original: {len(results)}, After deduplication: {len(deduplicated_results)}")
    return deduplicated_results


def process_results_to_dataframe(
    results: List[Tuple[Any, Dict[str, str]]], 
    df_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Process search results into a pandas DataFrame.

    Parameters:
        results: List of (document, metadata) tuples
        df_columns: List of column names for the DataFrame

    Returns:
        pandas.DataFrame with processed documents
    """
    if df_columns is None:
        df_columns = [
            "Date", "SourceID", "DocumentID", "Headline", "URL",
            "textChunks", "Companies_entities", "People_entities", "Topics", "QueryMeta"
        ]

    processed_docs = []
    total_docs = len(results)
    print(f"Processing {total_docs} documents for enrichment.")

    for idx, (doc, query_meta) in enumerate(results, start=1):
        if idx % 100 == 0:
            print(f"Processed {idx} of {total_docs} documents")
        try:
            doc_dict = process_doc_to_dict(doc)
            # Append query meta information
            doc_dict["QueryMeta"] = f"{query_meta['person']} | {query_meta['theme']} ({query_meta['theme_type']})"
            processed_docs.append(doc_dict)
        except Exception as e:
            print(f"Error processing doc {doc.id}: {str(e)}")

    df_results = pd.DataFrame(processed_docs, columns=df_columns)
    print(f"DataFrame created with {len(df_results)} rows.")
    return df_results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def convert_quarter_to_date(q: str) -> pd.Timestamp:
    """
    Converts a quarter string (e.g., "2023Q1") into a Timestamp representing the start date of that quarter.
    """
    year, quarter_str = q.split("Q")
    year = int(year)
    quarter = int(quarter_str)
    month = (quarter - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1)


def get_common_quarter_ticks(start: str = "2020Q1", end: str = "2025Q4") -> Tuple[List[pd.Timestamp], List[str]]:
    """
    Generates a list of tick positions and labels for each quarter between start and end.

    Parameters:
        start (str): The starting quarter, e.g., "2020Q1".
        end (str): The ending quarter, e.g., "2025Q4".

    Returns:
        tick_dates (list): List of quarter start dates.
        tick_text (list): Corresponding string labels.
    """
    quarters = pd.period_range(start=start, end=end, freq='Q')
    tick_dates = [q.start_time for q in quarters]
    tick_text = [str(q) for q in quarters]
    return tick_dates, tick_text


def prepare_quarterly_counts(df: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    """
    Process a DataFrame to extract quarterly document counts and add a mode column.

    Parameters:
        df (pd.DataFrame): DataFrame with at least a 'Date' column.
        mode_name (str): A string to label the search mode.

    Returns:
        pd.DataFrame: A DataFrame with columns: 'Quarter', 'Count', 'quarter_date', and 'Mode'.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    # Remove timezone info before converting to period to avoid warning
    df['Quarter'] = df['Date'].dt.tz_localize(None).dt.to_period('Q').astype(str)
    # Count the documents in each quarter
    quarterly_counts = df.groupby('Quarter').size().reset_index(name='Count')
    # Convert quarter strings to quarter start dates
    quarterly_counts['quarter_date'] = quarterly_counts['Quarter'].apply(convert_quarter_to_date)
    # Tag with the search mode
    quarterly_counts['Mode'] = mode_name
    return quarterly_counts


def plot_combined_monitoring_activity(
    df_strict: pd.DataFrame, 
    df_relaxed: pd.DataFrame, 
    df_relaxed_post: pd.DataFrame,
    title: str = "Monitoring Activity",
    x_range: Optional[List] = None, 
    tick_vals: Optional[List] = None, 
    tick_text: Optional[List] = None,
    interactive: bool = True
):
    """
    Generates a grouped bar chart displaying quarterly document counts for three search modes.

    Parameters:
        df_strict: DataFrame for the "Strict Mode" results.
        df_relaxed: DataFrame for the "Relaxed Mode" results.
        df_relaxed_post: DataFrame for the "Relaxed Post Mode" results.
        title: Title of the chart.
        x_range: Two-item list with the start and end x-axis range.
        tick_vals: List of datetime values for the x-axis ticks.
        tick_text: List of labels corresponding to tick_vals.
        interactive: If True, creates an interactive Plotly chart. If False, creates a static matplotlib chart.

    Returns:
        fig: A Plotly Figure object or matplotlib figure containing the grouped bar chart.
    """
    # Prepare quarterly counts for each search mode
    df1 = prepare_quarterly_counts(df_strict, "Strict")
    df2 = prepare_quarterly_counts(df_relaxed, "Relaxed")
    df3 = prepare_quarterly_counts(df_relaxed_post, "Relaxed Post")

    # Combine the data for all modes
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)

    if interactive:
        # Create interactive Plotly chart
        fig = px.bar(
            combined_df,
            x='quarter_date',
            y='Count',
            color='Mode',
            barmode='group',
            title=title,
            labels={'quarter_date': 'Quarter', 'Count': 'Distinct Documents Retrieved'}
        )

        # Update the x-axis with common tick values if provided
        if tick_vals is not None and tick_text is not None:
            fig.update_xaxes(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text
            )
        if x_range is not None:
            # Add padding to the left and right of the range
            range_start = x_range[0]
            range_end = x_range[1]
            padding = (range_end - range_start) * 0.03  # 3% padding
            fig.update_xaxes(range=[range_start - padding, range_end + padding])

        fig.update_layout(
            xaxis_title="Quarter",
            yaxis_title="Distinct Documents Retrieved",
            template="plotly_white"
        )

        return fig
    
    else:
        # Create static matplotlib chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get unique quarters and modes - use all quarters from tick range if provided
        if tick_vals is not None:
            # Use the full range of quarters from tick_vals
            all_quarters = [pd.Timestamp(tick_val) for tick_val in tick_vals]
            # Convert to quarters and remove duplicates while preserving order
            quarter_periods = []
            seen = set()
            for ts in all_quarters:
                quarter = ts.to_period('Q').start_time
                if quarter not in seen:
                    quarter_periods.append(quarter)
                    seen.add(quarter)
            quarters = sorted(quarter_periods)
        else:
            # Fallback to quarters from data
            quarters = sorted(combined_df['quarter_date'].unique())
        
        modes = combined_df['Mode'].unique()
        
        # Set up bar positions
        x = np.arange(len(quarters))
        width = 0.25
        multiplier = 0
        
        # Color map for consistency
        colors = ['#636EFA', '#EF553B', '#00CC96']
        
        # Create bars for each mode
        for i, mode in enumerate(modes):
            mode_data = combined_df[combined_df['Mode'] == mode]
            counts = []
            
            for quarter in quarters:
                quarter_data = mode_data[mode_data['quarter_date'] == quarter]
                if not quarter_data.empty:
                    counts.append(quarter_data['Count'].iloc[0])
                else:
                    counts.append(0)
            
            offset = width * multiplier
            bars = ax.bar(x + offset, counts, width, label=mode, color=colors[i % len(colors)])
            multiplier += 1
        
        # Customize the chart
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Distinct Documents Retrieved')
        ax.set_title(title)
        ax.set_xticks(x + width)
        
        # Format x-axis labels
        if tick_vals is not None and tick_text is not None:
            # Since we're now using quarters derived from tick_vals, 
            # we can map them more directly
            quarter_labels = []
            for quarter in quarters:
                # Find the corresponding tick_text for this quarter
                quarter_period = quarter.to_period('Q')
                label_found = False
                
                for i, tick_val in enumerate(tick_vals):
                    tick_period = pd.Timestamp(tick_val).to_period('Q')
                    if tick_period == quarter_period and i < len(tick_text):
                        quarter_labels.append(tick_text[i])
                        label_found = True
                        break
                
                if not label_found:
                    quarter_labels.append(str(quarter_period))
            
            ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
        else:
            # Default quarter labels
            quarter_labels = [str(q.to_period('Q')) for q in quarters]
            ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
        
        # Set x-axis range if provided  
        if x_range is not None:
            # For matplotlib, we need to ensure we show the full range
            # Don't restrict the x-axis if it would hide data
            ax.set_xlim(-0.5, len(quarters) - 0.5)
        else:
            # Default: show all quarters with some padding
            ax.set_xlim(-0.5, len(quarters) - 0.5)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig


# =============================================================================
# MAIN WORKFLOW FUNCTION  
# =============================================================================

def run_monitoring_workflow(
    date_periods: List[Tuple],  # Changed parameter name for clarity
    persons: Dict[str, Dict[str, Any]],
    company: Dict[str, str],
    company_name: str,
    management_themes: List[str],
    board_themes: List[str],
    search_mode: str = "strict",
    limit: int = 100,
    sources: Optional[Dict[str, str]] = None,
    use_source_filter: bool = True,
    run_search_function=None,
    **search_kwargs
) -> pd.DataFrame:
    """
    Complete workflow function that combines all steps.

    Parameters:
        date_periods: List of (start_date, end_date) tuples for search periods
        persons: Dictionary of persons with variations
        company: Dictionary containing company info
        company_name: Main company name for filtering
        management_themes: List of management themes
        board_themes: List of board themes
        search_mode: "strict", "relaxed", or "relaxed_post"
        limit: Maximum documents per query
        sources: Dictionary of source names to IDs
        use_source_filter: Whether to apply source filtering
        run_search_function: Your search function
        **search_kwargs: Additional search arguments

    Returns:
        pandas.DataFrame with processed results
        
    Example:
        date_periods = [
            ("2024-12-04", "2025-03-04"),
            ("2024-09-05", "2024-12-04"),
            ("2024-05-30", "2024-09-05"),
        ]
        
        df = run_complete_monitoring_workflow(
            date_periods=date_periods,
            persons=persons_dict,
            company=company_dict,
            company_name="GXO Logistics Inc.",
            management_themes=management_themes,
            board_themes=board_themes,
            search_mode="relaxed_post",
            run_search_function=run_search
        )
    """
    # 1. Build queries
    with timer("Building queries"):
        queries, date_ranges, query_details = build_queries_for_monitoring(
            date_periods=date_periods,
            persons=persons,
            company=company,
            management_themes=management_themes,
            board_themes=board_themes,
            search_mode=search_mode,
            sources=sources,
            use_source_filter=use_source_filter
        )

    # 2. Execute search
    with timer("Executing search"):
        if run_search_function:
            search_results = run_search_function(
                queries=queries,
                date_ranges=date_ranges,
                limit=limit,
                **search_kwargs
            )
        else:
            raise ValueError("run_search_function must be provided")

    # 3. Filter results
    with timer("Filtering results"):
        filtered_results = filter_documents_by_company(
            search_results=search_results,
            query_details_template=query_details,
            company_name=company_name,
            search_mode=search_mode
        )

    # 4. Deduplicate
    with timer("Deduplicating results"):
        deduplicated_results = deduplicate_results(filtered_results)

    # 5. Convert to DataFrame
    with timer("Converting to DataFrame"):
        df_results = process_results_to_dataframe(deduplicated_results)

    return df_results


def plot_management_single_distribution(
    df: pd.DataFrame,
    title: str = "Management vs Board Distribution by Quarter",
    x_range: Optional[List] = None, 
    tick_vals: Optional[List] = None, 
    tick_text: Optional[List] = None,
    interactive: bool = True,
    normalize_to_percentage: bool = True
):
    """
    Generates a bar chart showing the distribution of management vs board themes by quarter.
    
    Parameters:
        df: DataFrame with 'Date' and 'QueryMeta' columns
        title: Title of the chart
        x_range: Two-item list with the start and end x-axis range
        tick_vals: List of datetime values for the x-axis ticks
        tick_text: List of labels corresponding to tick_vals
        interactive: If True, creates an interactive Plotly chart. If False, creates a static matplotlib chart
        normalize_to_percentage: If True, normalizes values to percentages (0-100%). If False, shows absolute counts.
        
    Returns:
        fig: A Plotly Figure object or matplotlib figure containing the bar chart
    """
    if len(df) == 0:
        print("No data available for analysis")
        return None
    
    # Prepare the data
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    # Remove timezone info before converting to period
    df_copy['Quarter'] = df_copy['Date'].dt.tz_localize(None).dt.to_period('Q').astype(str)
    
    # Extract theme type from QueryMeta (management or board)
    df_copy['Theme_Type'] = df_copy['QueryMeta'].str.extract(r'\((management|board)\)')
    
    # Group by quarter and theme type, then count
    quarterly_theme_counts = df_copy.groupby(['Quarter', 'Theme_Type']).size().reset_index(name='Count')
    
    if normalize_to_percentage:
        # Calculate percentages within each quarter
        quarterly_totals = quarterly_theme_counts.groupby('Quarter')['Count'].sum().reset_index()
        quarterly_totals.rename(columns={'Count': 'Total'}, inplace=True)
        
        # Merge to get totals and calculate percentages
        quarterly_theme_counts = quarterly_theme_counts.merge(quarterly_totals, on='Quarter')
        quarterly_theme_counts['Value'] = (quarterly_theme_counts['Count'] / quarterly_theme_counts['Total']) * 100
        y_label = 'Percentage (%)'
        y_max = 100
    else:
        # Use absolute counts
        quarterly_theme_counts['Value'] = quarterly_theme_counts['Count']
        y_label = 'Count'
        y_max = None
    
    # Convert quarter strings to quarter start dates
    quarterly_theme_counts['quarter_date'] = quarterly_theme_counts['Quarter'].apply(convert_quarter_to_date)
    
    if interactive:
        # Create interactive Plotly chart
        fig = px.bar(
            quarterly_theme_counts,
            x='quarter_date',
            y='Value',
            color='Theme_Type',
            title=title,
            labels={'quarter_date': 'Quarter', 'Value': y_label},
            color_discrete_map={'management': '#636EFA', 'board': '#EF553B'}
        )
        
        # Update the x-axis with common tick values if provided
        if tick_vals is not None and tick_text is not None:
            fig.update_xaxes(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                tickangle=45,
                showticklabels=True
            )
        if x_range is not None:
            # Add padding to the left and right of the range
            range_start = x_range[0]
            range_end = x_range[1]
            padding = (range_end - range_start) * 0.03  # 3% padding
            fig.update_xaxes(range=[range_start - padding, range_end + padding])
            
        # Set y-axis properties based on normalization
        layout_updates = {
            "xaxis_title": "Quarter",
            "yaxis_title": y_label,
            "template": "plotly_white"
        }
        if y_max is not None:
            layout_updates["yaxis"] = dict(range=[0, y_max])
            
        fig.update_layout(**layout_updates)
        
        return fig
    
    else:
        # Create static matplotlib chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get unique quarters - use all quarters from tick range if provided
        if tick_vals is not None:
            # Use the full range of quarters from tick_vals
            all_quarters = [pd.Timestamp(tick_val) for tick_val in tick_vals]
            # Convert to quarters and remove duplicates while preserving order
            quarter_periods = []
            seen = set()
            for ts in all_quarters:
                quarter = ts.to_period('Q').start_time
                if quarter not in seen:
                    quarter_periods.append(quarter)
                    seen.add(quarter)
            quarters = sorted(quarter_periods)
        else:
            # Fallback to quarters from data
            quarters = sorted(quarterly_theme_counts['quarter_date'].unique())
        
        # Set up bar positions
        x = np.arange(len(quarters))
        width = 0.35
        
        # Prepare data for each theme type
        management_values = []
        board_values = []
        
        for quarter in quarters:
            quarter_data = quarterly_theme_counts[quarterly_theme_counts['quarter_date'] == quarter]
            
            mgmt_data = quarter_data[quarter_data['Theme_Type'] == 'management']
            board_data = quarter_data[quarter_data['Theme_Type'] == 'board']
            
            mgmt_val = mgmt_data['Value'].iloc[0] if not mgmt_data.empty else 0
            board_val = board_data['Value'].iloc[0] if not board_data.empty else 0
            
            management_values.append(mgmt_val)
            board_values.append(board_val)
        
        # Create bars
        bars1 = ax.bar(x - width/2, management_values, width, label='Management', color='#636EFA')
        bars2 = ax.bar(x + width/2, board_values, width, label='Board', color='#EF553B')
        
        # Customize the chart
        ax.set_xlabel('Quarter')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xticks(x)
        if y_max is not None:
            ax.set_ylim(0, y_max)
        
        # Format x-axis labels
        if tick_vals is not None and tick_text is not None:
            quarter_labels = []
            for quarter in quarters:
                quarter_period = quarter.to_period('Q')
                label_found = False
                
                for i, tick_val in enumerate(tick_vals):
                    tick_period = pd.Timestamp(tick_val).to_period('Q')
                    if tick_period == quarter_period and i < len(tick_text):
                        quarter_labels.append(tick_text[i])
                        label_found = True
                        break
                
                if not label_found:
                    quarter_labels.append(str(quarter_period))
            
            ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
        else:
            # Default quarter labels
            quarter_labels = [str(q.to_period('Q')) for q in quarters]
            ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bars, values in [(bars1, management_values), (bars2, board_values)]:
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    if normalize_to_percentage:
                        label_text = f'{val:.1f}%'
                        offset = 1
                    else:
                        label_text = f'{val:.0f}'
                        offset = max(management_values + board_values) * 0.02 if management_values + board_values else 1
                    ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                           label_text, ha='center', va='bottom', fontsize=9)
        
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return fig


def plot_management_distributions(
    df_strict: pd.DataFrame,
    df_relaxed: pd.DataFrame, 
    df_relaxed_post: pd.DataFrame,
    title: str = "Management vs Board Distribution",
    x_range: Optional[List] = None, 
    tick_vals: Optional[List] = None, 
    tick_text: Optional[List] = None,
    interactive: bool = True,
    normalize_to_percentage: bool = True,
    layout: str = "horizontal",
    show_legend: bool = True
):
    """
    Generates bar charts showing the distribution of management vs board themes by quarter
    for different search modes.
    
    Parameters:
        df_strict: DataFrame for strict mode results
        df_relaxed: DataFrame for relaxed mode results  
        df_relaxed_post: DataFrame for relaxed post mode results
        title: Overall title of the chart
        x_range: Two-item list with the start and end x-axis range
        tick_vals: List of datetime values for the x-axis ticks
        tick_text: List of labels corresponding to tick_vals
        interactive: If True, creates an interactive Plotly chart. If False, creates a static matplotlib chart
        normalize_to_percentage: If True, normalizes values to percentages (0-100%). If False, shows absolute counts.
        layout: "horizontal" for side-by-side (1 row, 3 cols) or "vertical" for stacked (3 rows, 1 col)
        show_legend: If True, shows the legend. If False, hides the legend.
        
    Returns:
        fig: A Plotly Figure object or matplotlib figure containing the three subplots
    """
    
    def prepare_data(df, mode_name):
        """Helper function to prepare data for each DataFrame"""
        if len(df) == 0:
            return pd.DataFrame(), 'Count', None
            
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy['Quarter'] = df_copy['Date'].dt.tz_localize(None).dt.to_period('Q').astype(str)
        df_copy['Theme_Type'] = df_copy['QueryMeta'].str.extract(r'\((management|board)\)')
        
        quarterly_theme_counts = df_copy.groupby(['Quarter', 'Theme_Type']).size().reset_index(name='Count')
        
        if normalize_to_percentage:
            quarterly_totals = quarterly_theme_counts.groupby('Quarter')['Count'].sum().reset_index()
            quarterly_totals.rename(columns={'Count': 'Total'}, inplace=True)
            quarterly_theme_counts = quarterly_theme_counts.merge(quarterly_totals, on='Quarter')
            quarterly_theme_counts['Value'] = (quarterly_theme_counts['Count'] / quarterly_theme_counts['Total']) * 100
            y_label = 'Percentage (%)'
            y_max = 100
        else:
            quarterly_theme_counts['Value'] = quarterly_theme_counts['Count']
            y_label = 'Count'
            y_max = None
            
        quarterly_theme_counts['quarter_date'] = quarterly_theme_counts['Quarter'].apply(convert_quarter_to_date)
        return quarterly_theme_counts, y_label, y_max
    
    # Prepare data for all three DataFrames
    data_strict, y_label, y_max = prepare_data(df_strict, "Strict")
    data_relaxed, _, _ = prepare_data(df_relaxed, "Relaxed") 
    data_relaxed_post, _, _ = prepare_data(df_relaxed_post, "Relaxed Post")
    
    # Define subplot titles
    subplot_titles = [
        "Strict",
        "Relaxed", 
        "Relaxed Post"
    ]
    
    if interactive:
        # Create Plotly subplots
        from plotly.subplots import make_subplots
        
        if layout == "horizontal":
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=subplot_titles,
                shared_yaxes=True,
                horizontal_spacing=0.08
            )
            subplot_positions = [(1, 1), (1, 2), (1, 3)]
            height = 500
            legend_y = -0.2
            margin_b = 100
        else:  # vertical layout
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=subplot_titles,
                shared_xaxes=True,
                vertical_spacing=0.15  # Increased spacing to avoid ticker overlap
            )
            subplot_positions = [(1, 1), (2, 1), (3, 1)]
            height = 1000  # Increased height for better spacing
            legend_y = -0.08
            margin_b = 80
        
        # Add stacked bars for each subplot
        datasets = [data_strict, data_relaxed, data_relaxed_post]
        
        for i, (data, pos) in enumerate(zip(datasets, subplot_positions)):
            if len(data) > 0:
                # Get all unique quarters for this dataset
                quarters = sorted(data['quarter_date'].unique())
                
                mgmt_values = []
                board_values = []
                quarter_dates = []
                
                for quarter in quarters:
                    quarter_data = data[data['quarter_date'] == quarter]
                    mgmt_data = quarter_data[quarter_data['Theme_Type'] == 'management']
                    board_data = quarter_data[quarter_data['Theme_Type'] == 'board']
                    
                    mgmt_val = mgmt_data['Value'].iloc[0] if not mgmt_data.empty else 0
                    board_val = board_data['Value'].iloc[0] if not board_data.empty else 0
                    
                    mgmt_values.append(mgmt_val)
                    board_values.append(board_val)
                    quarter_dates.append(quarter)
                
                # Add stacked bars
                fig.add_bar(
                    x=quarter_dates,
                    y=mgmt_values,
                    name='Management',
                    marker_color='#636EFA',
                    showlegend=(i == 0 and show_legend),  # Show legend only for first subplot and if enabled
                    row=pos[0], col=pos[1]
                )
                
                fig.add_bar(
                    x=quarter_dates,
                    y=board_values,
                    name='Board',
                    marker_color='#EF553B',
                    showlegend=(i == 0 and show_legend),  # Show legend only for first subplot and if enabled
                    row=pos[0], col=pos[1]
                )
        
        # Update axes for all subplots
        for i, pos in enumerate(subplot_positions):
            if tick_vals is not None and tick_text is not None:
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    tickangle=45,
                    showticklabels=True,
                    row=pos[0], col=pos[1]
                )
            if x_range is not None:
                # Add padding to the left and right of the range
                range_start = x_range[0]
                range_end = x_range[1]
                padding = (range_end - range_start) * 0.03  # 3% padding
                fig.update_xaxes(range=[range_start - padding, range_end + padding], row=pos[0], col=pos[1])
        
        # Update y-axes
        for i, pos in enumerate(subplot_positions):
            if layout == "horizontal":
                fig.update_yaxes(title_text=y_label if i == 0 else "", row=pos[0], col=pos[1])
            else:  # vertical layout
                fig.update_yaxes(title_text=y_label, row=pos[0], col=pos[1])
            
            if y_max is not None:
                fig.update_yaxes(range=[0, y_max], row=pos[0], col=pos[1])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,  # Center the title
                xanchor='center'
            ),
            template="plotly_white",
            barmode='stack',  # Make bars stacked
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,  # Position ABOVE the subplot titles, between main title and subplots
                xanchor="right",
                x=0.98  # Position on the right
            ),
            height=height,
            margin=dict(b=margin_b, t=120)  # More top margin for title and legend space
        )
        
        return fig
        
    else:
        # Create matplotlib subplots
        if layout == "horizontal":
            fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)  # Wider and shorter
        else:  # vertical layout
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)  # Wider
        
        datasets = [data_strict, data_relaxed, data_relaxed_post]
        
        for idx, (ax, data, subtitle) in enumerate(zip(axes, datasets, subplot_titles)):
            if len(data) > 0:
                # Get unique quarters
                if tick_vals is not None:
                    all_quarters = [pd.Timestamp(tick_val) for tick_val in tick_vals]
                    quarter_periods = []
                    seen = set()
                    for ts in all_quarters:
                        quarter = ts.to_period('Q').start_time
                        if quarter not in seen:
                            quarter_periods.append(quarter)
                            seen.add(quarter)
                    quarters = sorted(quarter_periods)
                else:
                    quarters = sorted(data['quarter_date'].unique())
                
                # Prepare data for stacked bars
                x = np.arange(len(quarters))
                width = 0.6
                
                management_values = []
                board_values = []
                
                for quarter in quarters:
                    quarter_data = data[data['quarter_date'] == quarter]
                    mgmt_data = quarter_data[quarter_data['Theme_Type'] == 'management']
                    board_data = quarter_data[quarter_data['Theme_Type'] == 'board']
                    
                    mgmt_val = mgmt_data['Value'].iloc[0] if not mgmt_data.empty else 0
                    board_val = board_data['Value'].iloc[0] if not board_data.empty else 0
                    
                    management_values.append(mgmt_val)
                    board_values.append(board_val)
                
                # Create stacked bars
                bars1 = ax.bar(x, management_values, width, label='Management', color='#636EFA')
                bars2 = ax.bar(x, board_values, width, bottom=management_values, label='Board', color='#EF553B')
                
                # Customize subplot
                ax.set_title(subtitle)
                ax.set_xticks(x)
                if y_max is not None:
                    ax.set_ylim(0, y_max)
                
                # Show x-axis labels only on the last subplot
                if layout == "horizontal":
                    is_last_subplot = (idx == 2)  # Last subplot in horizontal layout
                else:  # vertical layout
                    is_last_subplot = (idx == 2)  # Last subplot in vertical layout
                
                if is_last_subplot:
                    # Format x-axis labels for the last subplot
                    if tick_vals is not None and tick_text is not None:
                        quarter_labels = []
                        for quarter in quarters:
                            quarter_period = quarter.to_period('Q')
                            label_found = False
                            for i, tick_val in enumerate(tick_vals):
                                tick_period = pd.Timestamp(tick_val).to_period('Q')
                                if tick_period == quarter_period and i < len(tick_text):
                                    quarter_labels.append(tick_text[i])
                                    label_found = True
                                    break
                            if not label_found:
                                quarter_labels.append(str(quarter_period))
                        ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
                    else:
                        quarter_labels = [str(q.to_period('Q')) for q in quarters]
                        ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
                else:
                    # Remove x-axis labels for other subplots
                    ax.set_xticklabels([])  # No labels on x-axis
                    ax.tick_params(axis='x', which='both', length=0)  # Remove tick marks
                
                # Removed value labels on bars per user request
                
                ax.grid(True, alpha=0.3, axis='y')
        
        # Set labels based on layout
        if layout == "horizontal":
            axes[0].set_ylabel(y_label)
            # Create legend in the center, between title and first graph if enabled
            if show_legend:
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2)
        else:  # vertical layout
            for ax in axes:
                ax.set_ylabel(y_label)
            # Create legend in the center, between title and first graph if enabled
            if show_legend:
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2)
        
        # Set title and adjust layout
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Apply final adjustments after tight_layout
        if layout == "horizontal":
            if show_legend:
                plt.subplots_adjust(bottom=0.05, top=0.88)  # Reduced space between title and subplots
            else:
                plt.subplots_adjust(bottom=0.05, top=0.93)  # Less bottom space without legend
        else:  # vertical layout
            if show_legend:
                plt.subplots_adjust(bottom=0.02, top=0.88)  # Reduced space between title and subplots
            else:
                plt.subplots_adjust(bottom=0.02, top=0.95)  # Less bottom space without legend
        
        return fig


def generate_marketing_plot_distribution(
    df_relaxed: pd.DataFrame,
    title: str = "Management vs Board Distribution",
    x_range: Optional[List] = None, 
    tick_vals: Optional[List] = None, 
    tick_text: Optional[List] = None,
    interactive: bool = True,
    normalize_to_percentage: bool = True,
    show_legend: bool = True
):
    """
    Generates a bar chart showing the distribution of management vs board themes by quarter
    for a single DataFrame (relaxed mode).
    
    Parameters:
        df_relaxed: DataFrame for relaxed mode results
        title: Title of the chart
        x_range: Two-item list with the start and end x-axis range
        tick_vals: List of datetime values for the x-axis ticks
        tick_text: List of labels corresponding to tick_vals
        interactive: If True, creates an interactive Plotly chart. If False, creates a static matplotlib chart
        normalize_to_percentage: If True, normalizes values to percentages (0-100%). If False, shows absolute counts.
        show_legend: If True, shows the legend. If False, hides the legend.
        
    Returns:
        fig: A Plotly Figure object or matplotlib figure containing the bar chart
    """
    
    def prepare_data(df):
        """Helper function to prepare data for the DataFrame"""
        if len(df) == 0:
            return pd.DataFrame(), 'Count', None
            
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy['Quarter'] = df_copy['Date'].dt.tz_localize(None).dt.to_period('Q').astype(str)
        df_copy['Theme_Type'] = df_copy['QueryMeta'].str.extract(r'\((management|board)\)')
        
        quarterly_theme_counts = df_copy.groupby(['Quarter', 'Theme_Type']).size().reset_index(name='Count')
        
        if normalize_to_percentage:
            quarterly_totals = quarterly_theme_counts.groupby('Quarter')['Count'].sum().reset_index()
            quarterly_totals.rename(columns={'Count': 'Total'}, inplace=True)
            quarterly_theme_counts = quarterly_theme_counts.merge(quarterly_totals, on='Quarter')
            quarterly_theme_counts['Value'] = (quarterly_theme_counts['Count'] / quarterly_theme_counts['Total']) * 100
            y_label = 'Percentage (%)'
            y_max = 100
        else:
            quarterly_theme_counts['Value'] = quarterly_theme_counts['Count']
            y_label = 'Count'
            y_max = None
            
        quarterly_theme_counts['quarter_date'] = quarterly_theme_counts['Quarter'].apply(convert_quarter_to_date)
        return quarterly_theme_counts, y_label, y_max
    
    # Prepare data for the DataFrame
    data_relaxed, y_label, y_max = prepare_data(df_relaxed)
    
    if interactive:
        # Create Plotly chart
        if len(data_relaxed) > 0:
            # Get all unique quarters for this dataset
            quarters = sorted(data_relaxed['quarter_date'].unique())
            
            mgmt_values = []
            board_values = []
            quarter_dates = []
            
            for quarter in quarters:
                quarter_data = data_relaxed[data_relaxed['quarter_date'] == quarter]
                mgmt_data = quarter_data[quarter_data['Theme_Type'] == 'management']
                board_data = quarter_data[quarter_data['Theme_Type'] == 'board']
                
                mgmt_val = mgmt_data['Value'].iloc[0] if not mgmt_data.empty else 0
                board_val = board_data['Value'].iloc[0] if not board_data.empty else 0
                
                mgmt_values.append(mgmt_val)
                board_values.append(board_val)
                quarter_dates.append(quarter)
            
            # Create figure manually to match the original layout
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=1, cols=1)
            
            # Add stacked bars
            fig.add_bar(
                x=quarter_dates,
                y=mgmt_values,
                name='Management',
                marker_color='#636EFA',
                showlegend=show_legend
            )
            
            fig.add_bar(
                x=quarter_dates,
                y=board_values,
                name='Board',
                marker_color='#EF553B',
                showlegend=show_legend
            )
        
        # Update axes
        if tick_vals is not None and tick_text is not None:
            fig.update_xaxes(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                tickangle=45,
                showticklabels=True
            )
        if x_range is not None:
            # Add padding to the left and right of the range
            range_start = x_range[0]
            range_end = x_range[1]
            padding = (range_end - range_start) * 0.03  # 3% padding
            fig.update_xaxes(range=[range_start - padding, range_end + padding])
        
        # Update y-axes
        fig.update_yaxes(title_text=y_label)
        if y_max is not None:
            fig.update_yaxes(range=[0, y_max])
        
        # Update layout - IDENTICO alla funzione originale
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,  # Center the title
                xanchor='center'
            ),
            template="plotly_white",
            barmode='stack',  # Make bars stacked
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,  # Position ABOVE the subplot titles, between main title and subplots
                xanchor="right",
                x=0.98  # Position on the right
            ),
            height=500,
            margin=dict(b=100, t=120)  # More top margin for title and legend space
        )
        
        return fig
        
    else:
        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if len(data_relaxed) > 0:
            # Get unique quarters
            if tick_vals is not None:
                all_quarters = [pd.Timestamp(tick_val) for tick_val in tick_vals]
                quarter_periods = []
                seen = set()
                for ts in all_quarters:
                    quarter = ts.to_period('Q').start_time
                    if quarter not in seen:
                        quarter_periods.append(quarter)
                        seen.add(quarter)
                quarters = sorted(quarter_periods)
            else:
                quarters = sorted(data_relaxed['quarter_date'].unique())
            
            # Prepare data for stacked bars
            x = np.arange(len(quarters))
            width = 0.6
            
            management_values = []
            board_values = []
            
            for quarter in quarters:
                quarter_data = data_relaxed[data_relaxed['quarter_date'] == quarter]
                mgmt_data = quarter_data[quarter_data['Theme_Type'] == 'management']
                board_data = quarter_data[quarter_data['Theme_Type'] == 'board']
                
                mgmt_val = mgmt_data['Value'].iloc[0] if not mgmt_data.empty else 0
                board_val = board_data['Value'].iloc[0] if not board_data.empty else 0
                
                management_values.append(mgmt_val)
                board_values.append(board_val)
            
            # Create stacked bars
            bars1 = ax.bar(x, management_values, width, label='Management', color='#636EFA')
            bars2 = ax.bar(x, board_values, width, bottom=management_values, label='Board', color='#EF553B')
            
            # Customize chart
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_ylabel(y_label)
            if y_max is not None:
                ax.set_ylim(0, y_max)
            
            # Format x-axis labels
            if tick_vals is not None and tick_text is not None:
                quarter_labels = []
                for quarter in quarters:
                    quarter_period = quarter.to_period('Q')
                    label_found = False
                    for i, tick_val in enumerate(tick_vals):
                        tick_period = pd.Timestamp(tick_val).to_period('Q')
                        if tick_period == quarter_period and i < len(tick_text):
                            quarter_labels.append(tick_text[i])
                            label_found = True
                            break
                    if not label_found:
                        quarter_labels.append(str(quarter_period))
                ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
            else:
                quarter_labels = [str(q.to_period('Q')) for q in quarters]
                ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
            
            ax.grid(True, alpha=0.3, axis='y')
            
            if show_legend:
                ax.legend()
        
        plt.tight_layout()
        
        return fig


def plot_top_sources(df, person_name="Person", top_n=5, interactive=True):
    """
    Generate bar plot of top sources for person mentions.
    
    Parameters:
        df: DataFrame with 'SourceID' column
        person_name: Name of the person being monitored
        top_n: Number of top sources to display
        interactive: If True, create interactive Plotly plot. If False, create static matplotlib plot
        
    Returns:
        plotly.graph_objects.Figure or matplotlib.figure.Figure: Bar chart of top sources
    """
    if len(df) == 0:
        print("No data available for source analysis")
        return None
        
    top_sources = df['SourceID'].value_counts().head(top_n)
    
    if interactive:
        
        fig_sources = px.bar(
            x=top_sources.values,
            y=top_sources.index,
            orientation='h',
            title=f"Top {top_n} Sources - {person_name} Mentions (Relaxed Mode Search)",
            labels={'x': 'Number of Mentions', 'y': 'Source ID'}
        )
        fig_sources.update_layout(yaxis={'categoryorder':'total ascending'})
        #fig_sources.show()
        return fig_sources
    else:
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sources_reversed = top_sources[::-1]
        bars = ax.barh(range(len(sources_reversed)), sources_reversed.values)
        ax.set_yticks(range(len(sources_reversed)))
        ax.set_yticklabels(sources_reversed.index)
        ax.set_xlabel('Number of Mentions')
        ax.set_ylabel('Source ID')
        ax.set_title(f"Top {top_n} Sources - {person_name} Mentions")
        
        for i, (bar, value) in enumerate(zip(bars, sources_reversed.values)):
            ax.text(bar.get_width() + max(sources_reversed.values) * 0.01, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{value}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()

_intialization_sent = False 
def notebook_initialized():
    from importlib.metadata import version
    from bigdata_client import Bigdata
    from bigdata_client import tracking_services

    try:
        bigdata = Bigdata()
        global _intialization_sent
        if not _intialization_sent:
            trace = tracking_services.TraceEvent(event_name = "BigdataCookbookExecution", 
                       properties={"bigdataResearchToolsVersion": version("bigdata_research_tools"),
                                    "bigdataClientVersion": version("bigdata-client"),
                                   "cookbook_name": "BoardManagementMonitoring"
                                  })
            
            tracking_services.send_trace(bigdata_client = bigdata, trace = trace)
            _intialization_sent = True
    except Exception as e:
        pass
        
notebook_initialized()         