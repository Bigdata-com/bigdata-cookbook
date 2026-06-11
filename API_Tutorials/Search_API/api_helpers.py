"""
API Helper Functions for Bigdata.com Search API

This module provides utility functions to interact with the Bigdata.com Search API,
including search queries, grid parameter search, and visualizations.
"""

import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from itertools import product


def get_volume_dataframe(
    session,
    volume_endpoint: str,
    text: str,
    entities: List[str],
    start_date: str,
    end_date: str,
    entity_mode: str = "any_of",
    freshness_boost: float = 0
) -> Optional[pd.DataFrame]:
    """
    Fetch volume data from the Bigdata.com API and return as a DataFrame.
    Used for comparison with Search API results in optimization analysis.
    
    Args:
        session: requests.Session object with authentication configured
        volume_endpoint: Full URL of the volume API endpoint
        text: Search query text
        entities: List of entity IDs to filter by
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        entity_mode: How to apply entity filter ("any_of", "all_of", "none_of")
        freshness_boost: Freshness boost parameter (default 0)
    
    Returns:
        pd.DataFrame with columns ['date', 'documents', 'chunks'] or None if request fails
    """
    query = {
        "query": {
            "text": text,
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {
                    "start": f"{start_date}T00:00:00Z",
                    "end": f"{end_date}T23:59:59Z"
                },
                "entity": {entity_mode: entities}
            },
            "ranking_params": {
                "freshness_boost": freshness_boost
            }
        }
    }
    
    response = session.post(volume_endpoint, json=query)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    volume_list = data.get('results', {}).get('volume', [])
    
    if not volume_list:
        return pd.DataFrame(columns=['date', 'documents', 'chunks'])
    
    df = pd.DataFrame(volume_list)
    df = df.rename(columns={'day': 'date'})
    
    # Ensure proper column order
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    return df


def run_search(
    session,
    search_endpoint: str,
    text: str,
    entities: List[str],
    start_date: str,
    end_date: str,
    entity_mode: str = "any_of",
    freshness_boost: float = 5,
    source_boost: Optional[float] = None,
    max_chunks: int = 120
) -> Optional[Dict[str, Any]]:
    """
    Run a single search query against the Bigdata.com Search API.
    
    Args:
        session: requests.Session object with authentication configured
        search_endpoint: Full URL of the search API endpoint
        text: Search query text
        entities: List of entity IDs to filter by
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        entity_mode: How to apply entity filter ("any_of", "all_of", "none_of")
        freshness_boost: Freshness boost parameter (default 5)
        source_boost: Source boost parameter (optional)
        max_chunks: Maximum number of chunks to return (default 120)
    
    Returns:
        dict with API response or None if request fails
    """
    ranking_params = {"freshness_boost": freshness_boost}
    if source_boost is not None:
        ranking_params["source_boost"] = source_boost
    
    query = {
        "query": {
            "text": text,
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {
                    "start": f"{start_date}T00:00:00Z",
                    "end": f"{end_date}T23:59:59Z"
                },
                "entity": {entity_mode: entities}
            },
            "ranking_params": ranking_params,
            "max_chunks": max_chunks
        }
    }
    
    response = session.post(search_endpoint, json=query)
    return response.json() if response.status_code == 200 else None


def grid_parameter_search(
    session,
    search_endpoint: str,
    text: str,
    entities: List[str],
    start_date: str,
    end_date: str,
    max_chunks_values: List[int],
    freshness_values: Optional[List[float]] = None,
    source_boost_values: Optional[List[float]] = None,
    fixed_freshness_boost: float = 0,
    fixed_source_boost: Optional[float] = None,
    entity_mode: str = "any_of"
) -> Tuple[pd.DataFrame, Dict[Tuple[float, int], Any]]:
    """
    Run a grid search over ranking parameters and max_chunks.
    
    Can vary either freshness_boost OR source_boost (not both at once).
    
    Args:
        session: requests.Session object with authentication configured
        search_endpoint: Full URL of the search API endpoint
        text: Search query text
        entities: List of entity IDs to filter by
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        max_chunks_values: List of max_chunks values to test
        freshness_values: List of freshness_boost values to test (optional)
        source_boost_values: List of source_boost values to test (optional, alternative to freshness_values)
        fixed_freshness_boost: Fixed freshness_boost when varying source_boost (default 0)
        fixed_source_boost: Fixed source_boost when varying freshness (default None)
        entity_mode: How to apply entity filter ("any_of", "all_of", "none_of")
    
    Returns:
        Tuple of:
        - pd.DataFrame with columns for the varied parameter, max_chunks, docs, chunks
        - dict with raw results keyed by (param_value, max_chunks) tuples
    
    Examples:
        # Vary freshness_boost x max_chunks
        df, results = grid_parameter_search(..., freshness_values=[0, 1, 10], max_chunks_values=[100, 500])
        
        # Vary source_boost x max_chunks (with fixed freshness_boost=0)
        df, results = grid_parameter_search(..., source_boost_values=[0, 1, 5, 10], max_chunks_values=[500])
    """
    results_dict = {}
    rows = []
    
    # Determine which parameter to vary
    if source_boost_values is not None:
        # Vary source_boost
        param_name = 'source_boost'
        param_values = source_boost_values
        use_source_boost = True
    elif freshness_values is not None:
        # Vary freshness_boost
        param_name = 'freshness_boost'
        param_values = freshness_values
        use_source_boost = False
    else:
        raise ValueError("Must provide either freshness_values or source_boost_values")
    
    for param_val, chunks in product(param_values, max_chunks_values):
        key = (param_val, chunks)
        
        if use_source_boost:
            data = run_search(
                session=session,
                search_endpoint=search_endpoint,
                text=text,
                entities=entities,
                start_date=start_date,
                end_date=end_date,
                entity_mode=entity_mode,
                freshness_boost=fixed_freshness_boost,
                source_boost=param_val,
                max_chunks=chunks
            )
        else:
            data = run_search(
                session=session,
                search_endpoint=search_endpoint,
                text=text,
                entities=entities,
                start_date=start_date,
                end_date=end_date,
                entity_mode=entity_mode,
                freshness_boost=param_val,
                source_boost=fixed_source_boost,
                max_chunks=chunks
            )
        
        results_dict[key] = data
        
        if data:
            docs = data.get('results', [])
            doc_count = len(docs)
            total_chunks = sum(len(doc.get('chunks', [])) for doc in docs)
        else:
            doc_count = 0
            total_chunks = 0
        
        rows.append({
            param_name: param_val,
            'max_chunks': chunks,
            'docs': doc_count,
            'chunks': total_chunks
        })
    
    df = pd.DataFrame(rows)
    return df, results_dict


def _date_to_week(date_str: str) -> str:
    """Convert a date string to its week start (Monday)."""
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    week_start = dt - timedelta(days=dt.weekday())
    return week_start.strftime('%Y-%m-%d')


def plot_freshness_comparison(
    results: Dict[Tuple[float, int], Any],
    freshness_values: List[float],
    max_chunks_values: List[int],
    text: str,
    start_date: str,
    end_date: str,
    volume_results: Optional[List[dict]] = None,
    entity_id: str = "",
    group_by_week: bool = False
):
    """
    Create a plotly figure comparing results across freshness_boost values.
    
    Each subplot represents a different freshness_boost value, with lines
    showing document counts over time for each max_chunks value.
    
    Args:
        results: dict with raw results keyed by (freshness_boost, max_chunks) tuples
        freshness_values: List of freshness_boost values used
        max_chunks_values: List of max_chunks values used
        text: Search query text (for title)
        start_date: Start date (for title)
        end_date: End date (for title)
        volume_results: Optional list of volume data dicts with 'date' and 'documents' keys
        entity_id: Entity ID string (for title)
        group_by_week: If True, aggregate results by week (default: False)
    
    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from collections import Counter, defaultdict
    
    # Collect all dates from results
    all_dates_raw = sorted({
        doc['timestamp'][:10] 
        for (f, c), data in results.items() 
        if data 
        for doc in data.get('results', [])
    })
    
    # Add volume dates if available
    if volume_results:
        volume_dates = {v['date'] if isinstance(v['date'], str) else v['date'].strftime('%Y-%m-%d') for v in volume_results}
        all_dates_raw = sorted(set(all_dates_raw) | volume_dates)
    
    # Group by week if requested
    if group_by_week:
        all_periods = sorted(set(_date_to_week(d) for d in all_dates_raw))
        period_label = "Week"
    else:
        all_periods = all_dates_raw
        period_label = "Date"
    
    # Colors for max_chunks values
    chunks_colors = px.colors.qualitative.Plotly[:len(max_chunks_values)]
    
    # Create subplots: one row per FRESHNESS value
    n_rows = len(freshness_values)
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"freshness_boost = {f}" for f in freshness_values],
        specs=[[{"secondary_y": True}] for _ in range(n_rows)]
    )
    
    # Helper function to aggregate counts
    def aggregate_counts(counts_by_date):
        if group_by_week:
            weekly_counts = defaultdict(int)
            for date, count in counts_by_date.items():
                week = _date_to_week(date)
                weekly_counts[week] += count
            return weekly_counts
        return counts_by_date
    
    # First pass: collect all y values to calculate max_y
    all_y_values = []
    for row_idx, freshness in enumerate(freshness_values, start=1):
        for color_idx, chunks in enumerate(max_chunks_values):
            key = (freshness, chunks)
            data = results.get(key)
            if not data or not data.get('results'):
                continue
            counts = Counter(doc['timestamp'][:10] for doc in data['results'])
            aggregated = aggregate_counts(counts)
            y_vals = [aggregated.get(p, 0) for p in all_periods]
            all_y_values.extend(y_vals)
    
    # Calculate max y value across all subplots with 10% extra margin
    max_y = max(all_y_values) * 1.1 if all_y_values else 1
    
    # Plot each freshness as a separate subplot
    for row_idx, freshness in enumerate(freshness_values, start=1):
        
        # Add traces for each max_chunks value
        for color_idx, chunks in enumerate(max_chunks_values):
            key = (freshness, chunks)
            data = results.get(key)
            
            if not data or not data.get('results'):
                continue
                
            counts = Counter(doc['timestamp'][:10] for doc in data['results'])
            aggregated = aggregate_counts(counts)
            y_vals = [aggregated.get(p, 0) for p in all_periods]
            
            fig.add_trace(go.Scatter(
                x=all_periods, 
                y=y_vals,
                mode='lines+markers',
                name=f'chunks={chunks}',
                line=dict(color=chunks_colors[color_idx], width=2),
                marker=dict(size=6),
                legendgroup=f'chunks_{chunks}',
                showlegend=(row_idx == 1)
            ), row=row_idx, col=1, secondary_y=False)
        
        # Add Volume API bars (secondary y-axis)
        if volume_results:
            volume_by_date = {
                (v['date'] if isinstance(v['date'], str) else v['date'].strftime('%Y-%m-%d')): v 
                for v in volume_results
            }
            if group_by_week:
                weekly_vol = defaultdict(int)
                for date, v in volume_by_date.items():
                    week = _date_to_week(date)
                    weekly_vol[week] += v.get('documents', 0)
                vol_docs = [weekly_vol.get(p, 0) for p in all_periods]
            else:
                vol_docs = [volume_by_date.get(d, {}).get('documents', 0) for d in all_periods]
            
            fig.add_trace(go.Bar(
                x=all_periods, 
                y=vol_docs,
                name='Volume API',
                marker=dict(color='rgba(128, 0, 128, 0.3)'),
                opacity=0.4,
                legendgroup='volume',
                showlegend=(row_idx == 1)
            ), row=row_idx, col=1, secondary_y=True)
    
    # Update layout
    entity_str = f" | Entity: {entity_id}" if entity_id else ""
    grouping_str = " (Weekly)" if group_by_week else ""
    fig.update_layout(
        title=f'max_chunks Comparison by freshness_boost{grouping_str}<br><sup>Query: "{text}"{entity_str} | {start_date} to {end_date}</sup>',
        hovermode='x unified',
        height=300 * n_rows,
        legend=dict(
            orientation="v", 
            yanchor="top", 
            y=1, 
            xanchor="left", 
            x=1.02
        )
    )
    
    # Update axes labels with same max y value for all subplots
    for row_idx in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Docs", row=row_idx, col=1, secondary_y=False, range=[0, max_y])
        fig.update_yaxes(title_text="Volume", row=row_idx, col=1, secondary_y=True)
    
    fig.update_xaxes(title_text=period_label, row=n_rows, col=1)
    
    return fig


def plot_source_distribution(
    results_source: Dict[float, Any],
    source_boost_values: List[float],
    top_n_sources: int = 50,
    top_n_per_subplot: int = 15
) -> Tuple[Any, Any]:
    """
    Create bar and line charts showing source distribution by source_boost.
    
    Args:
        results_source: dict with results keyed by source_boost value
        source_boost_values: List of source_boost values used
        top_n_sources: Number of top sources for line chart (default 50)
        top_n_per_subplot: Number of top sources per subplot in bar chart (default 15)
    
    Returns:
        Tuple of (fig_bar, fig_line) plotly figures
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from collections import Counter
    
    # Get TOP N sources by total count across all queries
    source_totals = Counter()
    for data in results_source.values():
        if data and data.get('results'):
            source_totals.update(doc['source']['name'] for doc in data['results'])
    top_sources = [s for s, _ in source_totals.most_common(top_n_sources)]
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    # === BAR CHART: Subplots 2x2 (one per source_boost) ===
    n_plots = len(source_boost_values)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig_bar = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'source_boost={sb}' for sb in source_boost_values],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # First pass: collect all y values to calculate max_y for bar chart
    all_y_values_bar = []
    for sb in source_boost_values:
        data = results_source.get(sb)
        if data and data.get('results'):
            counts = Counter(doc['source']['name'] for doc in data['results'])
            top_for_query = [s for s, _ in counts.most_common(top_n_per_subplot)]
            y_vals = [counts.get(s, 0) for s in top_for_query]
            all_y_values_bar.extend(y_vals)
    
    max_y_bar = max(all_y_values_bar) * 1.1 if all_y_values_bar else 1
    
    # Second pass: add traces
    for idx, sb in enumerate(source_boost_values):
        row, col = (idx // n_cols) + 1, (idx % n_cols) + 1
        data = results_source.get(sb)
        if data and data.get('results'):
            counts = Counter(doc['source']['name'] for doc in data['results'])
            top_for_query = [s for s, _ in counts.most_common(top_n_per_subplot)]
            y_vals = [counts.get(s, 0) for s in top_for_query]
            fig_bar.add_trace(go.Bar(
                x=top_for_query,
                y=y_vals,
                name=f'sb={sb}',
                marker_color=colors[idx % len(colors)],
                showlegend=True
            ), row=row, col=col)
    
    fig_bar.update_layout(
        title='Source Distribution by source_boost (freshness_boost=0)',
        height=350 * n_rows,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_bar.update_xaxes(tickangle=45, tickfont=dict(size=8))
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig_bar.update_yaxes(title_text="Documents", row=r, col=c, range=[0, max_y_bar])
    
    # === LINE CHART: X-axis = Top N sources, one line per source_boost ===
    fig_line = go.Figure()
    
    for idx, sb in enumerate(source_boost_values):
        data = results_source.get(sb)
        if data and data.get('results'):
            counts = Counter(doc['source']['name'] for doc in data['results'])
            y_vals = [counts.get(s, 0) for s in top_sources]
            fig_line.add_trace(go.Scatter(
                x=top_sources,
                y=y_vals,
                mode='lines+markers',
                name=f'source_boost={sb}',
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=5)
            ))
    
    fig_line.update_layout(
        title=f'Source Counts - Top {top_n_sources} sources (freshness_boost=0)',
        xaxis_title='Source',
        yaxis_title='Documents',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_line.update_xaxes(tickangle=45, tickfont=dict(size=7))
    
    return fig_bar, fig_line


def plot_source_rank_distribution(
    results_source: Dict[float, Any],
    source_boost_values: List[float]
) -> Tuple[Any, Any]:
    """
    Create bar and line charts showing source rank distribution by source_boost.
    
    Args:
        results_source: dict with results keyed by source_boost value
        source_boost_values: List of source_boost values used
    
    Returns:
        Tuple of (fig_bar, fig_line) plotly figures
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from collections import Counter
    
    # Get all unique ranks
    all_ranks = sorted(set(
        doc['source'].get('rank', 'UNKNOWN') 
        for data in results_source.values() if data 
        for doc in data.get('results', [])
    ))
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    # === BAR CHART: Source Rank - Subplots ===
    n_plots = len(source_boost_values)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig_rank_bar = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'source_boost={sb}' for sb in source_boost_values],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # First pass: collect all y values to calculate max_y
    all_y_values_rank = []
    for sb in source_boost_values:
        data = results_source.get(sb)
        if data and data.get('results'):
            ranks = Counter(doc['source'].get('rank', 'UNKNOWN') for doc in data['results'])
            y_vals = [ranks.get(r, 0) for r in all_ranks]
            all_y_values_rank.extend(y_vals)
    
    max_y_rank = max(all_y_values_rank) * 1.1 if all_y_values_rank else 1
    
    # Second pass: add traces
    for idx, sb in enumerate(source_boost_values):
        row, col = (idx // n_cols) + 1, (idx % n_cols) + 1
        data = results_source.get(sb)
        if data and data.get('results'):
            ranks = Counter(doc['source'].get('rank', 'UNKNOWN') for doc in data['results'])
            y_vals = [ranks.get(r, 0) for r in all_ranks]
            fig_rank_bar.add_trace(go.Bar(
                x=all_ranks,
                y=y_vals,
                name=f'sb={sb}',
                marker_color=colors[idx % len(colors)],
                showlegend=True
            ), row=row, col=col)
    
    fig_rank_bar.update_layout(
        title='Source RANK Distribution by source_boost (freshness_boost=0)',
        height=300 * n_rows,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig_rank_bar.update_yaxes(title_text="Documents", row=r, col=c, range=[0, max_y_rank])
    
    # === LINE CHART: Source Rank ===
    fig_rank_line = go.Figure()
    
    for idx, sb in enumerate(source_boost_values):
        data = results_source.get(sb)
        if data and data.get('results'):
            ranks = Counter(doc['source'].get('rank', 'UNKNOWN') for doc in data['results'])
            y_vals = [ranks.get(r, 0) for r in all_ranks]
            fig_rank_line.add_trace(go.Scatter(
                x=all_ranks,
                y=y_vals,
                mode='lines+markers',
                name=f'source_boost={sb}',
                line=dict(color=colors[idx % len(colors)], width=3),
                marker=dict(size=10)
            ))
    
    fig_rank_line.update_layout(
        title='Source RANK Counts (freshness_boost=0)',
        xaxis_title='Source Rank',
        yaxis_title='Documents',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig_rank_bar, fig_rank_line


def get_source_rank_summary(
    results_source: Dict[float, Any],
    source_boost_values: List[float]
) -> pd.DataFrame:
    """
    Get a summary DataFrame of source rank distribution.
    
    Args:
        results_source: dict with results keyed by source_boost value
        source_boost_values: List of source_boost values used
    
    Returns:
        pd.DataFrame with source_boost as index and ranks as columns
    """
    from collections import Counter
    
    rows = []
    for sb in source_boost_values:
        data = results_source.get(sb)
        if data and data.get('results'):
            ranks = Counter(doc['source'].get('rank', 'UNKNOWN') for doc in data['results'])
            row = {'source_boost': sb}
            row.update(dict(sorted(ranks.items())))
            rows.append(row)
    
    return pd.DataFrame(rows).set_index('source_boost').fillna(0).astype(int)


def plot_chunks_vs_max_chunks(
    results: Dict[Tuple[float, int], Any],
    max_chunks_values: List[int],
    freshness_boost: float = 0,
    text: str = "",
    start_date: str = "",
    end_date: str = ""
):
    """
    Create a matplotlib plot showing chunks retrieved vs max_chunks parameter.
    
    Args:
        results: dict with results keyed by (param_value, max_chunks) tuples
        max_chunks_values: List of max_chunks values used
        freshness_boost: The freshness_boost value to filter results (default 0)
        text: Search query text (for title)
        start_date: Start date (for title)
        end_date: End date (for title)
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Extract data for the specified freshness_boost
    x = []
    y = []
    for chunks in max_chunks_values:
        key = (freshness_boost, chunks)
        data = results.get(key)
        if data:
            docs = data.get('results', [])
            total_chunks = sum(len(doc.get('chunks', [])) for doc in docs)
            x.append(chunks)
            y.append(total_chunks)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Actual chunks retrieved
    ax.plot(x, y, 'b-o', linewidth=2, markersize=8, label='Chunks Retrieved')
    
    # Linear reference line: chunks_retrieved = max_chunks (ideal case)
    ax.plot(x, x, 'r--', linewidth=1.5, alpha=0.7, label='Linear Reference (y = max_chunks)')
    
    ax.set_xlabel('max_chunks (requested)', fontsize=12)
    ax.set_ylabel('Chunks Retrieved (actual)', fontsize=12)
    title = f'Chunks Retrieved vs max_chunks (freshness_boost={freshness_boost})'
    if text:
        title += f'\n(Query: "{text}" | {start_date} to {end_date})'
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations for first and last points
    if y:
        ax.annotate(f'{y[0]}', (x[0], y[0]), textcoords="offset points", xytext=(0,10), ha='center')
        ax.annotate(f'{y[-1]}', (x[-1], y[-1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    return fig
