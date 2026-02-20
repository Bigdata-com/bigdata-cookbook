"""
API Helper Functions for Bigdata.com Volume API

This module provides utility functions to interact with the Bigdata.com Volume API,
including volume data retrieval and visualization.
"""

import pandas as pd
from typing import List, Optional, Dict, Any


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


def get_volume_totals(
    session,
    volume_endpoint: str,
    text: str,
    entities: List[str],
    start_date: str,
    end_date: str,
    entity_mode: str = "any_of",
    freshness_boost: float = 0
) -> Optional[dict]:
    """
    Fetch total volume data from the Bigdata.com API.
    
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
        dict with keys 'documents' and 'chunks' containing totals, or None if request fails
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
    return data.get('results', {}).get('total', None)


def plot_volume_evolution(
    volume_data: Dict[str, Any],
    text: str,
    start_date: str,
    end_date: str
):
    """
    Create a multi-panel plot showing volume evolution over time.
    
    Args:
        volume_data: Volume API response data
        text: Query text (for title)
        start_date: Start date
        end_date: End date
    
    Returns:
        plotly Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    
    if not volume_data or not volume_data.get("results", {}).get("volume"):
        print("❌ No volume data to plot")
        return None
    
    volume_list = volume_data["results"]["volume"]
    
    # Convert to DataFrame
    df = pd.DataFrame(volume_list)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate 7-day rolling averages
    df['docs_ma7'] = df['documents'].rolling(window=7, min_periods=1).mean()
    df['chunks_ma7'] = df['chunks'].rolling(window=7, min_periods=1).mean()
    df['sentiment_ma7'] = df['sentiment'].rolling(window=7, min_periods=1).mean()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=['Documents per Day', 'Chunks per Day', 'Sentiment per Day'],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Colors
    color_docs = '#2E86AB'
    color_chunks = '#A23B72'
    color_sentiment = '#F18F01'
    
    # Plot 1: Documents
    fig.add_trace(go.Bar(
        x=df['date'], y=df['documents'],
        name='Daily Docs',
        marker_color=color_docs,
        opacity=0.4
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['docs_ma7'],
        name='7-day MA',
        line=dict(color=color_docs, width=2.5),
        mode='lines'
    ), row=1, col=1)
    
    # Plot 2: Chunks
    fig.add_trace(go.Bar(
        x=df['date'], y=df['chunks'],
        name='Daily Chunks',
        marker_color=color_chunks,
        opacity=0.4,
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['chunks_ma7'],
        name='7-day MA',
        line=dict(color=color_chunks, width=2.5),
        mode='lines',
        showlegend=False
    ), row=2, col=1)
    
    # Plot 3: Sentiment
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['sentiment'],
        name='Daily Sentiment',
        line=dict(color=color_sentiment, width=1),
        mode='lines+markers',
        marker=dict(size=3),
        opacity=0.5
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['sentiment_ma7'],
        name='7-day MA',
        line=dict(color=color_sentiment, width=2.5),
        mode='lines',
        showlegend=False
    ), row=3, col=1)
    
    # Add zero line for sentiment
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'📊 Volume Evolution: "{text}"<br><sup>{start_date} to {end_date}</sup>',
        height=800,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes
    fig.update_yaxes(title_text="Documents", row=1, col=1)
    fig.update_yaxes(title_text="Chunks", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Documents: min={df['documents'].min():,}, max={df['documents'].max():,}, avg={df['documents'].mean():,.0f}")
    print(f"   Chunks: min={df['chunks'].min():,}, max={df['chunks'].max():,}, avg={df['chunks'].mean():,.0f}")
    print(f"   Sentiment: min={df['sentiment'].min():.3f}, max={df['sentiment'].max():.3f}, avg={df['sentiment'].mean():.3f}")
    
    return fig
