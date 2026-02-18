"""
Helper functions for processing BigData search results.

This module contains functions for:
- Loading the entity universe from a CSV file
- Exploding search results by entity
- Filtering entities based on a defined universe
- Creating sentiment/volume indicators
- Visualizing sentiment and volume trends
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Optional

# Optional imports for visualization
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_universe_entities(csv_path: str) -> tuple[set, dict]:
    """
    Load entity IDs and names from the universe CSV.
    
    Args:
        csv_path: Path to CSV file with 'id' and 'name' columns
        
    Returns:
        tuple containing:
            - set of valid entity IDs (for O(1) lookup)
            - dict mapping entity_id -> name
            
    Example:
        >>> entity_ids, id_to_name = load_universe_entities("us_top_id_with_names.csv")
        >>> print(f"Loaded {len(entity_ids)} entities")
        >>> print(id_to_name.get("00F26A"))  # "Atai Beckley Inc."
    """
    df = pd.read_csv(csv_path)
    
    # Create set for fast O(1) lookup
    entity_ids = set(df['id'].dropna().astype(str))
    
    # Create dictionary id -> name to enrich results
    id_to_name = dict(zip(df['id'].astype(str), df['name']))
    
    return entity_ids, id_to_name


def explode_by_entity(
    results: list[dict],
    valid_entity_ids: Optional[set] = None,
    id_to_name: Optional[dict] = None
) -> list[dict]:
    """
    Explode search results by entity, creating one row for each 
    combination of (entity_id, chunk, document).
    
    Each entity present in a chunk's detections generates a separate row.
    The same entity can appear in multiple chunks and documents.
    
    Args:
        results: List of documents from API search, each document 
                 contains chunks with detections
        valid_entity_ids: (Optional) Set of entity IDs to keep. 
                         If None, keeps all entities
        id_to_name: (Optional) Dictionary entity_id -> company name.
                   If provided, adds the 'entity_name' column
                   
    Returns:
        List of dictionaries, each with:
            - entity_name (if id_to_name provided)
            - entity_id
            - entity_type ('entity' or 'topic')
            - doc_id
            - doc_headline
            - doc_timestamp
            - source_name
            - chunk_cnum
            - chunk_text
            - chunk_relevance
            - chunk_sentiment
            - entity_start (position in text)
            - entity_end (position in text)
            
    Example:
        >>> # Without filter - keeps all entities
        >>> rows = explode_by_entity(results)
        
        >>> # With filter - keeps only universe entities
        >>> entity_ids, id_to_name = load_universe_entities("universe.csv")
        >>> rows = explode_by_entity(results, entity_ids, id_to_name)
    """
    exploded_rows = []
    
    for doc in results:
        doc_id = doc.get('id')
        doc_headline = doc.get('headline')
        doc_timestamp = doc.get('timestamp')
        source_name = doc.get('source', {}).get('name')
        
        for chunk in doc.get('chunks', []):
            chunk_cnum = chunk.get('cnum')
            chunk_text = chunk.get('text')
            chunk_relevance = chunk.get('relevance')
            chunk_sentiment = chunk.get('sentiment')
            
            for detection in chunk.get('detections', []):
                entity_id = detection.get('id')
                
                # Filter if required - immediate skip if not in universe
                if valid_entity_ids is not None and entity_id not in valid_entity_ids:
                    continue
                
                # Build the row
                row = {
                    'entity_id': entity_id,
                    'entity_type': detection.get('type'),
                    'doc_id': doc_id,
                    'doc_headline': doc_headline,
                    'doc_timestamp': doc_timestamp,
                    'source_name': source_name,
                    'chunk_cnum': chunk_cnum,
                    'chunk_text': chunk_text,
                    'chunk_relevance': chunk_relevance,
                    'chunk_sentiment': chunk_sentiment,
                    'entity_start': detection.get('start'),
                    'entity_end': detection.get('end'),
                }
                
                # Add name if available
                if id_to_name is not None:
                    row['entity_name'] = id_to_name.get(entity_id, None)
                
                exploded_rows.append(row)
    
    return exploded_rows


def explode_to_dataframe(
    data: pd.DataFrame | list[dict],
    universe_csv: Optional[str] = None,
    entity_ids_column: str = "entity_ids_companies",
    sort_by: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Explode by entity: one row per (original row, entity_id) for entity IDs in the universe.

    - If `data` is a **DataFrame**: expects column `entity_ids_column` (default
      "entity_ids_companies") with list of entity IDs per row. Requires `universe_csv`.
      Keeps only IDs present in the universe; creates one row per such ID, preserving
      all original columns and adding `entity_id` and `entity_name`.
    - If `data` is a **list[dict]** (raw API results): behaves as before: explodes
      by entity from documents/chunks/detections; `universe_csv` optional.

    Args:
        data: Either a DataFrame (with entity_ids_companies-like column) or list of
              documents from API search.
        universe_csv: Path to universe CSV (required when data is DataFrame).
        entity_ids_column: Name of column containing list of entity IDs (used when
                           data is DataFrame).
        sort_by: Optional list of columns to sort by.
                 Default: ['entity_id', 'doc_timestamp'] or ['entity_id', 'date'].

    Returns:
        DataFrame with one row per entity (per chunk when data is DataFrame).

    Example (DataFrame input):
        >>> df_chunks = convert_to_dataframe(results)  # exploded by chunk
        >>> df_chunks = map_create_only_companies_column(df_chunks, ...)
        >>> df_exploded = explode_to_dataframe(df_chunks, universe_csv="universe.csv")

    Example (list input, backward compatible):
        >>> df = explode_to_dataframe(results, universe_csv="us_top_id_with_names.csv")
    """
    if isinstance(data, pd.DataFrame):
        if universe_csv is None:
            raise ValueError("explode_to_dataframe(df, ...) requires universe_csv")
        if entity_ids_column not in data.columns:
            raise ValueError(f"DataFrame must have column '{entity_ids_column}'")
        entity_ids, id_to_name = load_universe_entities(universe_csv)
        rows = []
        for idx, row in data.iterrows():
            ids_in_row = row.get(entity_ids_column) or []
            if not isinstance(ids_in_row, (list, set)):
                ids_in_row = [ids_in_row] if pd.notna(ids_in_row) else []
            # Unique IDs in this row that are also in the universe
            unique_in_universe = sorted(set(str(e) for e in ids_in_row) & entity_ids)
            for eid in unique_in_universe:
                new_row = row.to_dict()
                new_row["entity_id"] = eid
                new_row["entity_name"] = id_to_name.get(eid)
                rows.append(new_row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "entity_name" in df.columns:
            ordered = ["entity_name", "entity_id"] + [
                c for c in df.columns if c not in ("entity_name", "entity_id")
            ]
            df = df[ordered]
        if sort_by is None:
            sort_by = ["entity_id", "date"] if "date" in df.columns else ["entity_id", "doc_timestamp"]
        sort_cols = [c for c in sort_by if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        return df

    # Original behavior: data is list[dict] (raw API results)
    results = data
    entity_ids = None
    id_to_name = None
    if universe_csv is not None:
        entity_ids, id_to_name = load_universe_entities(universe_csv)
    rows = explode_by_entity(results, entity_ids, id_to_name)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "entity_name" in df.columns:
        ordered_columns = ["entity_name", "entity_id"] + [
            col for col in df.columns if col not in ("entity_name", "entity_id")
        ]
        df = df[ordered_columns]
    if sort_by is None:
        sort_by = ["entity_id", "doc_timestamp"]
    sort_columns = [col for col in sort_by if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)
    return df


# Placeholders for masking (same as bigdata_research_tools.prompts.labeler)
TARGET_ENTITY_PLACEHOLDER = "Target_Company"
OTHER_ENTITY_PLACEHOLDER = "Other_Company"


def _mask_chunk_text(
    text: str,
    companies_detection: list[dict],
    target_entity_id: Optional[str] = None,
) -> tuple[str, list[tuple[int, str]]]:
    """
    Mask company spans in text: target entity -> TARGET placeholder,
    other entities -> OTHER_1, OTHER_2, ... (by first occurrence of each id).
    Process from end to start so indices stay valid.

    Returns:
        (masked_text, other_entities_map) with other_entities_map = [(number, entity_id), ...].
    """
    if not text or not companies_detection:
        return text, []
    entities = sorted(companies_detection, key=lambda x: x["start"], reverse=True)
    masked_text = text
    entity_counter: dict[str, int] = {}
    other_entities_map: list[tuple[int, str]] = []
    i = 1
    for entity in entities:
        start, end = entity["start"], entity["end"]
        eid = entity["id"]
        if target_entity_id is not None and str(eid) == str(target_entity_id):
            masked_text = f"{masked_text[:start]}{TARGET_ENTITY_PLACEHOLDER}{masked_text[end:]}"
        else:
            if eid not in entity_counter:
                entity_counter[eid] = i
                other_entities_map.append((i, eid))
                i += 1
            mask = f"{OTHER_ENTITY_PLACEHOLDER}_{entity_counter[eid]}"
            masked_text = f"{masked_text[:start]}{mask}{masked_text[end:]}"
    return masked_text, other_entities_map


def mask_companies_in_df(
    df: pd.DataFrame,
    text_column: str = "chunk_text",
    detection_column: str = "companies_detection",
    target_entity_id_column: str = "entity_id",
) -> pd.DataFrame:
    """
    Add masked_text and other_entities_map to the DataFrame.

    Designed for df_exploded_by_entity: target = entity_id, others = companies
    in companies_detection. Text comes from chunk_text.

    Args:
        df: DataFrame with chunk_text, companies_detection, and entity_id (target).
        text_column: Column containing the text to mask (default "chunk_text").
        detection_column: Column with list of {id, start, end, type} (default "companies_detection").
        target_entity_id_column: Column with the target entity id per row (default "entity_id").

    Returns:
        Copy of df with added columns: masked_text, other_entities_map.
    """
    df_out = df.copy()
    df_out["masked_text"] = None
    df_out["other_entities_map"] = None
    df_out["masked_text"] = df_out["masked_text"].astype("object")
    df_out["other_entities_map"] = df_out["other_entities_map"].astype("object")
    for idx, row in df_out.iterrows():
        text = row[text_column]
        detections = row[detection_column] if isinstance(row[detection_column], list) else []
        target_id = row.get(target_entity_id_column) if target_entity_id_column in row.index else None
        masked, other_map = _mask_chunk_text(text, detections, target_entity_id=target_id)
        df_out.at[idx, "masked_text"] = masked
        df_out.at[idx, "other_entities_map"] = other_map if other_map else None
    return df_out


def entity_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregate statistics for each entity in the exploded DataFrame.
    
    Args:
        df: DataFrame obtained from explode_to_dataframe()
        
    Returns:
        DataFrame with statistics per entity:
            - entity_name (if present)
            - entity_id
            - n_documents: number of unique documents
            - n_occurrences: total number of occurrences (chunk-detection)
            - avg_sentiment: weighted average sentiment
            - avg_relevance: average relevance
            
    Example:
        >>> df = explode_to_dataframe(results, universe_csv="universe.csv")
        >>> stats = entity_statistics(df)
        >>> print(stats.sort_values('n_occurrences', ascending=False).head(10))
    """
    # Columns for groupby
    group_cols = ['entity_id']
    if 'entity_name' in df.columns:
        group_cols = ['entity_name', 'entity_id']
    
    stats = df.groupby(group_cols).agg(
        n_documents=('doc_id', 'nunique'),
        n_occurrences=('doc_id', 'count'),
        avg_sentiment=('chunk_sentiment', 'mean'),
        avg_relevance=('chunk_relevance', 'mean'),
    ).reset_index()
    
    # Sort by occurrences descending
    stats = stats.sort_values('n_occurrences', ascending=False)
    
    return stats


def prepare_sentiment_dataframe(
    df_exploded: pd.DataFrame,
    date_col: str = 'doc_timestamp',
    entity_col: str = 'entity_name',
    sentiment_col: str = 'chunk_sentiment'
) -> pd.DataFrame:
    """
    Prepare exploded DataFrame for sentiment analysis.
    
    Converts the exploded DataFrame to the format required by 
    create_full_grid_indicators (from sentiment_analysis module).
    
    Each row represents one chunk occurrence, and a unique ID is assigned
    to ensure volume counts occurrences (not unique documents).
    
    Args:
        df_exploded: DataFrame from explode_to_dataframe()
        date_col: Column name containing timestamp
        entity_col: Column name containing entity identifier
        sentiment_col: Column name containing sentiment value
        
    Returns:
        DataFrame with columns:
            - Date: date only (no time)
            - Entity: entity name/id
            - Bigdata Sentiment: sentiment value
            - Document ID: unique ID per row (for counting occurrences)
            
    Example:
        >>> df_exploded = explode_to_dataframe(results, universe_csv="universe.csv")
        >>> df_sentiment = prepare_sentiment_dataframe(df_exploded)
    """
    df_sentiment = df_exploded.rename(columns={
        date_col: 'Date',
        entity_col: 'Entity',
        sentiment_col: 'Bigdata Sentiment',
    })[['Date', 'Entity', 'Bigdata Sentiment']].copy()
    
    # Create unique ID for each occurrence (row) so that volume = chunk count
    df_sentiment['Document ID'] = range(len(df_sentiment))
    
    # Convert to date only (no time) for proper aggregation
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date']).dt.date
    
    return df_sentiment


def get_top_entities_by_volume(
    df_exploded: pd.DataFrame,
    n: int = 5,
    entity_col: str = 'entity_name'
) -> list:
    """
    Get top N entities by volume (number of chunk occurrences).
    
    Args:
        df_exploded: DataFrame from explode_to_dataframe()
        n: Number of top entities to return
        entity_col: Column name containing entity identifier
        
    Returns:
        List of entity names/ids sorted by volume (descending)
        
    Example:
        >>> top_5 = get_top_entities_by_volume(df_exploded, n=5)
        >>> print(top_5)
        ['Wipro Ltd.', 'Apple Inc.', "Moody's Corp.", ...]
    """
    return df_exploded.groupby(entity_col).size().nlargest(n).index.tolist()


def display_sentiment_volume(
    df: pd.DataFrame,
    entity: str,
    sentiment_col: str = 'Sent_Rolling_30Days_Normalized',
    volume_type: str = "rolling",
    show_gauge: bool = True
):
    """
    Display sentiment and volume charts for a single entity.
    
    Creates a gauge chart showing mean sentiment and a time series plot
    showing sentiment trend and volume over time.
    
    Args:
        df: DataFrame from create_full_grid_indicators()
        entity: Entity name to display
        sentiment_col: Column to use for sentiment (default: normalized 30-day rolling)
        volume_type: "daily" for daily volume, "rolling" for 30-day rolling volume
        show_gauge: Whether to display the gauge chart (requires plotly)
        
    Example:
        >>> from src.sentiment_analysis import create_full_grid_indicators
        >>> daily_sentiment = create_full_grid_indicators(df_sentiment, start_date, end_date)
        >>> display_sentiment_volume(daily_sentiment, "Apple Inc.")
    """
    entity_data = df[df['Entity'] == entity].sort_values("Date")
    
    if entity_data.empty:
        return
    
    volume_col = "Volume" if volume_type == "daily" else "Volume_Rolling_30Days"
    
    # Fallback to Volume if rolling column not found
    if volume_col not in entity_data.columns:
        volume_col = "Volume"
    
    # Peak volume
    peak_vol_idx = entity_data["Volume"].idxmax()
    peak_vol_date = entity_data.loc[peak_vol_idx, "Date"]
    
    # Sentiment column - fallback to Sent_Rolling_30Days if specified column not found
    work_col = sentiment_col if sentiment_col in entity_data.columns else 'Sent_Rolling_30Days'
    if work_col not in entity_data.columns:
        return
    
    min_sent_idx = entity_data[work_col].idxmin()
    min_sent_date = entity_data.loc[min_sent_idx, "Date"]
    gauge_sent_value = entity_data[work_col].mean()
    
    # === GAUGE CHART ===
    if show_gauge and HAS_PLOTLY:
        gauge_min = -1 if work_col.endswith('_Normalized') else min(-1, gauge_sent_value * 1.1)
        gauge_max = 0
        third = (gauge_max - gauge_min) / 3
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_sent_value,
            title={'text': f"{entity} Sentiment (mean)"},
            gauge={
                'axis': {'range': [gauge_min, gauge_max]},
                'bar': {'color': "black", 'thickness': 0.25},
                'steps': [
                    {'range': [gauge_min, gauge_min + third], 'color': "red"},
                    {'range': [gauge_min + third, gauge_min + 2*third], 'color': "yellow"},
                    {'range': [gauge_min + 2*third, gauge_max], 'color': "green"}
                ]
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=80, b=40, l=0, r=0))
        fig_gauge.show()
    
    # === TIME SERIES ===
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Sentiment line
    ax1.plot(entity_data["Date"], entity_data[work_col], label='Sentiment', color='red', linestyle='--')
    ax1.axvline(min_sent_date, color='purple', linestyle=':', label='Most Negative Sentiment')
    ax1.scatter(min_sent_date, entity_data.loc[min_sent_idx, work_col], color='purple', zorder=5)
    
    # Volume line (secondary axis)
    ax2 = ax1.twinx()
    volume_label = 'Volume (Daily)' if volume_type == "daily" else 'Volume (Rolling 30D)'
    ax2.plot(entity_data["Date"], entity_data[volume_col], label=volume_label, color='green')
    ax2.axvline(peak_vol_date, color='blue', linestyle=':', label='Peak Volume')
    ax2.scatter(peak_vol_date, entity_data.loc[peak_vol_idx, volume_col], color='blue', zorder=5)
    
    # Labels and legend
    fig.legend(loc='upper left')
    ax1.set_title(f"{entity} - Sentiment & Volume Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sentiment")
    ax2.set_ylabel("Volume (Chunk Count)")
    plt.show()


def display_top_entities_dashboard(
    df_exploded: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    n: int = 5,
    sentiment_col: str = 'Sent_Rolling_30Days_Normalized',
    volume_type: str = "rolling",
    show_gauge: bool = True
):
    """
    Display sentiment and volume dashboard for top N entities by volume.
    
    Args:
        df_exploded: DataFrame from explode_to_dataframe()
        daily_sentiment: DataFrame from create_full_grid_indicators()
        n: Number of top entities to display
        sentiment_col: Column to use for sentiment
        volume_type: "daily" or "rolling"
        show_gauge: Whether to show gauge charts
        
    Example:
        >>> df_exploded = explode_to_dataframe(results, universe_csv="universe.csv")
        >>> df_sentiment = prepare_sentiment_dataframe(df_exploded)
        >>> daily_sentiment = create_full_grid_indicators(df_sentiment, start_date, end_date)
        >>> display_top_entities_dashboard(df_exploded, daily_sentiment, n=5)
    """
    top_entities = get_top_entities_by_volume(df_exploded, n=n)
    
    for entity in top_entities:
        display_sentiment_volume(
            daily_sentiment, 
            entity, 
            sentiment_col=sentiment_col,
            volume_type=volume_type,
            show_gauge=show_gauge
        )


# Impact score mapping for rolling signal (Positive=1, Negative=-1, Neutral/Unclear=0)
IMPACT_SCORE_MAP = {"Positive": 1, "Negative": -1, "Neutral": 0, "Unclear": 0}


def build_rolling_impact_signal(
    df_with_labels: pd.DataFrame,
    entity_col: str = "entity_name",
    date_col: str = "date",
    impact_col: str = "impact",
    is_theme_related_col: str = "is_theme_related",
    window_7d: int = 7,
    window_30d: int = 30,
    rolling_agg: Literal["mean", "sum"] = "mean",
) -> pd.DataFrame:
    """
    Build a rolling signal per entity: mean or sum of positive/negative news
    over the last 7 and 30 days (rows with is_theme_related == True only).

    Args:
        df_with_labels: DataFrame with columns entity_name, date, is_theme_related, impact.
        entity_col: Entity column name (default "entity_name").
        date_col: Date column name (default "date").
        impact_col: Impact column name (default "impact").
        is_theme_related_col: is_theme_related column name (default "is_theme_related").
        window_7d: Rolling window in days for weekly signal (default 7).
        window_30d: Rolling window in days for monthly signal (default 30).
        rolling_agg: "mean" = rolling mean of daily_score; "sum" = rolling sum (default "mean").

    Returns:
        DataFrame with columns:
        - entity_name (or entity_col)
        - date (or date_col)
        - daily_score: daily impact mean (1=Positive, -1=Negative, 0=Neutral/Unclear)
        - n_positive, n_negative, n_neutral, n_unclear: counts per day
        - volume: news count per day (sum of counts)
        - signal_7d, signal_30d: rolling (mean or sum) of daily_score
        - volume_7d, volume_30d: rolling sum of volume
    """
    required = [entity_col, date_col, impact_col, is_theme_related_col]
    missing = [c for c in required if c not in df_with_labels.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df_with_labels.loc[df_with_labels[is_theme_related_col] == True].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                entity_col,
                date_col,
                "daily_score",
                "n_positive",
                "n_negative",
                "n_neutral",
                "n_unclear",
                "volume",
                "signal_7d",
                "signal_30d",
                "volume_7d",
                "volume_30d",
            ]
        )

    df["impact_score"] = df[impact_col].map(IMPACT_SCORE_MAP).fillna(0)

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    daily = (
        df.groupby([entity_col, date_col])
        .agg(
            daily_score=("impact_score", "mean"),
            n_positive=(impact_col, lambda s: (s == "Positive").sum()),
            n_negative=(impact_col, lambda s: (s == "Negative").sum()),
            n_neutral=(impact_col, lambda s: (s == "Neutral").sum()),
            n_unclear=(impact_col, lambda s: (s == "Unclear").sum()),
        )
        .reset_index()
    )

    daily["volume"] = (
        daily["n_positive"] + daily["n_negative"] + daily["n_neutral"] + daily["n_unclear"]
    )
    daily = daily.sort_values([entity_col, date_col])

    if rolling_agg == "mean":
        daily["signal_7d"] = (
            daily.groupby(entity_col)["daily_score"]
            .transform(lambda s: s.rolling(window_7d, min_periods=1).mean())
        )
        daily["signal_30d"] = (
            daily.groupby(entity_col)["daily_score"]
            .transform(lambda s: s.rolling(window_30d, min_periods=1).mean())
        )
    else:
        daily["signal_7d"] = (
            daily.groupby(entity_col)["daily_score"]
            .transform(lambda s: s.rolling(window_7d, min_periods=1).sum())
        )
        daily["signal_30d"] = (
            daily.groupby(entity_col)["daily_score"]
            .transform(lambda s: s.rolling(window_30d, min_periods=1).sum())
        )

    daily["volume_7d"] = (
        daily.groupby(entity_col)["volume"]
        .transform(lambda s: s.rolling(window_7d, min_periods=1).sum())
    )
    daily["volume_30d"] = (
        daily.groupby(entity_col)["volume"]
        .transform(lambda s: s.rolling(window_30d, min_periods=1).sum())
    )

    return daily


def plot_top_entities_rolling_signal(
    df_rolling_signal: pd.DataFrame,
    entity_col: str = "entity_name",
    date_col: str = "date",
    signal_col: str = "signal_7d",
    top_n: int = 5,
    figsize: tuple[int, int] = (10, 5),
    show_volume: bool = True,
    volume_rolling: bool = True,
):
    """
    One chart per entity (top N by volume): each shows rolling signal + volume trend.

    Volume = theme-related news count (n_positive + n_negative + n_neutral + n_unclear).
    Entities are ordered by total volume; each entity gets a figure with signal (line) and volume (secondary axis).

    Args:
        df_rolling_signal: Output of build_rolling_impact_signal (with volume_7d, volume_30d if show_volume=True).
        entity_col: Entity column name (default "entity_name").
        date_col: Date column name (default "date").
        signal_col: Signal column to plot (default "signal_7d"; alternative "signal_30d").
        top_n: Number of entities to show = number of charts (default 5).
        figsize: Size of each figure (default (10, 5)).
        show_volume: Show volume trend (default True).
        volume_rolling: Use rolling volume (volume_7d/volume_30d) instead of daily volume (default True).
    """
    volume_cols = ["n_positive", "n_negative", "n_neutral", "n_unclear"]
    missing = [c for c in volume_cols if c not in df_rolling_signal.columns]
    if missing:
        raise ValueError(
            f"Missing columns for volume: {missing}. Use the output of build_rolling_impact_signal."
        )

    df = df_rolling_signal.copy()
    df["_volume"] = (
        df["n_positive"] + df["n_negative"] + df["n_neutral"] + df["n_unclear"]
    )
    volume_per_entity = df.groupby(entity_col)["_volume"].sum()
    top_entities = volume_per_entity.nlargest(top_n).index.tolist()

    if not top_entities:
        return

    # Volume column to plot: aligned with signal window
    if show_volume and volume_rolling:
        if signal_col == "signal_7d" and "volume_7d" in df.columns:
            vol_col = "volume_7d"
        elif signal_col == "signal_30d" and "volume_30d" in df.columns:
            vol_col = "volume_30d"
        else:
            vol_col = "_volume"
    elif show_volume:
        vol_col = "_volume"
    else:
        vol_col = None

    df_plot = df[df[entity_col].isin(top_entities)].copy()
    if not pd.api.types.is_datetime64_any_dtype(df_plot[date_col]):
        df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors="coerce")

    # Warm-up offset: start x-axis after this many days so the rolling signal is full (30d -> 30, else 7)
    padding_days = 30 if "30" in signal_col else 7

    for entity in top_entities:
        sub = df_plot[df_plot[entity_col] == entity].sort_values(date_col)
        if sub.empty:
            continue

        fig, ax1 = plt.subplots(figsize=figsize)

        # Signal (left axis)
        ax1.plot(
            sub[date_col], sub[signal_col], color="tab:blue", marker=".", markersize=3
        )
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_xlabel("Date")
        ax1.set_ylabel(signal_col, color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{entity}\nRolling signal and volume")

        # Start x-axis after warm-up so we only show dates where the rolling signal is full
        date_min = sub[date_col].min()
        date_max = sub[date_col].max()
        visible_start = date_min + pd.Timedelta(days=padding_days)
        ax1.set_xlim(left=visible_start, right=date_max)

        if vol_col:
            ax2 = ax1.twinx()
            ax2.fill_between(
                sub[date_col],
                sub[vol_col],
                alpha=0.25,
                color="tab:orange",
                label="Volume",
            )
            ax2.plot(
                sub[date_col], sub[vol_col], color="tab:orange", linewidth=1, alpha=0.8
            )
            ax2.set_ylabel("Volume" if vol_col == "_volume" else vol_col, color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
