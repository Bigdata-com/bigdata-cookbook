from .report_generator import SummaryGenerator
import pandas as pd

def get_narrative_windows(daily_df: pd.DataFrame, raw_df: pd.DataFrame, sentiment_col: str = 'avg_sent_smoothed', min_max: str = 'min', lookback_days: int = 30):
    # Select operation method from pandas Series
    operation = 'idxmin' if min_max == 'min' else 'idxmax'

    ## find peak coverage as idxmax of volume by entity
    peak_coverage =daily_df.groupby('Entity')['Volume'].idxmax()
    peak_sentiment = daily_df.groupby('Entity')[sentiment_col].agg(operation)

    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    peak_coverage_windows = {}
    peak_coverage_data = {}
    for entity, peak_idx in peak_coverage.items():
        peak_coverage_windows[entity] = {}
        peak_coverage_data[entity] = {}
        peak_date = daily_df.loc[peak_idx, 'Date']
        start_date_window = peak_date - pd.Timedelta(days=lookback_days-1)
        mask = (raw_df['Entity'] == entity) & (raw_df['Date'] >= start_date_window) & (raw_df['Date'] <= peak_date)
        peak_coverage_windows[entity]['volume'] = raw_df.loc[mask]
        peak_coverage_data[entity]['volume_index'] = peak_date
        peak_coverage_data[entity]['volume_value'] = daily_df.loc[peak_idx, 'Volume']

    for entity, peak_idx in peak_sentiment.items():
        peak_date = daily_df.loc[peak_idx, 'Date']
        start_date_window = peak_date - pd.Timedelta(days=lookback_days-1)
        mask = (raw_df['Entity'] == entity) & (raw_df['Date'] >= start_date_window) & (raw_df['Date'] <= peak_date)
        peak_coverage_windows[entity]['sentiment'] = raw_df.loc[mask]
        peak_coverage_data[entity]['sentiment_index'] = peak_date
        peak_coverage_data[entity]['sentiment_value'] = daily_df.loc[peak_idx][sentiment_col]

    return peak_coverage_windows, peak_coverage_data

def summarize_narratives(df: pd.DataFrame, entity: str, report_generator: SummaryGenerator, narrative_windows: dict):
    entity_narratives = {}
    for entity, df in narrative_windows.items():
        print(f"Processing entity: {entity}")
        entity_narratives[entity] = {}
        for key, value in df.items():
            print(f"  Processing window type: {key} with {len(value)} records")
            # NOTE: the original SDK's RiskLabeler emitted hierarchical
            # "Risk Channel" / "Risk Factor" taxonomy fields (from the removed
            # generate_risk_tree mind map). SimpleLabeler only emits a flat
            # "Sub-Scenario" label, so we substitute that here instead.
            report_text = report_generator.prepare_narrative_summary_input(value, entity_name=entity, date_col='Date', text_col='Quote', sentence_id_col='Document ID', summary_input=['Headline', 'Sub-Scenario', 'Quote'])

            summary = report_generator.summarize_string(report_text)
            entity_narratives[entity][key] = summary

        print(f"Completed processing for entity: {entity}\n")

    return entity_narratives

def display_dashboard(df: pd.DataFrame, entity: str, sentiment_col: str, narratives: dict, export_mode: bool = False, gauge_value: str = "min", volume_type: str = "daily"):
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    import plotly.graph_objects as go
    
    # Only import widgets if not in export mode
    if not export_mode:
        import ipywidgets as widgets

    entity_data = df[df['Entity'] == entity].sort_values("Date")

    # Determine which volume column to use for PLOTTING
    volume_col = "Volume" if volume_type == "daily" else "Volume_Rolling_30Days"
    
    # Check if the volume column exists
    if volume_col not in entity_data.columns:
        print(f"Warning: Volume column {volume_col} not found. Falling back to 'Volume'")
        volume_col = "Volume"
    
    # ALWAYS find peak volume using daily volume (for narrative consistency)
    peak_vol_idx = entity_data["Volume"].idxmax()
    peak_vol_date = entity_data.loc[peak_vol_idx, "Date"]
    peak_vol_value = entity_data.loc[peak_vol_idx, "Volume"]

    # Use the specified sentiment column
    work_col = sentiment_col
    
    # Check if the specified column exists
    if work_col not in entity_data.columns:
        print(f"Warning: Column {work_col} not found in data. Available columns: {list(entity_data.columns)}")
        # Fallback to a default column if available
        fallback_col = 'Sent_Rolling_30Days'
        if fallback_col in entity_data.columns:
            work_col = fallback_col
            print(f"Using fallback column: {work_col}")
        else:
            raise ValueError(f"Neither {sentiment_col} nor {fallback_col} found in data")
    
    # Calculate gauge value based on parameter
    if gauge_value == "mean":
        gauge_sent_value = entity_data[work_col].mean()
        min_sent_date = entity_data.loc[entity_data[work_col].idxmin(), "Date"]
    else:  # min: use minimum (most negative)
        min_sent_idx = entity_data[work_col].idxmin()
        min_sent_date = entity_data.loc[min_sent_idx, "Date"]
        gauge_sent_value = entity_data.loc[min_sent_idx, work_col]

    # # Print summary stats
    # print(f"Entity: {entity}")
    # print(f"Highest Volume: {peak_vol_value:.0f} on {peak_vol_date.date()}")
    # print(f"Most Negative Sentiment: {min_sent_value:.2f} on {min_sent_date.date()}")

    # Gauge chart - adjust range based on column type
    if work_col.endswith('_Normalized'):
        # Use fixed range [-1, 0] for normalized values
        gauge_min = -1
        gauge_max = 0
    else:
        # Use dynamic range for original values
        gauge_min = min(-1, gauge_sent_value * 1.1)
        gauge_max = 0

    third = (gauge_max - gauge_min) / 3
    zones = [
        {'range': [gauge_min, gauge_min + third], 'color': "red"},
        {'range': [gauge_min + third, gauge_min + 2*third], 'color': "yellow"},
        {'range': [gauge_min + 2*third, gauge_max], 'color': "green"}
    ]

    # Create title based on gauge_value parameter
    gauge_title_suffix = "(mean)" if gauge_value == "mean" else "(min)"
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_sent_value,
        title={'text': f"{entity} Risk Sentiment {gauge_title_suffix}"},
        gauge={
            'axis': {'range': [gauge_min, gauge_max], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "black", 'thickness': 0.25},
            'steps': zones,
            'threshold': {
                'line': {'color': "black", 'width': 8},
                'thickness': 1.0,
                'value': gauge_sent_value
            },
            'shape': "angular"
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=80, b=40, l=0, r=0))
    fig_gauge.show()

    # Time series plot
    fig, ax1 = plt.subplots(figsize=(14, 6))
    # Optional: plot daily sentiment if needed
    # ax1.plot(entity_data["Date"], entity_data['Avg_Sentiment'], label='Daily Sentiment', color='orange')
    # Use the same column as determined for gauge chart
    plot_col = work_col
    
    ax1.plot(entity_data["Date"], entity_data[plot_col], label='Smoothed Sentiment', color='red', linestyle='--')
    ax1.axvline(min_sent_date, color='purple', linestyle=':', label='Most Negative Sentiment')
    ax1.scatter(min_sent_date, entity_data.loc[entity_data["Date"] == min_sent_date, plot_col].iloc[0], color='purple', zorder=5)
    ax2 = ax1.twinx()
    
    # Plot volume based on volume_type parameter
    volume_label = 'Volume (Daily)' if volume_type == "daily" else 'Volume (Rolling 30D)'
    ax2.plot(entity_data["Date"], entity_data[volume_col], label=volume_label, color='green')
    
    # Peak marker and narrative always based on daily volume peak
    ax2.axvline(peak_vol_date, color='blue', linestyle=':', label='Daily Peak Volume')
    # Get the y-value from the plotted series at the peak date for correct marker placement
    peak_vol_y_on_plot = entity_data.loc[peak_vol_idx, volume_col]
    ax2.scatter(peak_vol_date, peak_vol_y_on_plot, color='blue', zorder=5)
    ax2.annotate(
        "Read narrative at daily peak volume ↓",
        xy=(peak_vol_date, peak_vol_y_on_plot),
        xytext=(peak_vol_date, entity_data[volume_col].max()),
        bbox=dict(boxstyle="round,pad=0.5", fc="#e0f7fa", alpha=0.8),
        fontsize=11
    )
    # Get the y-coordinate for annotation
    annotation_y = entity_data.loc[entity_data["Date"] == min_sent_date, plot_col].iloc[0]
    min_y = entity_data[plot_col].min()
    
    ax1.annotate(
        "Read narrative at peak sentiment ↓",
        xy=(min_sent_date, annotation_y),
        xytext=(min_sent_date, min_y),
        bbox=dict(boxstyle="round,pad=0.5", fc="#e0f7fa", alpha=0.8),
        fontsize=11
    )
    fig.legend(loc='upper left')
    ax1.set_title(f"{entity} Sentiment Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Entity Sentiment")
    ax2.set_ylabel("Volume")
    plt.show()

    narratives = narratives[entity]
    
    # Format dates for display
    peak_vol_date_str = peak_vol_date.strftime('%d-%m-%Y')
    min_sent_date_str = min_sent_date.strftime('%d-%m-%Y')
    
    if export_mode:
        # For export mode, display narratives as regular markdown with headers
        display(Markdown(f"## Peak Volume Narrative - Date: {peak_vol_date_str}"))
        display(Markdown(narratives['volume']))
        display(Markdown(f"## Most Negative Sentiment Narrative - Date: {min_sent_date_str}"))
        display(Markdown(narratives['sentiment']))
    else:
        # For interactive mode, use accordion widgets
        output_volume = widgets.Output()
        output_sentiment = widgets.Output()
        with output_volume:
            display(Markdown(narratives['volume']))
        with output_sentiment:
            display(Markdown(narratives['sentiment']))

        accordion = widgets.Accordion(children=[output_volume, output_sentiment])
        accordion.set_title(0, f'Peak Volume Narrative - Date: {peak_vol_date_str}')
        accordion.set_title(1, f'Most Negative Sentiment Narrative - Date: {min_sent_date_str}')
        display(accordion)