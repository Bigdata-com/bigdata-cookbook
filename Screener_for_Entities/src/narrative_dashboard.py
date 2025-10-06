from .report_generator import SummaryGenerator
import pandas as pd

def get_narrative_windows(daily_df: pd.DataFrame, raw_df: pd.DataFrame,  sentiment_cols: dict = {'average':'Avg_Sentiment', 'smoothed':'avg_sent_smoothed'}, min_max: str = 'min', lookback_days: int = 30):
    # Select operation method from pandas Series
    operation = 'idxmin' if min_max == 'min' else 'idxmax'

    ## find peak coverage as idxmax of volume by entity
    peak_coverage =daily_df.groupby('Entity')['Volume'].idxmax()
    peak_sentiment = daily_df.groupby('Entity')[sentiment_cols['smoothed']].agg(operation)

    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    peak_coverage_windows = {}
    for entity, peak_idx in peak_coverage.items():
        peak_coverage_windows[entity] = {}
        peak_date = daily_df.loc[peak_idx, 'Date']
        start_date_window = peak_date - pd.Timedelta(days=lookback_days-1)
        mask = (raw_df['Entity'] == entity) & (raw_df['Date'] >= start_date_window) & (raw_df['Date'] <= peak_date)
        peak_coverage_windows[entity]['volume'] = raw_df.loc[mask]

    for entity, peak_idx in peak_sentiment.items():
        peak_date = daily_df.loc[peak_idx, 'Date']
        start_date_window = peak_date - pd.Timedelta(days=lookback_days-1)
        mask = (raw_df['Entity'] == entity) & (raw_df['Date'] >= start_date_window) & (raw_df['Date'] <= peak_date)
        peak_coverage_windows[entity]['sentiment'] = raw_df.loc[mask]

    return peak_coverage_windows

def summarize_narratives(df: pd.DataFrame, entity: str, report_generator: SummaryGenerator, narrative_windows: dict):
    entity_narratives = {}
    for entity, df in narrative_windows.items():
        print(f"Processing entity: {entity}")
        entity_narratives[entity] = {}
        for key, value in df.items():
            print(f"  Processing window type: {key} with {len(value)} records")

            report_text = report_generator.prepare_narrative_summary_input(value, entity_name=entity, date_col='Date', text_col='Quote', sentence_id_col='Document ID', summary_input=['Headline', 'Risk Channel', 'Risk Factor', 'Quote'])

            summary = report_generator.summarize_string(report_text)
            entity_narratives[entity][key] = summary

        print(f"Completed processing for entity: {entity}\n")

    return entity_narratives

def display_dashboard(df: pd.DataFrame, entity: str, sentiment_cols: dict, narratives: dict):
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    import ipywidgets as widgets
    import plotly.graph_objects as go

    entity_data = df[df['Entity'] == entity].sort_values("Date")

    # Find highest volume and its date
    peak_vol_idx = entity_data["Volume"].idxmax()
    peak_vol_date = entity_data.loc[peak_vol_idx, "Date"]
    peak_vol_value = entity_data.loc[peak_vol_idx, "Volume"]

    # Find most negative smoothed entity sentiment and its date
    min_sent_idx = entity_data[sentiment_cols['average']].idxmin()
    min_sent_date = entity_data.loc[min_sent_idx, "Date"]
    min_sent_value = entity_data.loc[min_sent_idx, sentiment_cols['average']]

    # Print summary stats
    print(f"Entity: {entity}")
    print(f"Highest Volume: {peak_vol_value:.0f} on {peak_vol_date.date()}")
    print(f"Most Negative Sentiment: {min_sent_value:.2f} on {min_sent_date.date()}")

    # Gauge chart
    third = (0 - (-1)) / 3
    zones = [
        {'range': [-1, -1 + third], 'color': "red"},
        {'range': [-1 + third, -1 + 2*third], 'color': "yellow"},
        {'range': [-1 + 2*third, 0], 'color': "green"}
    ]

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min_sent_value,
        title={'text': f"{entity} Risk Sentiment (Lowest)"},
        gauge={
            'axis': {'range': [-1, 0], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "black", 'thickness': 0.25},
            'steps': zones,
            'threshold': {
                'line': {'color': "black", 'width': 8},
                'thickness': 1.0,
                'value': min_sent_value
            },
            'shape': "angular"
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
    fig_gauge.show()

    # Time series plot
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(entity_data["Date"], entity_data[sentiment_cols['average']], label='Sentiment', color='orange')
    ax1.plot(entity_data["Date"], entity_data[sentiment_cols['smoothed']], label='Smoothed Sentiment', color='red', linestyle='--')
    ax1.axvline(min_sent_date, color='purple', linestyle=':', label='Most Negative Sentiment')
    ax1.scatter(min_sent_date, min_sent_value, color='purple', zorder=5)
    ax2 = ax1.twinx()
    ax2.plot(entity_data["Date"], entity_data["Volume"], label='Volume', color='green')
    ax2.axvline(peak_vol_date, color='blue', linestyle=':', label='Peak Volume')
    ax2.scatter(peak_vol_date, peak_vol_value, color='blue', zorder=5)
    ax2.annotate(
        "Read narrative at peak volume ↓",
        xy=(peak_vol_date, entity_data.loc[peak_vol_idx, "Volume"]),
        xytext=(peak_vol_date, entity_data["Volume"].max()),
        bbox=dict(boxstyle="round,pad=0.5", fc="#e0f7fa", alpha=0.8),
        fontsize=11
    )
    ax1.annotate(
        "Read narrative at peak sentiment ↓",
        xy=(min_sent_date, entity_data.loc[min_sent_idx, sentiment_cols['average']]),
        xytext=(min_sent_date, entity_data[sentiment_cols['average']].min()),
        bbox=dict(boxstyle="round,pad=0.5", fc="#e0f7fa", alpha=0.8),
        fontsize=11
    )
    fig.legend(loc='upper left')
    ax1.set_title(f"{entity} Sentiment & Volume Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Entity Sentiment")
    ax2.set_ylabel("Volume")
    plt.show()

    narratives = narratives[entity]
    output_volume = widgets.Output()
    output_sentiment = widgets.Output()
    with output_volume:
        display(Markdown(narratives['volume']))
    with output_sentiment:
        display(Markdown(narratives['sentiment']))

    accordion = widgets.Accordion(children=[output_volume, output_sentiment])
    accordion.set_title(0, 'Peak Volume Narrative')
    accordion.set_title(1, 'Most Negative Sentiment Narrative')
    display(accordion)