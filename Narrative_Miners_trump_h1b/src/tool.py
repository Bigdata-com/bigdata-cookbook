import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_results(file_path, source_type):
    """
    Load and clean narrative mining results.

    Parameters:
        file_path (str): Path to the Excel file containing mining results.
        source_type (str): Type of data source (News, Earnings Call, SEC Filing).

    Returns:
        pd.DataFrame: Cleaned dataframe with source type label.
    """
    df = pd.read_excel(file_path, header=1).reset_index(drop=True)
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in str(col)])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Source_Type'] = source_type  # Add source type column
    print(f"Loaded {len(df)} narrative records from {source_type}")
    return df


def prepare_narrative_data(df, freq='W'):
    """
    Prepare narrative data for visualization by creating time series of narrative counts,
    converting to z-scores, and applying smoothing.
    """
    pivot_df = pd.pivot_table(df, index='Date', columns='Label', aggfunc='size', fill_value=0)
    resampled_df = pivot_df.resample(freq).sum()

    # Calculate z-scores for each narrative
    zscore_df = pd.DataFrame()
    for column in resampled_df.columns:
        mean = resampled_df[column].mean()
        std = resampled_df[column].std()
        if std == 0:
            zscore_df[column] = 0
        else:
            zscore_df[column] = (resampled_df[column] - mean) / std

    # Apply smoothing using Gaussian filter
    smoothed_df = pd.DataFrame(index=zscore_df.index)
    for column in zscore_df.columns:
        smoothed_data = gaussian_filter1d(zscore_df[column].fillna(0).values, sigma=2)
        smoothed_df[column] = smoothed_data
    return smoothed_df


def calculate_source_scores(df):
    """
    Calculate overall narrative scores (z-scores) across all narratives by source.
    """
    date_counts = df.groupby('Date').size()
    weekly_counts = date_counts.resample('W').sum().fillna(0)
    mean = weekly_counts.mean()
    std = weekly_counts.std()
    if std == 0:
        zscore = weekly_counts * 0
    else:
        zscore = (weekly_counts - mean) / std
    smoothed = gaussian_filter1d(zscore.fillna(0).values, sigma=2)
    return pd.Series(smoothed, index=zscore.index)


def visualize_cross_source_narratives(news_df, transcripts_df, filings_df, interactive=True):
    """Create a comparative visualization of narrative prevalence across sources with fixed annotation arrows"""
    # Prepare data for each source
    news_score = calculate_source_scores(news_df)
    transcript_score = calculate_source_scores(transcripts_df)
    filing_score = calculate_source_scores(filings_df)

    # Align indices across all sources
    all_dates = sorted(set(news_score.index) |
                       set(transcript_score.index) |
                       set(filing_score.index))

    # Create dataframe with aligned dates
    comparison_df = pd.DataFrame(index=all_dates)
    comparison_df['News Media'] = pd.Series(news_score)
    comparison_df['Earnings Calls'] = pd.Series(transcript_score)
    comparison_df['SEC Filings'] = pd.Series(filing_score)
    comparison_df = comparison_df.sort_index().fillna(method='ffill').fillna(0)

    # Define colors
    source_colors = {
        'News Media': '#FF6B6B',
        'Earnings Calls': '#4ECDC4',
        'SEC Filings': '#6A0572'
    }

    # Find important points for annotations
    peak_news = comparison_df['News Media'].idxmax()
    peak_earnings = comparison_df['Earnings Calls'].idxmax()
    peak_filings = comparison_df['SEC Filings'].idxmax()

    # Use matplotlib for interactive=False, Plotly for interactive=True
    if not interactive:
        # MATPLOTLIB VERSION (lightweight)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot lines for each source
        for source, color in source_colors.items():
            ax.plot(comparison_df.index, comparison_df[source], 
                    color=color, linewidth=3, label=source, alpha=0.9)
        
        # Add annotations
        ax.annotate('Peak news coverage\nof AI bubble concerns',
                   xy=(peak_news, comparison_df.loc[peak_news, 'News Media']),
                   xytext=(peak_news, comparison_df.loc[peak_news, 'News Media'] + 0.5),
                   arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2),
                   fontsize=10, ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#FF6B6B', alpha=0.8))
        
        ax.annotate('Executives address\nbubble concerns on\nearnings calls',
                   xy=(peak_earnings, comparison_df.loc[peak_earnings, 'Earnings Calls']),
                   xytext=(peak_earnings, comparison_df.loc[peak_earnings, 'Earnings Calls'] + 0.8),
                   arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=2),
                   fontsize=10, ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#4ECDC4', alpha=0.8))
        
        ax.annotate('Peak mentions in\nSEC filings',
                   xy=(peak_filings, comparison_df.loc[peak_filings, 'SEC Filings']),
                   xytext=(peak_filings, comparison_df.loc[peak_filings, 'SEC Filings'] - 0.5),
                   arrowprops=dict(arrowstyle='->', color='#6A0572', lw=2),
                   fontsize=10, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#6A0572', alpha=0.8))
        
        # Add horizontal reference line at y=0
        ax.axhline(y=0, color='#666666', linestyle='--', alpha=0.7, linewidth=1)
        
        # Customize the plot
        ax.set_title('AI Bubble Narrative: Media vs. Corporate Communications', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Narrative Intensity (z-score)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Set y-axis limits
        ax.set_ylim(-1.5, 3.5)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('white')
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, 
                  fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create a wrapper class that mimics Plotly's show() method
        class MatplotlibFigWrapper:
            def __init__(self, fig):
                self.fig = fig
            
            def show(self):
                plt.show()
                
            def __getattr__(self, name):
                return getattr(self.fig, name)
        
        return MatplotlibFigWrapper(fig)
    
    else:
        # PLOTLY VERSION (interactive)
        fig = go.Figure()

        for source, color in source_colors.items():
            # Configure hover template based on interactive mode
            hover_template = (
                f"<b>{source}</b><br>" +
                "Date: %{x|%B %d, %Y}<br>" +
                "Intensity: %{y:.2f}<extra></extra>"
            ) if interactive else None
            
            fig.add_trace(
                go.Scatter(
                    x=comparison_df.index,
                    y=comparison_df[source],
                    mode='lines',
                    name=source,
                    line=dict(width=3, color=color),
                    hovertemplate=hover_template,
                    hoverinfo='none' if not interactive else 'all'
                )
            )

        # Create annotations with fixed arrows (only if interactive)
        annotations = []
        if interactive:
            annotations = [
                # Peak news annotation
                dict(
                    x=peak_news,
                    y=comparison_df.loc[peak_news, 'News Media'],
                    text="Peak news coverage<br>of AI bubble concerns",
                    showarrow=True,
                    arrowhead=2,
                    ax=10,
                    ay=-30,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#FF6B6B",
                    borderwidth=2,
                    font=dict(size=10),
                    xanchor="right"
                ),

                # Earnings calls annotation
                dict(
                    x=peak_earnings,
                    y=comparison_df.loc[peak_earnings, 'Earnings Calls'],
                    text="Executives address<br>bubble concerns on<br>earnings calls",
                    showarrow=True,
                    arrowhead=2,
                    ax=90,
                    ay=-20,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#4ECDC4",
                    borderwidth=2,
                    font=dict(size=10),
                    xanchor="right"
                ),
                
                # SEC filings annotation
                dict(
                    x=peak_filings,
                    y=comparison_df.loc[peak_filings, 'SEC Filings'],
                    text="Peak mentions in<br>SEC filings",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=60,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#6A0572",
                    borderwidth=2,
                    font=dict(size=10),
                    xanchor="center",
                    yanchor="top"
                )
            ]

        # Customize the layout
        fig.update_layout(
            title={
                'text': 'AI Bubble Narrative: Media vs. Corporate Communications',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18, color='#1f1f1f')
            },
            xaxis=dict(
                title='',
                gridcolor='rgba(100, 100, 100, 0.2)',
                tickangle=-45,
                tickformat='%b %Y',
                tickfont=dict(color='#1f1f1f', size=10),
                showgrid=True,
                fixedrange=not interactive  # Disable zoom/pan when not interactive
            ),
            yaxis=dict(
                title=dict(text='Narrative Intensity (z-score)', font=dict(color='#1f1f1f')),
                tickfont=dict(color='#1f1f1f'),
                gridcolor='rgba(100, 100, 100, 0.2)',
                zerolinecolor='rgba(0, 0, 0, 0.4)',
                range=[-1.5, 3.5],
                automargin=True,
                fixedrange=not interactive  # Disable zoom/pan when not interactive
            ),
            hovermode='closest' if interactive else False,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.2,
                xanchor='center',
                x=0.5,
                font=dict(size=12, color='#1f1f1f'),
                bgcolor='rgba(255,255,255,0.8)'
            ),
            annotations=annotations,
            margin=dict(l=50, r=50, t=70, b=120),
            template='plotly',
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            height=600,
            showlegend=True
        )

        # Add horizontal reference line
        fig.add_shape(
            type='line',
            x0=comparison_df.index.min(),
            y0=0,
            x1=comparison_df.index.max(),
            y1=0,
            line=dict(
                color='#666666',
                width=1,
                dash='dash'
            )
        )

        # Add time period selectors only if interactive
        if interactive:
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ]),
                    bgcolor='rgba(150,150,150,0.2)'
                )
            )
        
        # Configure static mode
        if not interactive:
            fig.update_layout(
                dragmode=False
            )

        return fig


def visualize_cross_source_narratives_matplotlib(news_df, transcripts_df, filings_df, interactive=False):
    """
    Create a comparative visualization using matplotlib (lightweight alternative to Plotly)
    """
    # Prepare data for each source
    news_score = calculate_source_scores(news_df)
    transcript_score = calculate_source_scores(transcripts_df)
    filing_score = calculate_source_scores(filings_df)

    # Align indices across all sources
    all_dates = sorted(set(news_score.index) |
                       set(transcript_score.index) |
                       set(filing_score.index))

    # Create dataframe with aligned dates
    comparison_df = pd.DataFrame(index=all_dates)
    comparison_df['News Media'] = pd.Series(news_score)
    comparison_df['Earnings Calls'] = pd.Series(transcript_score)
    comparison_df['SEC Filings'] = pd.Series(filing_score)
    comparison_df = comparison_df.sort_index().fillna(method='ffill').fillna(0)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors (same as Plotly version)
    source_colors = {
        'News Media': '#FF6B6B',
        'Earnings Calls': '#4ECDC4',
        'SEC Filings': '#6A0572'
    }
    
    # Plot lines for each source
    for source, color in source_colors.items():
        ax.plot(comparison_df.index, comparison_df[source], 
                color=color, linewidth=3, label=source, alpha=0.9)
    
    # Find important points for annotations
    peak_news = comparison_df['News Media'].idxmax()
    peak_earnings = comparison_df['Earnings Calls'].idxmax()
    peak_filings = comparison_df['SEC Filings'].idxmax()
    
    # Add annotations (only if interactive=False to keep it simple)
    if not interactive:
        # Peak news annotation
        ax.annotate('Peak news coverage\nof AI bubble concerns',
                   xy=(peak_news, comparison_df.loc[peak_news, 'News Media']),
                   xytext=(peak_news, comparison_df.loc[peak_news, 'News Media'] + 0.5),
                   arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2),
                   fontsize=10, ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#FF6B6B', alpha=0.8))
        
        # Earnings calls annotation
        ax.annotate('Executives address\nbubble concerns on\nearnings calls',
                   xy=(peak_earnings, comparison_df.loc[peak_earnings, 'Earnings Calls']),
                   xytext=(peak_earnings, comparison_df.loc[peak_earnings, 'Earnings Calls'] + 0.8),
                   arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=2),
                   fontsize=10, ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#4ECDC4', alpha=0.8))
        
        # SEC filings annotation
        ax.annotate('Peak mentions in\nSEC filings',
                   xy=(peak_filings, comparison_df.loc[peak_filings, 'SEC Filings']),
                   xytext=(peak_filings, comparison_df.loc[peak_filings, 'SEC Filings'] - 0.5),
                   arrowprops=dict(arrowstyle='->', color='#6A0572', lw=2),
                   fontsize=10, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#6A0572', alpha=0.8))
    
    # Add horizontal reference line at y=0
    ax.axhline(y=0, color='#666666', linestyle='--', alpha=0.7, linewidth=1)
    
    # Customize the plot
    ax.set_title('AI Bubble Narrative: Media vs. Corporate Communications', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Narrative Intensity (z-score)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set_ylim(-1.5, 3.5)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, 
              fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def visualize_news_narrative_breakdown(news_df, interactive=True):
    """Create a stacked area chart showing the breakdown of specific narratives in news with unique colors"""
    # Prepare news narrative data
    news_narratives = prepare_narrative_data(news_df, freq='W')

    # Filter to only include the top narratives
    max_values = news_narratives.max()
    top_narratives = max_values.nlargest(10).index.tolist()
    filtered_narratives = news_narratives[top_narratives]

    # Create an expanded color palette with distinct colors for up to 10 narratives
    # Each color is uniquely chosen to have good contrast with adjacent colors
    distinct_colors = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#FFD166',  # Gold
        '#6A0572',  # Purple
        '#2D7DD2',  # Blue
        '#97CC04',  # Green
        '#F45D01',  # Orange
        '#8367C7',  # Lavender
        '#C44536',  # Brick
        '#2A6041'   # Forest green
    ]

    # Use matplotlib for interactive=False, Plotly for interactive=True
    if not interactive:
        # MATPLOTLIB VERSION (lightweight)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create stacked area plot
        ax.stackplot(filtered_narratives.index, 
                    *[filtered_narratives[col] for col in filtered_narratives.columns],
                    labels=filtered_narratives.columns,
                    colors=distinct_colors[:len(filtered_narratives.columns)],
                    alpha=0.8)
        
        # Customize the plot
        ax.set_title('Breakdown of AI Bubble Narratives in News Media', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Narrative Prevalence (z-score)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('white')
        
        # Add legend (horizontal layout below the plot)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, 
                  fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create a wrapper class that mimics Plotly's show() method
        class MatplotlibFigWrapper:
            def __init__(self, fig):
                self.fig = fig
            
            def show(self):
                plt.show()
                
            def __getattr__(self, name):
                return getattr(self.fig, name)
        
        return MatplotlibFigWrapper(fig)
    
    else:
        # PLOTLY VERSION (interactive)
        fig = go.Figure()

        # Add traces for each narrative with its unique color
        for i, column in enumerate(filtered_narratives.columns):
            color = distinct_colors[i]  # Each narrative gets its own unique color
            
            # Configure hover template based on interactive mode
            hover_template = (
                "<b>%{fullData.name}</b><br>" +
                "Date: %{x|%B %d, %Y}<br>" +
                "Z-Score: %{y:.2f}<extra></extra>"
            ) if interactive else None
            
            fig.add_trace(
                go.Scatter(
                    x=filtered_narratives.index,
                    y=filtered_narratives[column],
                    mode='lines',
                    name=column,
                    line=dict(width=0.5, color=color),
                    stackgroup='one',
                    fillcolor=color,
                    hovertemplate=hover_template,
                    hoverinfo='none' if not interactive else 'all'
                )
            )

        # Customize layout
        fig.update_layout(
            title={
                'text': 'Breakdown of AI Bubble Narratives in News Media',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18, color='#1f1f1f')
            },
            xaxis=dict(
                title='',
                gridcolor='rgba(100, 100, 100, 0.2)',
                tickangle=-45,
                tickformat='%b %Y',
                tickfont=dict(color='#1f1f1f', size=11),
                showgrid=True,
                fixedrange=not interactive  # Disable zoom/pan when not interactive
            ),
            yaxis=dict(
                title=dict(text='Narrative Prevalence (z-score)', font=dict(color='#1f1f1f')),
                tickfont=dict(color='#1f1f1f'),
                gridcolor='rgba(100, 100, 100, 0.2)',
                zerolinecolor='rgba(0, 0, 0, 0.4)',
                fixedrange=not interactive  # Disable zoom/pan when not interactive
            ),
            hovermode='closest' if interactive else False,
            # Use horizontal legend below the chart with improved formatting
            legend=dict(
                orientation='h',  # Horizontal orientation for better space usage
                yanchor='top',
                y=-0.25,  # Further below the chart
                xanchor='center',
                x=0.5,     # Centered
                font=dict(size=13),  # Larger font size
                bgcolor='rgba(255,255,255,0.95)',  # More opaque background
                bordercolor='rgba(100,100,100,0.5)',  # More visible border
                borderwidth=1,
                itemsizing='constant',  # Fixed size markers
                itemwidth=40,  # Wider legend items
                traceorder='normal'
            ),
            margin=dict(l=50, r=50, t=80, b=280),  # Increased bottom margin for legend
            template='plotly',
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            height=750,  # Increased height to accommodate legend
            showlegend=True
        )

        # Add time period selectors only if interactive
        if interactive:
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    bgcolor='rgba(150,150,150,0.2)'
                )
            )
        
        # Configure static mode
        if not interactive:
            fig.update_layout(
                dragmode=False
            )

        return fig


def visualize_news_narrative_breakdown_matplotlib(news_df, interactive=False):
    """
    Create a stacked area chart showing breakdown of news narratives using matplotlib
    """
    # Prepare news narrative data
    news_narratives = prepare_narrative_data(news_df, freq='W')
    
    # Filter to only include the top narratives
    max_values = news_narratives.max()
    top_narratives = max_values.nlargest(10).index.tolist()
    filtered_narratives = news_narratives[top_narratives]
    
    # Create distinct colors for up to 10 narratives
    distinct_colors = [
        '#FF6B6B', '#4ECDC4', '#FFD166', '#6A0572', '#2D7DD2',
        '#97CC04', '#F45D01', '#8367C7', '#C44536', '#2A6041'
    ]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create stacked area plot
    ax.stackplot(filtered_narratives.index, 
                *[filtered_narratives[col] for col in filtered_narratives.columns],
                labels=filtered_narratives.columns,
                colors=distinct_colors[:len(filtered_narratives.columns)],
                alpha=0.8)
    
    # Customize the plot
    ax.set_title('Breakdown of AI Bubble Narratives in News Media', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Narrative Prevalence (z-score)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    # Add legend (horizontal layout below the plot)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, 
              fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_lightweight_plots(news_df, transcripts_df, filings_df, save_plots=True):
    """
    Create both plots using matplotlib and optionally save them
    """
    print("📊 Creating lightweight matplotlib plots...")
    
    # Create plots
    fig1 = visualize_cross_source_narratives_matplotlib(news_df, transcripts_df, filings_df, interactive=False)
    fig2 = visualize_news_narrative_breakdown_matplotlib(news_df, interactive=False)
    
    if save_plots:
        # Save plots
        fig1.savefig('output/cross_source_narratives_matplotlib.png', dpi=300, bbox_inches='tight')
        fig2.savefig('output/news_breakdown_matplotlib.png', dpi=300, bbox_inches='tight')
        print("✅ Plots saved to:")
        print("  - output/cross_source_narratives_matplotlib.png")
        print("  - output/news_breakdown_matplotlib.png")
    
    # Show plots
    plt.show()
    
    return fig1, fig2


def extract_narrative_insights(news_df, transcripts_df, filings_df):
    """
    Extract key insights from narrative mining data.
    """
    news_score = calculate_source_scores(news_df)
    transcript_score = calculate_source_scores(transcripts_df)
    filing_score = calculate_source_scores(filings_df)

    peak_news_month = news_score.idxmax().strftime('%B %Y')
    peak_transcript_month = transcript_score.idxmax().strftime('%B %Y')
    peak_filing_month = filing_score.idxmax().strftime('%B %Y')

    news_narrative_counts = news_df['Label'].value_counts()
    transcript_narrative_counts = transcripts_df['Label'].value_counts()
    filing_narrative_counts = filings_df['Label'].value_counts()

    top_news_narrative = news_narrative_counts.index[0]
    top_transcript_narrative = transcript_narrative_counts.index[0]
    top_filing_narrative = filing_narrative_counts.index[0]

    total_news_mentions = len(news_df)
    total_transcript_mentions = len(transcripts_df)
    total_filing_mentions = len(filings_df)

    news_peaks = pd.Series(news_score).nlargest(3)
    transcript_peaks = pd.Series(transcript_score).nlargest(3)
    filing_peaks = pd.Series(filing_score).nlargest(3)

    avg_lag_days = []
    for news_date in news_peaks.index:
        closest_filing_date = min(filing_peaks.index, key=lambda x: abs((x - news_date).days))
        lag_days = (closest_filing_date - news_date).days
        avg_lag_days.append(lag_days)
    avg_lag = np.mean(avg_lag_days)

    return {
        "peak_news_month": peak_news_month,
        "peak_transcript_month": peak_transcript_month,
        "peak_filing_month": peak_filing_month,
        "top_news_narrative": top_news_narrative,
        "top_transcript_narrative": top_transcript_narrative,
        "top_filing_narrative": top_filing_narrative,
        "total_news_mentions": total_news_mentions,
        "total_transcript_mentions": total_transcript_mentions,
        "total_filing_mentions": total_filing_mentions,
        "avg_lag_days": int(avg_lag)
    }


def create_source_summary(news_df, transcripts_df, filings_df):
    """Create a summary dataframe of the dataset sizes"""
    source_summary = pd.DataFrame({
        'Source Type': ['News Media', 'Earnings Calls', 'SEC Filings'],
        'Record Count': [len(news_df), len(transcripts_df), len(filings_df)],
        'Date Range': [
            f"{news_df['Date'].min().strftime('%Y-%m-%d')} to {news_df['Date'].max().strftime('%Y-%m-%d')}",
            f"{transcripts_df['Date'].min().strftime('%Y-%m-%d')} to {transcripts_df['Date'].max().strftime('%Y-%m-%d')}",
            f"{filings_df['Date'].min().strftime('%Y-%m-%d')} to {filings_df['Date'].max().strftime('%Y-%m-%d')}"
        ],
        'Unique Narratives': [
            news_df['Label'].nunique(),
            transcripts_df['Label'].nunique(),
            filings_df['Label'].nunique()
        ]
    })
    return source_summary


def display_sample_data(news_df, transcripts_df, filings_df):
    """Display sample data from each source"""
    print("\n======= SAMPLE NEWS NARRATIVES =======")
    display(news_df[['Date', 'Headline', 'Label', 'Chunk Text']].head(3))

    print("\n======= SAMPLE EARNINGS CALL NARRATIVES =======")
    display(transcripts_df[['Date', 'Headline', 'Label', 'Chunk Text']].head(3))

    print("\n======= SAMPLE SEC FILING NARRATIVES =======")
    display(filings_df[['Date', 'Headline', 'Label', 'Chunk Text']].head(3))