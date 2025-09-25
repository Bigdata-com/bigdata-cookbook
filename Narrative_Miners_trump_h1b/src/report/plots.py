import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_enhanced_interactive_chart(df_narrative, df_citation, df_top3, df_companies, narrative_dates):
    """
    Create an enhanced interactive line chart with improved visuals and functionality.
    """
    
    # Create subplots with secondary y-axis for better scaling
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Main Narrative Trends', 'Trump-specific Activity'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}]]
    )
    
    # Prepare Trump data
    trump_citations = [len(df_citation[df_citation['Date'] == date]) for date in narrative_dates]
    trump_narrative = [len(df_narrative[df_narrative['Date'] == date]) for date in narrative_dates]
    
    # Add Trump data to bottom subplot
    fig.add_trace(
        go.Scatter(
            x=narrative_dates,
            y=trump_citations,
            mode='lines+markers',
            name='Trump Citations',
            line=dict(color='#e74c3c', width=3, dash='solid'),
            marker=dict(size=8, symbol='circle'),
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.1)',
            hovertemplate='<b>Trump Citations</b><br>' +
                          'Date: %{x|%Y-%m-%d}<br>' +
                          'Count: %{y}<br>' +
                          '<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=narrative_dates,
            y=trump_narrative,
            mode='lines+markers',
            name='Trump Narrative',
            line=dict(color='#3498db', width=3, dash='dot'),
            marker=dict(size=8, symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)',
            hovertemplate='<b>Trump Narrative</b><br>' +
                          'Date: %{x|%Y-%m-%d}<br>' +
                          'Count: %{y}<br>' +
                          '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Enhanced color palette with better contrast
    colors = [
        '#f39c12',  # Orange - Amazon
        '#27ae60',  # Green - Wipro
        '#9b59b6',  # Purple - Microsoft
        '#e67e22',  # Dark Orange - Alphabet
        '#2c3e50',  # Dark Blue - Infosys
        '#16a085',  # Teal - TCS
        '#c0392b',  # Dark Red
        '#8e44ad'   # Dark Purple
    ]
    
    # Symbols for better differentiation
    symbols = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'cross', 'x', 'star']
    
    # Get entities and sort by average occurrence for better legend order
    entities = df_top3['Entity'].unique()
    entity_avg = {}
    for entity in entities:
        counts = [len(df_top3[(df_top3['Date'] == date) & (df_top3['Entity'] == entity)]) for date in narrative_dates]
        entity_avg[entity] = np.mean(counts)
    
    entities_sorted = sorted(entities, key=lambda x: entity_avg[x], reverse=True)
    
    # Add company traces to main subplot
    for i, entity in enumerate(entities_sorted):
        entity_counts = []
        hover_texts = []
        dates_with_data = []
        
        for date in narrative_dates:
            count = len(df_top3[(df_top3['Date'] == date) & (df_top3['Entity'] == entity)])
            entity_counts.append(count)
            dates_with_data.append(date)
            
            # Enhanced hover information
            company_data = df_companies[(df_companies['Date'] == date) & (df_companies['Entity'] == entity)]
            
            if len(company_data) > 0:
                enhanced_key_points = company_data['enhanced_key_points'].iloc[0]
                enhanced_summary = company_data['enhanced_summary'].iloc[0]
                
                # Better formatting for hover text
                if isinstance(enhanced_key_points, list) and enhanced_key_points:
                    key_points_text = '<br>📌 '.join([str(point)[:100] + '...' if len(str(point)) > 100 else str(point) 
                                                     for point in enhanced_key_points if point])
                    if key_points_text:
                        key_points_text = '📌 ' + key_points_text
                else:
                    key_points_text = "No key points available"
                
                # Add summary preview
                summary_preview = str(enhanced_summary)[:150] + '...' if len(str(enhanced_summary)) > 150 else str(enhanced_summary)
                
                hover_text = f"<b>Summary:</b><br>{summary_preview}<br><br><b>Key Points:</b><br>{key_points_text}"
            else:
                hover_text = "No data available for this date"
            
            hover_texts.append(hover_text)
        
        # Add trend line
        fig.add_trace(
            go.Scatter(
                x=dates_with_data,
                y=entity_counts,
                mode='lines+markers',
                name=f'{entity} (avg: {entity_avg[entity]:.1f})',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8, symbol=symbols[i % len(symbols)], 
                           line=dict(width=2, color='white')),
                customdata=hover_texts,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Date: %{x|%Y-%m-%d}<br>' +
                              'Occurrences: %{y}<br><br>' +
                              '%{customdata}<br>' +
                              '<extra></extra>',
                connectgaps=True
            ),
            row=1, col=1
        )
        
        # Add smoothed trend line
        if len(entity_counts) > 2:
            from scipy.signal import savgol_filter
            try:
                smoothed = savgol_filter(entity_counts, window_length=min(5, len(entity_counts)), polyorder=2)
                fig.add_trace(
                    go.Scatter(
                        x=dates_with_data,
                        y=smoothed,
                        mode='lines',
                        name=f'{entity} (trend)',
                        line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                        opacity=0.6,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
            except:
                pass  # Skip if smoothing fails
    
    # Add annotations for peak dates
    max_dates = {}
    for entity in entities_sorted[:3]:  # Only for top 3 entities
        counts = [len(df_top3[(df_top3['Date'] == date) & (df_top3['Entity'] == entity)]) for date in narrative_dates]
        max_idx = np.argmax(counts)
        if counts[max_idx] > 0:
            max_dates[entity] = (narrative_dates[max_idx], counts[max_idx])
    
    for entity, (date, count) in max_dates.items():
        fig.add_annotation(
            x=date, y=count,
            text=f"Peak: {entity}<br>{count}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="rgba(0,0,0,0.5)",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=10),
            row=1, col=1
        )
    
    # Enhanced layout
    fig.update_layout(
        title={
            'text': '<b>H-1B Visa Narrative Analysis Dashboard</b><br><sub>Corporate Response and Media Coverage Timeline</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'},
            'y': 0.95
        },
        width=1400,
        height=900,
        template='plotly_white',
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(r=200, t=100, b=80, l=80),
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        tickformat='%m/%d',
        tickangle=45,
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        tickformat='%m/%d',
        tickangle=45,
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Corporate Mentions",
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Trump Activity",
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        row=2, col=1
    )
    
    # Add range selector
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=3, label="3d", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),
            type="date"
        )
    )
    
    return fig
