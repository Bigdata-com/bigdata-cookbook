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
                import ast
                
                # Try to parse enhanced_key_points if it's a string representation of a list
                key_points_list = None
                
                if isinstance(enhanced_key_points, list):
                    key_points_list = enhanced_key_points
                elif isinstance(enhanced_key_points, str) and enhanced_key_points.strip():
                    # Check if it looks like a list representation
                    if enhanced_key_points.strip().startswith('[') and enhanced_key_points.strip().endswith(']'):
                        try:
                            key_points_list = ast.literal_eval(enhanced_key_points.strip())
                        except (ValueError, SyntaxError):
                            # If parsing fails, treat as single string
                            key_points_list = [enhanced_key_points.strip()]
                    else:
                        key_points_list = [enhanced_key_points.strip()]
                
                # Adjust summary and key points to ensure compact hover text
                if key_points_list:
                    # Limit characters to reduce width
                    valid_points = [str(point).strip() for point in key_points_list if point and str(point).strip()]
                    if valid_points:
                        # Shorten each point to 20 characters
                        key_points_text = '<br>• '.join([
                            point[:20] + '...' if len(point) > 20 else point 
                            for point in valid_points[:2]
                        ])
                        key_points_text = '• ' + key_points_text
                        if len(valid_points) > 2:
                            key_points_text += f'<br>• ... and {len(valid_points) - 2} more'
                    else:
                        key_points_text = "No highlights available"
                else:
                    key_points_text = "No highlights available"
                
                # Shorten summary to 30 characters
                summary_preview = str(enhanced_summary)[:30] + '...' if len(str(enhanced_summary)) > 30 else str(enhanced_summary)
                
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
                name=f'{entity}',
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


def create_company_document_coverage_chart(company_stats, relative_overall=False):
    """
    Create an interactive line chart showing daily document coverage for companies.
    Shows the number of unique documents mentioning each company per day.
    
    Args:
        company_stats: DataFrame with company statistics by date
        relative_overall: If True, plot percentages instead of absolute numbers
    """
    
    # Get unique dates and companies
    dates = sorted(company_stats['Date'].unique())
    companies = company_stats['Company'].unique()
    
    # Enhanced color palette
    colors = [
        '#f39c12',  # Orange
        '#27ae60',  # Green
        '#9b59b6',  # Purple
        '#e67e22',  # Dark Orange
        '#2c3e50',  # Dark Blue
        '#16a085',  # Teal
        '#c0392b',  # Dark Red
        '#8e44ad',  # Dark Purple
        '#f1c40f',  # Yellow
        '#e74c3c'   # Red
    ]
    
    # Symbols for better differentiation
    symbols = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'cross', 'x', 'star', 'pentagon', 'hexagon']
    
    # Create figure
    fig = go.Figure()
    
    # Calculate average document coverage for sorting
    company_avg = {}
    for company in companies:
        company_data = company_stats[company_stats['Company'] == company]
        if relative_overall:
            avg_value = company_data['Percentage_Documents'].mean()
        else:
            avg_value = company_data['Unique_Documents'].mean()
        company_avg[company] = avg_value
    
    # Sort companies by average document coverage
    companies_sorted = sorted(companies, key=lambda x: company_avg[x], reverse=True)
    
    # Add traces for each company
    for i, company in enumerate(companies_sorted):
        company_data = company_stats[company_stats['Company'] == company].sort_values('Date')
        
        # Choose what to plot based on relative_overall parameter
        if relative_overall:
            y_values = company_data['Percentage_Documents']
            hover_metric = 'Percentage: %{y:.2f}%<br>'
            hover_extra = f'Unique Documents: %{{customdata}}<br>'
            customdata_values = company_data['Unique_Documents']
        else:
            y_values = company_data['Unique_Documents']
            hover_metric = 'Unique Documents: %{y}<br>'
            hover_extra = 'Percentage: %{customdata:.2f}%<br>'
            customdata_values = company_data['Percentage_Documents']
        
        fig.add_trace(
            go.Scatter(
                x=company_data['Date'],
                y=y_values,
                mode='lines+markers',
                name=f'{company}',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8, symbol=symbols[i % len(symbols)], 
                           line=dict(width=2, color='white')),
                customdata=customdata_values,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Date: %{x|%Y-%m-%d}<br>' +
                              hover_metric +
                              hover_extra +
                              '<extra></extra>',
                connectgaps=True
            )
        )
    
    # Update layout
    title_text = '<b>News Document Coverage by Company</b><br><sub>'
    if relative_overall:
        title_text += 'Daily Percentage of Documents Mentioning Each Company</sub>'
    else:
        title_text += 'Daily Number of Unique Documents Mentioning Each Company</sub>'
    
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'},
            'y': 0.95
        },
        width=1400,
        height=600,
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
        tickangle=45
    )
    
    y_axis_title = "Percentage (%)" if relative_overall else "Number of Unique Documents"
    fig.update_yaxes(
        title_text=y_axis_title,
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)'
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


def create_entity_temporal_coverage_chart(df):
    """
    Create an interactive line chart showing temporal evolution of entity coverage.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns:
        - Date: date of the coverage
        - Entity: name of the entity
        - Entity Type: type of entity (COMP, PEOP, ORGA)
        - Novel Daily Summary: summary text for that date
        - Highlights: key points for that date (fallback to Daily Highlights or enhanced_key_points)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    import ast
    
    # Ensure Date is datetime
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get all unique dates sorted
    all_dates = sorted(df['Date'].unique())
    
    # Get all unique entities
    entities = df['Entity'].unique()
    
    # Calculate average occurrences for each entity to sort them
    entity_avg = {}
    for entity in entities:
        entity_data = df[df['Entity'] == entity]
        daily_counts = entity_data.groupby('Date').size()
        entity_avg[entity] = daily_counts.mean()
    
    # Sort entities by average occurrence (descending)
    entities_sorted = sorted(entities, key=lambda x: entity_avg[x], reverse=True)
    
    # Sort entities by total occurrences before adding traces
    entities_sorted_by_occurrences = sorted(entities_sorted, key=lambda entity: df[df['Entity'] == entity].shape[0], reverse=True)

    # Define color schemes for different entity types
    color_scheme = {
        'COMP': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],  # Blues/teals for companies
        'PEOP': ['#E63946', '#F77F00', '#D62828', '#F4A261', '#E76F51'],  # Reds/oranges for people
        'ORGA': ['#06A77D', '#118AB2', '#073B4C', '#3A6351', '#06D6A0']   # Greens/cyans for organizations
    }
    
    # Marker symbols
    symbols = ['circle', 'diamond', 'square', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']
    
    # Create figure
    fig = go.Figure()
    
    # Add separate traces to figure for each entity type
    for i, entity in enumerate(entities_sorted_by_occurrences):
        # Get entity type
        entity_type = df[df['Entity'] == entity]['Entity Type'].iloc[0]
        
        # Get total occurrences for the entity
        total_count = df[df['Entity'] == entity].shape[0]
        
        # Skip entity if total count is zero
        if total_count == 0:
            continue
        
        # Select color based on entity type
        type_colors = color_scheme.get(entity_type, color_scheme['COMP'])
        color_idx = sum(1 for e in entities_sorted_by_occurrences[:i] if df[df['Entity'] == e]['Entity Type'].iloc[0] == entity_type)
        color = type_colors[color_idx % len(type_colors)]
        
        entity_counts = []
        hover_texts = []
        dates_with_data = []
        
        for date in all_dates:
            # Count occurrences for this entity on this date
            date_data = df[(df['Date'] == date) & (df['Entity'] == entity)]
            count = len(date_data)
            
            entity_counts.append(count)
            dates_with_data.append(date)
            
            # Build hover text
            if len(date_data) > 0:
                # Get Novel Daily Summary for summary
                summary = date_data['Novel Daily Summary'].iloc[0] if 'Novel Daily Summary' in date_data.columns else ""
                
                # Get highlights - try Highlights first, then fallback to old column names
                highlights = None
                if 'Highlights' in date_data.columns and pd.notna(date_data['Highlights'].iloc[0]):
                    highlights = date_data['Highlights'].iloc[0]
                elif 'Daily Highlights' in date_data.columns and pd.notna(date_data['Daily Highlights'].iloc[0]):
                    highlights = date_data['Daily Highlights'].iloc[0]
                elif 'enhanced_key_points' in date_data.columns and pd.notna(date_data['enhanced_key_points'].iloc[0]):
                    highlights = date_data['enhanced_key_points'].iloc[0]
                
                # Parse highlights if it's a list or string representation of a list
                key_points_list = None
                
                if highlights is not None:
                    if isinstance(highlights, list):
                        key_points_list = highlights
                    elif isinstance(highlights, str) and highlights.strip():
                        # Check if it looks like a list representation
                        if highlights.strip().startswith('[') and highlights.strip().endswith(']'):
                            try:
                                key_points_list = ast.literal_eval(highlights.strip())
                            except (ValueError, SyntaxError):
                                # If parsing fails, treat as single string
                                key_points_list = [highlights.strip()]
                        else:
                            key_points_list = [highlights.strip()]
                
                # Adjust summary and key points to ensure compact hover text
                if key_points_list:
                    # Limit characters to reduce width
                    valid_points = [str(point).strip() for point in key_points_list if point and str(point).strip()]
                    if valid_points:
                        # Shorten each point to 20 characters
                        key_points_text = '<br>• '.join([
                            point[:20] + '...' if len(point) > 20 else point 
                            for point in valid_points[:2]
                        ])
                        key_points_text = '• ' + key_points_text
                        if len(valid_points) > 2:
                            key_points_text += f'<br>• ... and {len(valid_points) - 2} more'
                    else:
                        key_points_text = "No highlights available"
                else:
                    key_points_text = "No highlights available"
                
                # Format summary to wrap at 100 characters, breaking at spaces
                summary_words = str(summary).split()
                summary_preview = ''
                summary_line = ''
                for word in summary_words:
                    if len(summary_line) + len(word) + 1 <= 100:
                        summary_line += (' ' + word if summary_line else word)
                    else:
                        summary_preview += (summary_line + '<br>')
                        summary_line = word
                summary_preview += summary_line

                # Correct bullet list formatting to wrap text properly within a single bullet point, avoiding extra bullets
                key_points_text = ''
                for point in valid_points:
                    wrapped_point = ' • '  # Single bullet per point
                    char_count = 0
                    for word in point.split():
                        if char_count + len(word) + 1 > 100:  # Wrap line if it exceeds 100 characters
                            wrapped_point += '<br>'
                            char_count = 0
                        wrapped_point += (' ' + word)
                        char_count += len(word) + 1
                    key_points_text += (wrapped_point + '<br>')
                
                hover_text = (
                    f"<b>Summary:</b><br>{summary_preview}<br><br>"
                    f"<b>Highlights:</b><br>{key_points_text}"
                )
            else:
                hover_text = "No data available for this date"
            
            hover_texts.append(hover_text)
        
        # Determine entity type name for legend grouping
        entity_type_name = {
            'PEOP': 'People',
            'COMP': 'Companies',
            'ORGA': 'Organizations'
        }.get(entity_type, 'Entities')
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=dates_with_data,
                y=entity_counts,
                mode='lines+markers',
                name=f'{entity} ({total_count})',
                legendgroup=entity_type_name,
                legendgrouptitle_text=entity_type_name,
                line=dict(color=color, width=2.5),
                marker=dict(
                    size=8, 
                    symbol=symbols[i % len(symbols)],
                    color=color,
                    line=dict(width=1.5, color='white')
                ),
                customdata=hover_texts,
                hovertemplate=(
                    '<b>%{fullData.name}</b><br>' +
                    'Date: %{x|%Y-%m-%d}<br>' +
                    'Occurrences: %{y}<br><br>' +
                    '%{customdata}<br>' +
                    '<extra></extra>'
                ),
                connectgaps=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Entity Coverage Timeline',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        xaxis_title='Date',
        yaxis_title='Number of Occurrences',
        hovermode='closest',
        height=700,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12, color='#2c3e50'),
        margin=dict(l=80, r=250, t=100, b=80)
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        tickformat='%Y-%m-%d',
        tickangle=45,
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.3)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='rgba(128,128,128,0.3)'
    )
    
    # Add range selector
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='rgba(255,255,255,0.9)',
            activecolor='rgba(46, 134, 171, 0.3)',
            font=dict(color='#2c3e50')
        ),
        rangeslider=dict(visible=False),
        type="date"
    )
    
    return fig
