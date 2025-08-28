import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display, HTML


def add_line_breaks(text, max_length=80):
    """
    Add line breaks to text for better readability in hover
    
    Parameters:
    -----------
    text : str
        Text to add line breaks to
    max_length : int
        Maximum length per line
        
    Returns:
    --------
    str
        Text with line breaks added
    """
    if pd.isna(text) or text == '':
        return text
    words = str(text).split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return '<br>'.join(lines)


def get_company_details(df, company, sector=None):
    """
    Get company details from dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing company data
    company : str
        Company name
    sector : str, optional
        Sector filter
        
    Returns:
    --------
    dict
        Dictionary with company details
    """
    if sector:
        company_data = df[(df['Sector'] == sector) & (df['Company'] == company)].copy()
    else:
        company_data = df[df['Company'] == company].copy()
        
    if company_data.empty:
        return {
            'Company': company,
            'Headline': '',
            'Motivation': '',
            'Text': ''
        }
        
    company_data['Date'] = pd.to_datetime(company_data['Date'])
    date_counts = company_data.groupby('Date').size()
    max_date = date_counts.idxmax()
    max_date_data = company_data[company_data['Date'] == max_date]
    last_row = max_date_data.iloc[-1]

    return {
        'Company': last_row['Company'],
        'Headline': last_row['Headline'],
        'Motivation': last_row['Motivation'],
        'Text': last_row['Quote']
    }


def plot_top_companies_by_sector(df, min_companies=1, title_suffix="", top_sectors=10, interactive=True):
    """
    Plot top 5 companies by occurrences for each sector

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns 'Sector', 'Company', 'Date', 'Headline', 'Motivation', 'Quote'
    min_companies : int
        Minimum number of companies required in a sector to include it in the plot
    title_suffix : str
        Additional text to add to the plot title
    top_sectors : int
        Number of top sectors to show (ranked by total occurrences)
    interactive : bool
        If True, create interactive plot. If False, create static plot
    """
    # Count occurrences by Sector and Company
    sector_company_counts = df.groupby(['Sector', 'Company']).size().reset_index(name='count')

    # Filter sectors that have at least min_companies
    companies_per_sector = sector_company_counts.groupby('Sector')['Company'].nunique()
    valid_sectors = companies_per_sector[companies_per_sector >= min_companies].index

    # Filter data to include only valid sectors
    sector_company_counts = sector_company_counts[sector_company_counts['Sector'].isin(valid_sectors)]

    # Get top 5 companies per sector
    top_companies_per_sector = []
    for sector in sector_company_counts['Sector'].unique():
        sector_data = sector_company_counts[sector_company_counts['Sector'] == sector]
        top_5 = sector_data.nlargest(5, 'count')
        top_companies_per_sector.append(top_5)

    top_companies_per_sector = pd.concat(top_companies_per_sector, ignore_index=True)

    # Rank sectors by total occurrences and get top sectors
    sector_ranking = (top_companies_per_sector.groupby('Sector')['count']
                      .sum()
                      .sort_values(ascending=False)
                      .index)[:top_sectors]

    if interactive:
        # Create interactive plotly version
        n_cols = 4
        n_rows = (len(sector_ranking) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"{sector} <br> Sector" for sector in sector_ranking],
            vertical_spacing=0.4,
            horizontal_spacing=0.1,
            specs=[[{"type": "bar"}] * n_cols for _ in range(n_rows)]
        )

        # Create bar charts for each sector
        for i, sector in enumerate(sector_ranking):
            row = i // n_cols + 1
            col = (i % n_cols) + 1

            # Get top 5 companies for this sector
            sector_data = top_companies_per_sector[top_companies_per_sector['Sector'] == sector]

            # Create custom hover text with line breaks
            hover_texts = []
            for _, row_data in sector_data.iterrows():
                details = get_company_details(df, row_data['Company'], row_data['Sector'])
                hover_text = f"<b>Company:</b> {details['Company']}<br><br><b>Headline:</b> {add_line_breaks(details['Headline'])}<br><br><b>Motivation:</b> {add_line_breaks(details['Motivation'])}<br><br><b>Text:</b> {add_line_breaks(details['Text'])}"
                hover_texts.append(hover_text)

            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=sector_data['Company'],
                    y=sector_data['count'],
                    name=f"{sector} Sector",
                    showlegend=False,
                    hovertext=hover_texts,
                    hoverinfo="text",
                    marker_color='#3366CC',
                    hoverlabel=dict(font=dict(size=16))
                ),
                row=row, col=col
            )

            # Update x-axis for each subplot
            fig.update_xaxes(
                tickfont=dict(size=18),
                gridwidth=0.45,
                tickangle=70,
                gridcolor='LightGray',
                row=row, col=col
            )

            # Update y-axis for each subplot
            fig.update_yaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='LightGray',
                row=row, col=col
            )

        # Update layout
        fig.update_layout(
            height=550 * n_rows,
            title={
                'text': f"Top Companies by Sector {title_suffix}",
                'x': 0.5,
                'y': 0.98,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'family': 'Arial Black'}
            },
            margin=dict(t=120)
        )

        pyo.iplot(fig)

    else:
        # Create static matplotlib version
        n_cols = 4
        n_rows = (len(sector_ranking) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5.5 * n_rows))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for i, sector in enumerate(sector_ranking):
            ax = axes_flat[i]
            
            # Get top 5 companies for this sector
            sector_data = top_companies_per_sector[top_companies_per_sector['Sector'] == sector]
            
            # Create bar chart
            bars = ax.bar(sector_data['Company'], sector_data['count'], color='#3366CC')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
            
            # Set title
            ax.set_title(f"{sector}\nSector", fontsize=14, fontweight='bold', pad=20)
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=70, labelsize=12)
            ax.tick_params(axis='y', labelsize=10)
            
            # Add grid
            ax.grid(True, alpha=0.3, color='lightgray')
            ax.set_axisbelow(True)
        
        # Hide unused subplots
        for i in range(len(sector_ranking), len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        
        # Add main title positioned high like in Plotly version (after tight_layout)
        fig.suptitle(f"Top Companies by Sector {title_suffix}", fontsize=24, fontweight='bold', y=1.25)
        
        # Adjust layout to make room for the high title
        plt.subplots_adjust(top=0.88)
        plt.show()


def identify_basket_and_plot_confidence(df_pos, df_neg, basket_size=30, sector_column='Sector', 
                                        theme_name='Theme', positive_label='Positive', negative_label='Negative',
                                        interactive=True):
    """
    Identify basket of companies and plot confidence analysis from positive and negative dataframes

    Parameters:
    -----------
    df_pos : pandas.DataFrame
        DataFrame with positive exposure data
    df_neg : pandas.DataFrame
        DataFrame with negative exposure data
    basket_size : int
        Number of top companies to include in the basket
    sector_column : str
        Column to use for sector information ('Sector' or 'Industry')
    theme_name : str
        Name of the theme being analyzed (e.g., 'Liquid Cooling', 'Pricing Power')
    positive_label : str
        Label for positive exposure (e.g., 'Strong Pricing Power', 'Liquid Cooling Benefits')
    negative_label : str
        Label for negative exposure (e.g., 'Weak Pricing Power', 'Liquid Cooling Challenges')
    interactive : bool
        If True, create interactive plot. If False, create static plot
        
    Returns:
    --------
    pandas.DataFrame
        Top companies basket with confidence metrics
    """
    # Count occurrences for positive and negative dataframes
    pos_counts = df_pos.groupby('Company').size().reset_index(name='positive_exp')
    neg_counts = df_neg.groupby('Company').size().reset_index(name='negative_exp')

    # Get all unique companies
    all_companies = set(pos_counts['Company'].tolist() + neg_counts['Company'].tolist())

    # Create complete dataframe with all companies
    companies_df = pd.DataFrame({'Company': list(all_companies)})
    companies_df = companies_df.merge(pos_counts, on='Company', how='left')
    companies_df = companies_df.merge(neg_counts, on='Company', how='left')
    companies_df = companies_df.fillna(0)

    # Calculate total exposure
    companies_df['total_exposure'] = companies_df['positive_exp'] + companies_df['negative_exp']

    # Add sector information from one of the dataframes
    sector_info = pd.concat([df_pos[['Company', sector_column]], df_neg[['Company', sector_column]]]).drop_duplicates()
    companies_df = companies_df.merge(sector_info, on='Company', how='left')

    # Calculate percentages
    companies_df['positive_exp_pct'] = (companies_df['positive_exp'] * 100 / companies_df['total_exposure']).fillna(0)
    companies_df['negative_exp_pct'] = (companies_df['negative_exp'] * 100 / companies_df['total_exposure']).fillna(0)

    # Get details for positive exposure
    pos_details = []
    for company in companies_df['Company']:
        if company in df_pos['Company'].values:
            details = get_company_details(df_pos, company)
            pos_details.append({
                'Company': company,
                'headline_positive_exp': details['Headline'],
                'text_positive_exp': details['Text'],
                'motivation_positive_exp': details['Motivation']
            })
        else:
            pos_details.append({
                'Company': company,
                'headline_positive_exp': '',
                'text_positive_exp': '',
                'motivation_positive_exp': ''
            })

    pos_details_df = pd.DataFrame(pos_details)
    companies_df = companies_df.merge(pos_details_df, on='Company', how='left')

    # Get details for negative exposure
    neg_details = []
    for company in companies_df['Company']:
        if company in df_neg['Company'].values:
            details = get_company_details(df_neg, company)
            neg_details.append({
                'Company': company,
                'headline_negative_exp': details['Headline'],
                'text_negative_exp': details['Text'],
                'motivation_negative_exp': details['Motivation']
            })
        else:
            neg_details.append({
                'Company': company,
                'headline_negative_exp': '',
                'text_negative_exp': '',
                'motivation_negative_exp': ''
            })

    neg_details_df = pd.DataFrame(neg_details)
    companies_df = companies_df.merge(neg_details_df, on='Company', how='left')

    # Select top companies by total exposure
    top_exposed = companies_df.nlargest(basket_size, 'total_exposure').sort_values('total_exposure').reset_index(drop=True)

    # Plot confidence bar chart
    plot_confidence_bar_chart(top_exposed, theme_name, ['positive_exp', 'negative_exp'],
                              [positive_label, negative_label], 
                              confidence_analysis_title='Confidence Analysis',
                              chart_subtitle='Analysis by Company',
                              interactive=interactive)

    return top_exposed


def plot_confidence_bar_chart(basket_df, title_theme, df_names, legend_labels, 
                              confidence_analysis_title='Confidence Analysis', 
                              chart_subtitle='Analysis by Company', interactive=True):
    """
    Create a stacked bar chart showing relative exposure to themes
    
    Parameters:
    -----------
    basket_df : pandas.DataFrame
        DataFrame with company basket data
    title_theme : str
        Title theme for the chart
    df_names : list
        List of data column names
    legend_labels : list
        List of legend labels
    confidence_analysis_title : str
        Title for the confidence analysis section (default: 'Confidence Analysis')
    chart_subtitle : str
        Subtitle for the main chart (default: 'Analysis by Company')
    interactive : bool
        If True, create interactive plot. If False, create static plot
    """
    df = basket_df.copy()

    if interactive:
        # Create interactive plotly version
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            subplot_titles=(confidence_analysis_title, "Total Volume"),
            horizontal_spacing=0.15
        )

        fig.update_layout(
            barmode='stack',
            title=f'{title_theme} {confidence_analysis_title} {chart_subtitle}',
            xaxis=dict(range=[0, 100], dtick=10),
            yaxis=dict(title='Company Name'),
            template='plotly_white',
            showlegend=True,
        )

        fig.update_layout(yaxis=dict(
            tickfont=dict(
                size=18-(2*len(basket_df)//10))
        ))

        fig.update_yaxes(showticklabels=True, row=1, col=1)

        colors = ['green', 'red', 'grey', 'orange']

        for i, (name, leg_label, color) in enumerate(zip(df_names, legend_labels, colors[:len(df_names)])):
            base_val = [0]*len(df['Company']) if i == 0 else df[[f'{name}_pct' for name in df_names[:i]]].abs().sum(axis=1)
            text_col = f'text_{name}'

            # Create custom hover text with text and motivation, adding dynamic line breaks for better readability
            hover_text = [
                f"<b>{leg_label}:</b> {round(row[f'{name}_pct'], 2)}%<br><br><b>Company:</b> {row['Company']}<br><br><b>Headline:</b> {add_line_breaks(row[f'headline_{name}'])}<br><br><b>Motivation:</b> {add_line_breaks(row[f'motivation_{name}'])}<br><br><b>Text:</b> {add_line_breaks(row[text_col])}"
                for index, row in df.iterrows()
            ]

            fig.add_trace(go.Bar(
                y=df['Company'],
                x=df[f'{name}_pct'],
                name=leg_label,
                orientation='h',
                marker=dict(color=color),
                hovertemplate='%{hovertext}<extra></extra>',
                text=df[f'{name}_pct'].apply(lambda x: f'{x:.2f}%'),
                hovertext=hover_text,
                base=base_val,
                textposition='inside',
                insidetextanchor='start',
            ), row=1, col=1)

        # Add vertical line at 50%
        fig.add_shape(
            type='line',
            x0=50,
            x1=50,
            y0=-0.5,
            y1=len(df['Company']) - 0.5,
            line=dict(color='black', width=3, dash='dash'),
            row=1, col=1
        )

        fig.add_trace(go.Bar(
            y=df['Company'],
            x=df['total_exposure'],
            name='Total Volume',
            orientation='h',
            marker=dict(color='blue'),
            hovertemplate='%{x}<extra></extra>',
            text=df['total_exposure'].apply(lambda x: f'{x}'),
            textposition='inside',
            textangle=0,
            showlegend=False
        ), row=1, col=2)

        # Update layout
        fig.update_layout(
            height=900,
            width=1400,
            title={
                'text': f"{title_theme} Watchlist {confidence_analysis_title}",
                'x': 0.5,
                'y': 0.98,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'family': 'Arial Black'}
            }
        )

        fig.update_yaxes(showticklabels=False, row=1, col=2)

        fig.show()

    else:
        # Create static matplotlib version
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={'width_ratios': [0.7, 0.3]})
        
        # Left subplot: Confidence Analysis
        colors = ['green', 'red', 'grey', 'orange']
        y_pos = np.arange(len(df['Company']))
        
        # Create stacked horizontal bar chart
        left_vals = np.zeros(len(df['Company']))
        
        for i, (name, leg_label, color) in enumerate(zip(df_names, legend_labels, colors[:len(df_names)])):
            values = df[f'{name}_pct'].values
            bars = ax1.barh(y_pos, values, left=left_vals, color=color, label=leg_label)
            
            # Add percentage labels
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 5:  # Only show label if bar is wide enough
                    ax1.text(left_vals[j] + val/2, bar.get_y() + bar.get_height()/2,
                            f'{val:.1f}%', ha='center', va='center', fontsize=8, fontweight='bold')
            
            left_vals += values

        # Add vertical line at 50%
        ax1.axvline(x=50, color='black', linestyle='--', linewidth=3)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df['Company'], fontsize=12)
        ax1.set_xlabel('Percentage', fontsize=12)
        ax1.set_title(confidence_analysis_title, fontsize=16, fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.legend(loc='upper right')

        # Right subplot: Total Volume
        bars2 = ax2.barh(y_pos, df['total_exposure'], color='blue')
        
        # Add value labels
        for bar, val in zip(bars2, df['total_exposure']):
            if val > 0:
                ax2.text(val/2, bar.get_y() + bar.get_height()/2,
                        f'{int(val)}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])  # Hide y-axis labels for right subplot
        ax2.set_xlabel('Total Volume', fontsize=12)
        ax2.set_title('Total Volume', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Add main title positioned high like in Plotly version
        fig.suptitle(f"{title_theme} Watchlist {confidence_analysis_title}", fontsize=24, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Leave more space for the high title
        plt.show()


def get_day_with_max_occurrences_for_exposure(df_week_data):
    """
    Get details from the day with most occurrences for this specific exposure type
    
    Parameters:
    -----------
    df_week_data : pandas.DataFrame
        DataFrame with weekly data for specific exposure
        
    Returns:
    --------
    dict
        Dictionary with day details
    """
    if df_week_data.empty:
        return {
            'Headline': '',
            'Quote': '',
            'Motivation': '',
            'count': 0
        }

    # Count occurrences per day for this exposure type
    daily_counts = df_week_data.groupby('Date').size()
    max_day = daily_counts.idxmax()

    # Get all data from the day with most occurrences for this exposure
    max_day_data = df_week_data[df_week_data['Date'] == max_day].sort_values('Date')

    # Get the last occurrence of that day
    last_occurrence = max_day_data.iloc[-1]

    return {
        'Headline': str(last_occurrence['Headline']) if pd.notna(last_occurrence['Headline']) else '',
        'Quote': str(last_occurrence['Quote']) if pd.notna(last_occurrence['Quote']) else '',
        'Motivation': str(last_occurrence['Motivation']) if pd.notna(last_occurrence['Motivation']) else '',
        'count': len(df_week_data)  # Total count for the week
    }


def get_weekly_volume_with_separate_daily_details(df_pos, df_neg, df_names, sector_column='Sector', start_date=None, end_date=None, frequency='weekly'):
    """
    Get weekly or monthly volume with separate daily details for positive and negative exposures

    Parameters:
    -----------
    df_pos : pandas.DataFrame
        DataFrame with positive pricing power data
    df_neg : pandas.DataFrame
        DataFrame with negative pricing power data
    df_names : list
        List of names for the exposures ['positive_exp', 'negative_exp']
    sector_column : str
        Column to use for sector information
    start_date : str, optional
        Start date for analysis
    end_date : str, optional
        End date for analysis
    frequency : str
        Frequency for grouping data ('weekly' or 'monthly')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with weekly or monthly exposure data
    """
    # Create final data structure (weekly or monthly)
    period_data = []

    # Determine period grouping based on frequency
    if frequency == 'monthly':
        period_freq = 'M'
        period_label = 'Month'
    else:  # default to weekly
        period_freq = 'W-MON'
        period_label = 'Week'

    # Get all unique combinations of period, company, sector from both dataframes
    all_combinations = set()

    # Process positive exposure
    pos_details = {}
    if not df_pos.empty:
        df_pos_copy = df_pos.copy()
        df_pos_copy['Date'] = pd.to_datetime(df_pos_copy['Date'])

        if start_date and end_date:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            df_pos_copy = df_pos_copy[(df_pos_copy['Date'] >= start_date_dt) & (df_pos_copy['Date'] <= end_date_dt)]

        df_pos_copy['Period'] = df_pos_copy['Date'].dt.to_period(period_freq)

        # Get positive exposure details by period
        for (period, company, sector), group in df_pos_copy.groupby(['Period', 'Company', sector_column]):
            details = get_day_with_max_occurrences_for_exposure(group)
            pos_details[(period, company, sector)] = details
            all_combinations.add((period, company, sector))

    # Process negative exposure
    neg_details = {}
    if not df_neg.empty:
        df_neg_copy = df_neg.copy()
        df_neg_copy['Date'] = pd.to_datetime(df_neg_copy['Date'])

        if start_date and end_date:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            df_neg_copy = df_neg_copy[(df_neg_copy['Date'] >= start_date_dt) & (df_neg_copy['Date'] <= end_date_dt)]

        df_neg_copy['Period'] = df_neg_copy['Date'].dt.to_period(period_freq)

        # Get negative exposure details by period
        for (period, company, sector), group in df_neg_copy.groupby(['Period', 'Company', sector_column]):
            details = get_day_with_max_occurrences_for_exposure(group)
            neg_details[(period, company, sector)] = details
            all_combinations.add((period, company, sector))

    # Combine all data
    for period, company, sector in all_combinations:
        # Get positive details if available
        pos_detail = pos_details.get((period, company, sector), {
            'Headline': '', 'Quote': '', 'Motivation': '', 'count': 0
        })

        # Get negative details if available
        neg_detail = neg_details.get((period, company, sector), {
            'Headline': '', 'Quote': '', 'Motivation': '', 'count': 0
        })

        period_data.append({
            'Date': period.start_time,
            'Company': company,
            sector_column: sector,
            df_names[0]: pos_detail['count'],  # positive_exp
            df_names[1]: neg_detail['count'],  # negative_exp
            f'headline_{df_names[0]}': pos_detail['Headline'],
            f'text_{df_names[0]}': pos_detail['Quote'],
            f'motivation_{df_names[0]}': pos_detail['Motivation'],
            f'headline_{df_names[1]}': neg_detail['Headline'],
            f'text_{df_names[1]}': neg_detail['Quote'],
            f'motivation_{df_names[1]}': neg_detail['Motivation']
        })

    if not period_data:
        return pd.DataFrame()

    final_df = pd.DataFrame(period_data)

    # Calculate total exposure
    final_df['total_exposure'] = final_df[[f'{name}' for name in df_names]].abs().sum(axis=1)

    # Sort by date
    final_df = final_df.sort_values('Date').reset_index(drop=True)

    # Add zero values for missing periods to ensure every company appears in every period
    if not final_df.empty:
        # Create complete date range based on start_date and end_date
        if start_date and end_date:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
        else:
            # Fallback to min/max dates in data
            start_date_dt = final_df['Date'].min()
            end_date_dt = final_df['Date'].max()
        
        # Generate ALL periods in the date range
        if frequency == 'monthly':
            all_periods = pd.date_range(start=start_date_dt, end=end_date_dt, freq='MS').to_period('M')
        else:  # weekly
            all_periods = pd.date_range(start=start_date_dt, end=end_date_dt, freq='W-MON').to_period('W-MON')
        
        # Convert to start_time for consistency
        all_periods = [period.start_time for period in all_periods]

        # Create a mapping of company -> sector (each company belongs to one sector)
        company_sector_map = final_df.groupby('Company')[sector_column].first().to_dict()

        # Create all possible combinations of period and company
        missing_rows = []

        for period in all_periods:
            existing_companies_this_period = set(final_df[final_df['Date'] == period]['Company'].unique())
            all_companies = set(company_sector_map.keys())

            # Find companies missing for this period
            missing_companies = all_companies - existing_companies_this_period

            for company in missing_companies:
                sector = company_sector_map[company]

                # Create row with zero values
                zero_row = {
                    'Date': period,
                    'Company': company,
                    sector_column: sector,
                    df_names[0]: 0,  # positive_exp
                    df_names[1]: 0,  # negative_exp
                    f'headline_{df_names[0]}': '',
                    f'text_{df_names[0]}': '',
                    f'motivation_{df_names[0]}': '',
                    f'headline_{df_names[1]}': '',
                    f'text_{df_names[1]}': '',
                    f'motivation_{df_names[1]}': '',
                    'total_exposure': 0
                }
                missing_rows.append(zero_row)

        # Add missing rows to final_df
        if missing_rows:
            missing_df = pd.DataFrame(missing_rows)
            final_df = pd.concat([final_df, missing_df], ignore_index=True)

            # Sort again by date and company
            final_df = final_df.sort_values(['Date', 'Company']).reset_index(drop=True)

    return final_df


def track_basket_sectors_weekly_complete_hover_labeled(df, df_names, companies_list, sector_column, 
                                                      sectors_list=[], theme_name='Theme', 
                                                      positive_hover_label='Positive', negative_hover_label='Negative',
                                                      y_axis_label='Net Exposure', analysis_suffix='Analysis',
                                                      interactive=True, frequency='weekly'):
    """
    Track basket sectors with complete hover info with customizable labels

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with weekly or monthly exposure data
    df_names : list
        List of exposure names ['positive_exp', 'negative_exp']
    companies_list : list
        List of companies to track
    sector_column : str
        Column name for sector grouping
    sectors_list : list, optional
        Optional list of sectors to focus on
    theme_name : str
        Name of the theme being analyzed (e.g., 'Liquid Cooling', 'Pricing Power')
    positive_hover_label : str
        Label for positive exposure in hover text (e.g., 'Positive', 'Benefits')
    negative_hover_label : str
        Label for negative exposure in hover text (e.g., 'Negative', 'Challenges')
    y_axis_label : str
        Label for y-axis (e.g., 'Net Exposure', 'Net Pricing Power Exposure')
    analysis_suffix : str
        Suffix for analysis title (e.g., 'Analysis', 'Weekly Analysis', 'Monthly Analysis')
    interactive : bool
        If True, create interactive plot. If False, create static plot
    frequency : str
        Frequency for analysis ('weekly' or 'monthly')
    """
    # Determine period label based on frequency
    period_label = 'Month' if frequency == 'monthly' else 'Week'
    
    if not sectors_list:
        sectors_list = (df.groupby(sector_column, group_keys=False)['total_exposure']
                      .sum()
                      .sort_values(ascending=False)
                      .index)[:8]

    if interactive:
        # Interactive plotly version
        for i, group in enumerate(sectors_list):
            group_data = df[df[sector_column] == group]
            unique_entities = group_data.loc[group_data['Company'].isin(companies_list)]['Company'].unique()

            if len(unique_entities) == 0:
                continue

            color_palette = plt.get_cmap('tab20', 20).colors
            color_palette = [mcolors.rgb2hex(color) for color in color_palette]
            color_map = dict(zip(unique_entities, color_palette[:len(unique_entities)]))

            fig = go.Figure()

            for entity in unique_entities:
                entity_data = group_data[group_data['Company'] == entity].copy()

                # Create complete hover text with (Positive)/(Negative) labels
                hover_texts = []
                for _, row in entity_data.iterrows():

                    hover_parts = [f"<b>Company:</b> {row['Company']}", f"<b>{period_label}:</b> {row['Date'].strftime('%B %d, %Y')}"]

                    # Positive exposure details with (Positive) labels
                    pos_headline = str(row[f'headline_{df_names[0]}']) if pd.notna(row[f'headline_{df_names[0]}']) else ''
                    pos_motivation = str(row[f'motivation_{df_names[0]}']) if pd.notna(row[f'motivation_{df_names[0]}']) else ''
                    pos_text = str(row[f'text_{df_names[0]}']) if pd.notna(row[f'text_{df_names[0]}']) else ''

                    if pos_headline and pos_headline != 'nan' and pos_headline.strip():
                        hover_parts.extend([
                            f"<b>Headline ({positive_hover_label}): </b> {add_line_breaks(pos_headline)}",
                            f"<b>Motivation ({positive_hover_label}): </b> {add_line_breaks(pos_motivation)}",
                            f"<b>Text ({positive_hover_label}): </b> {add_line_breaks(pos_text)}"
                        ])

                    # Negative exposure details with (Negative) labels
                    neg_headline = str(row[f'headline_{df_names[1]}']) if pd.notna(row[f'headline_{df_names[1]}']) else ''
                    neg_motivation = str(row[f'motivation_{df_names[1]}']) if pd.notna(row[f'motivation_{df_names[1]}']) else ''
                    neg_text = str(row[f'text_{df_names[1]}']) if pd.notna(row[f'text_{df_names[1]}']) else ''

                    if neg_headline and neg_headline != 'nan' and neg_headline.strip():
                        hover_parts.extend([
                            f"<b>Headline ({negative_hover_label}): </b> {add_line_breaks(neg_headline)}",
                            f"<b>Motivation ({negative_hover_label}): </b> {add_line_breaks(neg_motivation)}",
                            f"<b>Text ({negative_hover_label}): </b> {add_line_breaks(neg_text)}"
                        ])

                    # If no details available for either
                    if len(hover_parts) == 1:  # Only company name
                        hover_parts.extend([
                            "<b>Headline:</b> No details available",
                            "<b>Motivation:</b> No details available",
                            "<b>Text:</b> No details available"
                        ])

                    hover_text = "<br><br>".join(hover_parts)
                    hover_texts.append(hover_text)

                entity_data['hovertext'] = hover_texts

                # Main line trace
                # Get color with fallback if entity not in color_map
                entity_color = color_map.get(entity, '#808080')  # Default to gray if not found
                
                fig.add_trace(go.Scatter(
                    x=entity_data['Date'],
                    y=entity_data['net_exposure'].fillna(0),
                    mode='lines+markers',
                    name=entity,
                    hovertext=entity_data['hovertext'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    line=dict(color=entity_color, width=2),
                    marker=dict(color=entity_color, size=6),
                    legendgroup=entity,
                ))

                # Add larger markers for non-zero net_exposure
                non_zero_data = entity_data.loc[entity_data['net_exposure'] != 0]
                if not non_zero_data.empty:
                    fig.add_trace(go.Scatter(
                        x=non_zero_data['Date'],
                        y=non_zero_data['net_exposure'],
                        mode='markers',
                        name=entity,
                        hovertext=non_zero_data['hovertext'],
                        hovertemplate='%{hovertext}<extra></extra>',
                        marker=dict(color=entity_color, size=12, symbol='circle'),
                        legendgroup=entity,
                        showlegend=False
                    ))

            # Add horizontal dashed line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=2)

            # Calculate symmetric y-axis range
            all_net_exposures = group_data['net_exposure'].fillna(0)
            max_abs_value = all_net_exposures.abs().max()

            # Add a small padding (10% of the max value)
            if max_abs_value > 0:
                padding = max_abs_value * 0.1
                y_range = [-max_abs_value - padding, max_abs_value + padding]
            else:
                y_range = [-1, 1]

            # Update layout
            fig.update_layout(
                height=500,
                title={
                    'text': f"{group} - {theme_name} {analysis_suffix}",
                    'x': 0.5,
                    'y': 0.98,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 24, 'family': 'Arial Black'}
                },
                template='plotly_white',
                xaxis=dict(
                    title=period_label,
                    tickformat="%b %Y",
                    dtick="M1" if frequency == 'monthly' else "M1",
                    tickangle=45,
                    gridcolor='lightgray',
                    showgrid=True
                ),
                yaxis=dict(
                    title=y_axis_label,
                    gridcolor='lightgray',
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=2,
                    range=y_range
                ),
                hovermode='closest',
                plot_bgcolor='white',
                margin=dict(t=100, b=100)
            )

            fig.show()

            # Add space between plots
            if i < len(sectors_list) - 1:
                display(HTML("<div style='height: 50px;'></div>"))

    else:
        # Static matplotlib version
        for i, group in enumerate(sectors_list):
            group_data = df[df[sector_column] == group]
            unique_entities = group_data.loc[group_data['Company'].isin(companies_list)]['Company'].unique()

            if len(unique_entities) == 0:
                continue

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Use same color palette as plotly
            color_palette = plt.get_cmap('tab20', 20).colors
            color_map = dict(zip(unique_entities, color_palette[:len(unique_entities)]))

            for entity in unique_entities:
                entity_data = group_data[group_data['Company'] == entity].copy()
                
                # Get color with fallback if entity not in color_map
                entity_color = color_map.get(entity, '#808080')  # Default to gray if not found
                
                # Plot main line
                ax.plot(entity_data['Date'], entity_data['net_exposure'].fillna(0), 
                       color=entity_color, linewidth=2, label=entity, marker='o', markersize=4)
                
                # Add larger markers for non-zero net_exposure
                non_zero_data = entity_data.loc[entity_data['net_exposure'] != 0]
                if not non_zero_data.empty:
                    ax.scatter(non_zero_data['Date'], non_zero_data['net_exposure'], 
                             color=entity_color, s=100, alpha=0.8, zorder=5)

            # Add horizontal dashed line at y=0
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2)

            # Calculate symmetric y-axis range
            all_net_exposures = group_data['net_exposure'].fillna(0)
            max_abs_value = all_net_exposures.abs().max()

            if max_abs_value > 0:
                padding = max_abs_value * 0.1
                y_range = [-max_abs_value - padding, max_abs_value + padding]
                ax.set_ylim(y_range)
            else:
                ax.set_ylim([-1, 1])

            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
            
            # Set labels and title
            ax.set_xlabel(period_label, fontsize=12)
            ax.set_ylabel(y_axis_label, fontsize=12)
            ax.set_title(f"{group} - {theme_name} {analysis_suffix}", fontsize=20, fontweight='bold', y=1.08)
            
            # Add grid
            ax.grid(True, alpha=0.3, color='lightgray')
            ax.set_axisbelow(True)
            
            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Leave more space for the high title
            plt.show()

            # Add space between plots (simulate display(HTML) behavior)
            if i < len(sectors_list) - 1:
                print("\n" * 2)


def analyze_basket_with_labels(df_pos, df_neg, companies_basket, start_date, end_date, 
                                      sector_column='Sector', theme_name='Theme', 
                                      positive_hover_label='Positive', negative_hover_label='Negative',
                                      y_axis_label='Net Exposure', analysis_suffix='Weekly Analysis',
                                      interactive=True, frequency='weekly'):
    """
    Main function for weekly or monthly analysis with labeled hover info

    Parameters:
    -----------
    df_pos : pandas.DataFrame
        DataFrame with positive exposure data
    df_neg : pandas.DataFrame
        DataFrame with negative exposure data
    companies_basket : pandas.DataFrame
        DataFrame with basket companies information
    start_date : str
        Start date for analysis
    end_date : str
        End date for analysis
    sector_column : str
        Column to use for sector grouping ('Sector' or 'Industry')
    theme_name : str
        Name of the theme being analyzed (e.g., 'Liquid Cooling', 'Pricing Power')
    positive_hover_label : str
        Label for positive exposure in hover text (e.g., 'Positive', 'Benefits')
    negative_hover_label : str
        Label for negative exposure in hover text (e.g., 'Negative', 'Challenges')
    y_axis_label : str
        Label for y-axis (e.g., 'Net Exposure', 'Net Liquid Cooling Exposure')
    analysis_suffix : str
        Suffix for analysis title (e.g., 'Weekly Analysis', 'Monthly Analysis')
    interactive : bool
        If True, create interactive plot. If False, create static plot
    frequency : str
        Frequency for analysis ('weekly' or 'monthly')

    Returns:
    --------
    pandas.DataFrame
        DataFrame with weekly or monthly exposure data
    """
    companies_list = companies_basket['Company'].unique()

    # Get weekly or monthly exposure data with separate positive/negative daily details
    period_exposure = get_weekly_volume_with_separate_daily_details(
        df_pos, df_neg,
        ['positive_exp', 'negative_exp'],
        sector_column,
        start_date, end_date, frequency
    )

    if period_exposure.empty:
        return period_exposure

        # Calculate net exposure (positive exposure minus negative exposure)
    period_exposure['net_exposure'] = period_exposure['positive_exp'].abs() - period_exposure['negative_exp'].abs()
    
    # Filter period_exposure to include only companies from the original basket
    period_exposure = period_exposure[period_exposure['Company'].isin(companies_list)]

    # Track over time with labeled hover
    track_basket_sectors_weekly_complete_hover_labeled(
        period_exposure,
        ['positive_exp', 'negative_exp'],
        companies_list,
        sector_column,
        theme_name=theme_name,
        positive_hover_label=positive_hover_label,
        negative_hover_label=negative_hover_label,
        y_axis_label=y_axis_label,
        analysis_suffix=analysis_suffix,
        interactive=interactive,
        frequency=frequency
    )

    return period_exposure


def transform_to_reference_format(df_second):
    """
    Transform a dataframe to match the reference format structure
    
    Parameters:
    df_second (pd.DataFrame): The dataframe to transform
    
    Returns:
    pd.DataFrame: The transformed dataframe
    """
    # Create a copy of the second dataframe to avoid modifying the original
    df_transformed = df_second.copy()
    df_transformed = df_transformed.rename(columns={
    'entity_sector': 'Sector',
    'entity_industry': 'Industry',
    'entity_name': 'Company',
    'timestamp_utc': 'Date',
    'headline': 'Headline',
    'motivation': 'Motivation',
    'text': 'Quote'
        })
    
    # Convert Date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_transformed['Date']):
        df_transformed['Date'] = pd.to_datetime(df_transformed['Date'])
    
    # Rename columns to match the reference format
    df_transformed = df_transformed.rename(columns={
        'entity_country': 'Country',
        'entity_ticker': 'Ticker',
        'rp_document_id': 'Document ID'
    })
    
    # Add missing columns that are in the reference dataframe
    df_transformed['Time Period'] = df_transformed['Date'].dt.strftime('%b %Y')  # Extract month and year
    
    # Convert Date column to match the reference format (date only, no time) 
    df_transformed['Date'] = df_transformed['Date'].dt.date
    
    return df_transformed


