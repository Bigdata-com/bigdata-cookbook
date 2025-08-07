from . import keywords_functions
from .label_extractor import extract_label
import pandas as pd
import asyncio
import os
from bigdata import Bigdata
from bigdata.query import Similarity, All
from bigdata.models.search import DocumentType, SortBy
from bigdata.daterange import RollingDateRange, AbsoluteDateRange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import collections
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import seaborn as sns
from collections import defaultdict
from .cost_estimation import compute_costs
import configparser
from tqdm.notebook import tqdm
import matplotlib.colors as mcolors
import json
from pqdm.processes import pqdm



def update_fig_layout(fig):
    fig.update_layout(
        paper_bgcolor="#1f2630",
        plot_bgcolor="#1f2630",
        font=dict(color="#2cfec1"),
        xaxis=dict(tickfont=dict(color="#2cfec1"), gridcolor="#5b5b5b"),
        yaxis=dict(tickfont=dict(color="#2cfec1"), gridcolor="#5b5b5b"),
        title=dict(font=dict(color="#2cfec1")),
        legend=dict(font=dict(color="#2cfec1"), orientation="v"),
        hovermode="closest",
        autosize=True
    )
    return fig



def clean_text(text):
    text = text.replace('{', '').replace('}', '')
    return text

def clean_text_masked(text):
    text = text.replace('{', '').replace('}', '')
    return text



def drop_duplicates(df = pd.DataFrame(), date_column = '', duplicate_set = []):
    df = df.sort_values(by=date_column)
    non_dup_index = []
    dup_index = []
    for i, gp in df.groupby(duplicate_set):
        non_dup_index.append(gp.iloc[0].name)
        dup_index.extend(gp.iloc[1:].index.values)
    deduplicated_df = df.loc[non_dup_index,:].copy().sort_values(by=date_column).reset_index(drop=True)
    duplicates = df.loc[dup_index,:].copy().sort_values(by=date_column).reset_index(drop=True)
    
    return deduplicated_df, duplicates

def reinstate_duplicates(labelled_df= pd.DataFrame(), duplicated_chunks = pd.DataFrame(), date_column = '', duplicate_set = []):
    
    final_df = pd.concat([labelled_df, duplicated_chunks], axis=0).sort_values(by=duplicate_set+[date_column]).reset_index(drop=True)
    final_df['label'] = final_df.groupby(duplicate_set)['label'].ffill()
    final_df = final_df.sort_values(by=date_column).reset_index(drop=True)
    
    return final_df


def plot_labels_distribution(df, labels_names_dict):
    total_records = len(df)

    # Calculate counts and percentages
    rating_c = df['label'].value_counts(normalize=False).sort_index()
    rating_percentages = df['label'].value_counts(normalize=True).sort_index() * 100

    color_palette = plt.get_cmap('tab10', 7).colors
    color_palette = [mcolors.rgb2hex(color) for color in color_palette]
    
    # Create the main plot
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Distribution of Labels', 'Percentage Distribution of Labels'])

    # Add bar plot for counts
    fig.add_trace(go.Bar(
        x=[labels_names_dict[label] for label in rating_c.index],
        y=rating_c.values,
        marker=dict(color=color_palette),
        text=rating_c.values,
        textposition='inside',
        name='Count'
    ), row=1, col=1)

    # Add bar plot for percentages
    fig.add_trace(go.Bar(
        x=[labels_names_dict[label] for label in rating_percentages.index],
        y=rating_percentages.values,
        marker=dict(color=color_palette),
        text=[f'{value:.2f}%' for value in rating_percentages.values],
        textposition='inside',
        name='Percentage'
    ), row=1, col=2)

    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis_title='Labels',
        yaxis_title='Count',
        xaxis2_title='Labels',
        yaxis2_title='Percentage'
    )

    return fig

def plot_monthly_document_volume(df_filtered_ai_cost_reduction):
    df_filtered = df_filtered_ai_cost_reduction.copy(deep=True)

    # Parse timestamp to extract month and year
    df_filtered['timestamp_utc'] = pd.to_datetime(df_filtered['timestamp_utc'])
    df_filtered['MonthYear'] = df_filtered['timestamp_utc'].dt.to_period('D')

    # Group by month and entity, then count unique documents
    grouped = df_filtered.groupby(['MonthYear']).agg({'sentence_id': pd.Series.nunique}).reset_index()
    grouped.rename(columns={'sentence_id': 'DocumentCount'}, inplace=True)

    # Sum the document counts for each month
    monthly_volume = grouped.groupby('MonthYear').agg({'DocumentCount': 'sum'}).reset_index()

    # Convert MonthYear to datetime for plotting
    monthly_volume['MonthYear'] = monthly_volume['MonthYear'].dt.to_timestamp()
    monthly_volume = monthly_volume.sort_values('MonthYear')

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_volume['MonthYear'],
        y=monthly_volume['DocumentCount'],
        mode='lines+markers',
        marker=dict(symbol='circle', size=8),
        line=dict(width=2),
        name='Document Count'
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Document Count',
        xaxis=dict(tickformat='%Y-%m-%d', tickangle=45),
        showlegend=False,
    )

    return fig


# Function to add line breaks
def add_line_breaks(text, max_len=50):
    
    if type(text)==str:
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > max_len:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " "
                current_line += word
        lines.append(current_line)
        return "<br>".join(lines)
    else:
        return None

def lookup_sector_information(df_filtered, bigdata_credentials):
    bigdata = bigdata_credentials
    
    entity_key_to_sector = {}
    entity_key_to_group = {}
    entity_key_to_sector_consolidated = {}
    entity_key_to_name = {}
    
    # Lookup entity details by their keys
    entities_lookup = bigdata.knowledge_graph.get_entities(df_filtered['rp_entity_id'].unique().tolist())
    
    for entity in entities_lookup:

        if hasattr(entity, 'id') and hasattr(entity, 'name') and entity.entity_type == 'COMP':
            entity_key_to_name[entity.id] = entity.name
            if hasattr(entity, 'sector'):
                entity_key_to_sector[entity.id] = entity.sector
                entity_key_to_sector_consolidated[entity.id] = entity.sector
            else:
                entity_key_to_sector_consolidated[entity.id] = entity.industry_group
            
            if hasattr(entity, 'industry_group'):
                entity_key_to_group[entity.id] = entity.industry_group
        
    # Display the resulting DataFrame
    df_filtered['sector'] = df_filtered.rp_entity_id.map(entity_key_to_sector)
    df_filtered['industry_group'] = df_filtered.rp_entity_id.map(entity_key_to_group)
    df_filtered['sector_consolidated'] = df_filtered.rp_entity_id.map(entity_key_to_sector_consolidated)
    
    if 'entity_name' not in df_filtered.columns:
        df_filtered['entity_name'] = df_filtered.rp_entity_id.map(entity_key_to_name)
    
    return df_filtered

def get_chunk_when_max_coverage(df_filtered):
    
    entity_day_counts = df_filtered.groupby(['rp_entity_id', 'date']).sentence_id.nunique().reset_index(name='counts')

    # Identify the day with the maximum mentions for each entity
    max_entity_day = entity_day_counts.loc[entity_day_counts.groupby('rp_entity_id')['counts'].idxmax()]

    # Merge back with the original dataframe to get the text value
    result = pd.merge(max_entity_day, df_filtered, on=['rp_entity_id', 'date'], how='left').drop(columns='counts')
    result = result.groupby(['rp_entity_id', 'date'])[['text','motivation']].last().reset_index()
    
    return result


def unmask_motivation(df=pd.DataFrame(), motivation_col='motivation'):
    import re
    
    if type(df[motivation_col])==str:
        unmasked_string = re.sub(r'Target Company_\d{1,2}', df['entity_name'], df[motivation_col])
        if 'other_entities_map' in df.index:
            if df['other_entities_map']:
                if isinstance(df['other_entities_map'],str):
                    for key, name in eval(df['other_entities_map']):
                        unmasked_string = unmasked_string.replace(f'Other Company_{key}', str(name))
                else:
                    for key, name in df['other_entities_map']:
                        unmasked_string = unmasked_string.replace(f'Other Company_{key}', str(name))
    else:
        unmasked_string = None
    
    return unmasked_string

    
    
def top_companies_sector(df_filtered, bigdata_credentials, sector_column, dash = False): 
    
    entity_news_count = defaultdict(int)

    #df_filtered = lookup_sector_information(df_filtered, bigdata_credentials)
    
    df_news_count = df_filtered.groupby(['rp_entity_id']).agg({'sentence_id': pd.Series.nunique}).reset_index()

    df_news_count.rename(columns={'sentence_id': 'NewsCount'}, inplace=True)

    
    #df_news_count = df_news_count.loc[~df_news_count['sector'].isin(['Technology'])]

    chunk_text_df = get_chunk_when_max_coverage(df_filtered)
    df_news_count = df_news_count.merge(df_filtered[['rp_entity_id', 'entity_name', sector_column]], how='left', on='rp_entity_id')
    
    df_news_count = df_news_count.merge(chunk_text_df[['rp_entity_id','text', 'motivation']], on='rp_entity_id', how='left')
    
    # Drop duplicates to ensure each company is counted only once
    df_news_count = df_news_count.drop_duplicates(subset=['rp_entity_id'])

    # Get the top 5 companies by sector
    top_5_by_sector = df_news_count.groupby(sector_column, group_keys=False).apply(lambda x: x.nlargest(5, 'NewsCount')).reset_index(drop=True)

    # Rank sectors by the total number of news mentions
    sector_ranking = (top_5_by_sector.groupby(sector_column, group_keys=False)['NewsCount']
                      .sum()
                      .sort_values(ascending=False)
                      .index)[:10]

    # Determine the number of rows needed for 5 columns
    n_cols = 5
    n_rows = (len(sector_ranking) + n_cols - 1) // n_cols

    # Prepare the layout for subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,  # Adjust the number of columns to accommodate the bar plots
        subplot_titles=[f"{sector} <br> {sector_column.replace('_', ' ').capitalize()}" for sector in sector_ranking],  # Title for each bar plot
        specs=[[{"type": "bar"}] * n_cols for _ in range(n_rows)]
    )

    # Flatten the sector ranking for easy iteration
    sector_list = list(sector_ranking)

    for i, sector in enumerate(sector_list):
        row = i // n_cols + 1
        col = (i % n_cols) + 1

        # Plot bar chart for top 5 companies in the sector
        sector_data = top_5_by_sector[top_5_by_sector[sector_column] == sector]
        
        # Create custom hover text with text and motivation, adding dynamic line breaks for better readability
        hover_text = [
            f"<b>Entity:</b> {row.entity_name}<br><br><b>Text</b>: {add_line_breaks(row.text)}<br><br><b>Motivation</b>: {add_line_breaks(unmask_motivation(row))}"
            for idx, row in sector_data.iterrows()
        ]

        fig.add_trace(
            go.Bar(
                x=sector_data['entity_name'],
                y=sector_data['NewsCount'],
                name=f"{sector} {sector_column.replace('_', ' ').capitalize()}",
                showlegend=False,
                hovertext=hover_text,
                hoverinfo="text",
                marker_color='#3366CC',  # Set all bars to the same color
                hoverlabel=dict(font=dict(size=16))  # Set hover text font size
            ),
            row=row, col=col
        )

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
        height=900 * n_rows,
    )

    if dash:
        return fig
    # Show the plot

    else :
        fig.show()
    
## Obtain volume exposure to a theme and identify a basket of companies
def identify_basket_of_companies(df, dfs_labels,df_names,bigdata_credentials, sector_column,basket_size, start_date, end_date):
    
    # Filter the DataFrame
    df_filt = df.loc[df.label.isin(dfs_labels)].copy().reset_index(drop=True)

    # Perform groupby operations
    total_vol = df_filt.groupby('rp_entity_id')['sentence_id'].nunique().rename('total_exposure').reset_index()
    overall_vol = df.groupby('rp_entity_id')['sentence_id'].nunique().rename('total_volume').reset_index()
    grouped_df = df_filt.groupby(['rp_entity_id', 'label']).size().unstack(fill_value=0).reset_index()

    # Create a dictionary mapping rp_entity_id to entity_name
    entity_name_mapping = df.set_index('rp_entity_id')['entity_name'].to_dict()

    # Map entity_name back to grouped_df
    grouped_df['entity_name'] = grouped_df['rp_entity_id'].map(entity_name_mapping)


    # grouped_df = lookup_sector_information(grouped_df, bigdata_credentials)
    grouped_df.columns.name = None
    
    grouped_df = grouped_df.merge(total_vol, on='rp_entity_id', how='left')
    grouped_df = grouped_df.merge(overall_vol, on='rp_entity_id', how='left')
    
    for label, name in zip(dfs_labels, df_names):
        grouped_df[f'{name}_pct'] = grouped_df[label]*100/grouped_df['total_exposure']
        
        chunk_text_df = get_chunk_when_max_coverage(df_filt.loc[df_filt.label.eq(label)])
        
        grouped_df = pd.concat([grouped_df.set_index('rp_entity_id'), chunk_text_df.rename(columns={'text':f'text_{name}', 'motivation':f'motivation_{name}'}).set_index(
            'rp_entity_id')[[f'text_{name}',f'motivation_{name}']]],axis=1).reset_index()
        
         # Create a dictionary mapping rp_entity_id to entity_name
        entity_name_mapping = df.set_index('rp_entity_id')['entity_name'].to_dict()

        # Map entity_name back to grouped_df
        grouped_df['entity_name'] = grouped_df['rp_entity_id'].map(entity_name_mapping)
        
        grouped_df[f'motivation_{name}'] = grouped_df.apply(unmask_motivation, motivation_col=f'motivation_{name}', axis=1)
        
#     #select those companies that have at least 1 chunk per month on average, remove those companies that have 50% positive and 50% negative exposure
#     total_months = round(((pd.to_datetime(end_date).year-pd.to_datetime(start_date).year)*365 +
#                           (pd.to_datetime(end_date).month - pd.to_datetime(start_date).month)*30 +
#                           (pd.to_datetime(end_date).day - pd.to_datetime(start_date).day))/30,0)
    
#     top_exposed = grouped_df.loc[grouped_df.total_exposure.ge(total_months)].reset_index(drop=True).copy()
#     top_exposed = top_exposed.sort_values('entity_name').reset_index(drop=True)

    ##select top companies in basket
    top_exposed = grouped_df.nlargest(basket_size, ['total_exposure','total_volume'], keep='all').sort_values(f'total_exposure').reset_index(drop=True)

    return top_exposed.drop('total_volume',axis=1)


def get_daily_volume_by_label(df, dfs_labels,df_names,bigdata_credentials, sector_column, start_date, end_date):
    
    from functools import reduce

    dfs_by_date = []
    
    for label,name in zip(dfs_labels, df_names):
        
        df = df.sort_values(by=['date'])
        df.date = pd.to_datetime(df.date)
        df_label = df.loc[df.label.eq(label)].copy().reset_index(drop=True)
        
        vol_by_date = df_label.groupby(['date','rp_entity_id']).agg({'sentence_id':'nunique', 'text':'last', 'motivation':'last'}).reset_index()

        # Get all unique dates and entity names
        all_dates = pd.date_range(start=start_date,end=end_date, freq='D')
        all_entities = df_label['rp_entity_id'].unique()

        # Create a full index of all combinations of dates and entity names
        full_index = pd.MultiIndex.from_product([all_dates, all_entities], names=['date', 'rp_entity_id'])

        # Reindex the original DataFrame to this full index
        vol_by_date_full = vol_by_date.set_index(['date', 'rp_entity_id']).reindex(full_index).reset_index()

        # Fill missing values with zero
        vol_by_date_full['sentence_id'] = vol_by_date_full['sentence_id'].fillna(0)
        vol_by_date_full.date = pd.to_datetime(vol_by_date_full.date)
        # Display the resulting DataFrame
        # vol_by_date_full = lookup_sector_information(vol_by_date_full, bigdata_credentials)
         # Create a dictionary mapping rp_entity_id to entity_name
        entity_name_mapping = df.set_index('rp_entity_id')['entity_name'].to_dict()
        sector_mapping = df.set_index('rp_entity_id')[sector_column].to_dict()

        # Map entity_name back to grouped_df
        vol_by_date_full['entity_name'] = vol_by_date_full['rp_entity_id'].map(entity_name_mapping)
        vol_by_date_full[sector_column] =  vol_by_date_full['rp_entity_id'].map(sector_mapping)
        vol_by_date_full['motivation'] = vol_by_date_full.apply(unmask_motivation, motivation_col=f'motivation', axis=1)
        vol_by_date_full = vol_by_date_full.rename(columns={'sentence_id':name, 'text':f'text_{name}','motivation':f'motivation_{name}'})
        
        dfs_by_date.append(vol_by_date_full[['date', 'entity_name', 'rp_entity_id', sector_column, name, f'text_{name}', f'motivation_{name}']])
        
    if len(dfs_by_date)>1:
        
        final_df = reduce(lambda df1,df2: pd.merge(df1,df2,on=['date', 'entity_name','rp_entity_id', sector_column], how='outer'), dfs_by_date)
        final_df[[f'{name}' for name in df_names]] = final_df[[f'{name}' for name in df_names]].fillna(0)
        final_df['total_exposure'] = final_df[[f'{name}' for name in df_names]].abs().sum(axis=1)
    else:
        final_df = dfs_by_date[0]
        final_df[[f'{name}' for name in df_names]] = final_df[[f'{name}' for name in df_names]].fillna(0)
        final_df['total_exposure'] = final_df[[f'{name}' for name in df_names]].abs().sum(axis=1)
    
    return final_df

def display_basket(df):
    display(df.sort_values(by='total_exposure', ascending=False).head(5))

## Create a stacked bar chart showing relative exposure to the theme, ideally positive/negative exposure
def plot_confidence_bar_chart(basket_df, title_theme, df_names, legend_labels,dash = False):
    
    df = basket_df.copy() 
    
    # Create the figure
    fig = go.Figure()
    
    # Add positive exposure bars
    # Update layout

    # Create the figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=(f"{title_theme}", "Total Volume"),
        horizontal_spacing=0.15
    )

    fig.update_layout(
        barmode='stack',
        yaxis=dict(title='Entity Name'),
        template='plotly_white',
        showlegend=True
    )
    
    colors = ['green','red','grey','orange']
    
    for i, (name, leg_label, color) in enumerate(zip(df_names, legend_labels, colors[:len(df_names)])): ##automate color generator
        base_val = [0]*len(df['entity_name']) if i == 0 else df[[f'{name}_pct' for name in df_names[:i]]].abs().sum(axis=1)
        text_col = f'text_{name}'
        # Create custom hover text with text and motivation, adding dynamic line breaks for better readability
        hover_text = [
            f"<b>{leg_label}:</b> {round(row[f'{name}_pct'],2)}%<br><br><b>Entity:</b>{row['entity_name']}<br><br><b>Text:</b> {add_line_breaks(row[text_col])}<br><br><b>Motivation</b>: {add_line_breaks(unmask_motivation(row, f'motivation_{name}'))}" for index, row in df.iterrows()]
        
        fig.add_trace(go.Bar(
            y=df['entity_name'],
            x=df[f'{name}_pct'],
            name=leg_label,
            orientation='h',
            marker=dict(color=color),
            hovertemplate='%{hovertext}<extra></extra>', #f"<b>{title_theme}:</b> %{x:.2f}%<br><br><b>Entity:</b>%{y}<br><br><b>Text:</b> {add_line_breaks(df[text_col])}<extra></extra>",
            text = df[f'{name}_pct'].apply(lambda x: f'{x:.2f}%'),
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
        y1=len(df['entity_name']) - 0.5,
        line=dict(color='black', width=3, dash='dash'), row=1, col=1
    )

    fig.add_trace(go.Bar(
        y=df['entity_name'],
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
    )

    fig.update_yaxes(showticklabels=False, row=1, col=2)

    if dash:
        return fig
    # Show the plot

    else :
        fig.show()
    
    
def track_basket_sectors_over_time(df, df_names, companies_list, sector_column, sectors_list=[], dash=False):
    import matplotlib.colors as mcolors
    
    if not sectors_list:
        # Rank sectors by the total number of news mentions
        sectors_list = (df.groupby(sector_column, group_keys=False)['total_exposure']
                      .sum()
                      .sort_values(ascending=False)
                      .index)[:5]

    # List to store individual figures
    figs = []

    # Plot lines for each entity within each industry group
    for i, group in enumerate(sectors_list):
        group_data = df[df[sector_column] == group]
        unique_entities = group_data.loc[group_data.rp_entity_id.isin(companies_list)]['entity_name'].unique()

        # Define a colormap with up to 20 distinct colors
        color_palette = plt.get_cmap('tab20', 20).colors
        color_palette = [mcolors.rgb2hex(color) for color in color_palette]
        color_map = dict(zip(unique_entities, color_palette[:len(unique_entities)]))

        fig = go.Figure()

        for entity in unique_entities:
            entity_data = group_data[group_data['entity_name'] == entity].copy()
            entity_data['hovertext'] = entity_data.apply(
                lambda row: f"<b>{row['entity_name']}</b><br><b>Number of Positive Chunks:</b> {row[f'{df_names[0]}']}<br>"
                            f"<b>Number of Negative Chunks:</b> {row[f'{df_names[1]}']}<br>"
                            f"<b>Text (Positive):</b> {add_line_breaks(row[f'text_{df_names[0]}'])}<br>"
                            f"<b>Motivation (Positive):</b> {add_line_breaks(unmask_motivation(row,f'motivation_{df_names[0]}'))}<br>"
                            f"<b>Text (Negative):</b> {add_line_breaks(row[f'text_{df_names[1]}'])}<br>"
                            f"<b>Motivation (Negative):</b> {add_line_breaks(unmask_motivation(row,f'motivation_{df_names[1]}'))}<br>", axis=1)

            # Ensure hovertext is not empty
            entity_data['hovertext'] = entity_data['hovertext'].apply(lambda x: x if x.strip() else "No details available")

            # Positive exposure
            fig.add_trace(go.Scatter(
                x=entity_data['date'],
                y=entity_data['net_exposure'].fillna(0),
                mode='lines',
                name=entity,
                hovertext=entity_data['hovertext'],
                hovertemplate='%{hovertext}<extra></extra>',
                line=dict(color=color_map[entity]),
                legendgroup=entity,
            ))
            
            # Add marker trace for non-zero net_exposure
            non_zero_data = entity_data.loc[entity_data['net_exposure'] != 0]
            if not non_zero_data.empty:
                fig.add_trace(go.Scatter(
                    x=non_zero_data['date'],
                    y=non_zero_data['net_exposure'],
                    mode='markers',
                    name=entity,
                    hovertext=non_zero_data['hovertext'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    marker=dict(color=color_map[entity], size=12),
                    legendgroup=entity,
                    showlegend=False
                ))
            
            # Add marker trace for zero net_exposure with non-null text_positive_exp
            zero_nonnull_data = entity_data.loc[(entity_data['net_exposure'] == 0) & (entity_data[f'text_{df_names[0]}'].notnull())]
            if not zero_nonnull_data.empty:
                fig.add_trace(go.Scatter(
                    x=zero_nonnull_data['date'],
                    y=zero_nonnull_data['net_exposure'],
                    mode='markers',
                    name=entity,
                    hovertext=zero_nonnull_data['hovertext'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    marker=dict(color=color_map[entity], size=12),
                    legendgroup=entity,
                    showlegend=False
                ))

        # Update layout
        fig.update_layout(
            height=300,
            title_text=f"{group}",
            template='plotly_white'
        )
        
        figs.append(fig)
    
    if dash:
        return figs
    else:
        for fig in figs:
            fig.show()
        
        

def top_companies_time(df_filtered, top_n=4, dash=False):
    # Convert timestamp to datetime and extract daily period
    df_filtered['timestamp_utc'] = pd.to_datetime(df_filtered['timestamp_utc'])
    df_filtered['Day'] = df_filtered['timestamp_utc'].dt.to_period('D')

    # Group by day and entity, then count unique documents
    grouped = df_filtered.groupby(['Day', 'entity_name']).agg({'sentence_id': pd.Series.nunique}).reset_index()
    grouped.rename(columns={'sentence_id': 'DocumentCount'}, inplace=True)

    # Ensure all days are represented for each entity
    all_days = pd.date_range(start=grouped['Day'].min().to_timestamp(), end=grouped['Day'].max().to_timestamp(), freq='D').to_period('D')
    all_entities = grouped['entity_name'].unique()
    full_index = pd.MultiIndex.from_product([all_days, all_entities], names=['Day', 'entity_name'])

    # Reindex the DataFrame to include all days and entities, filling missing values with 0
    daily_volume = grouped.set_index(['Day', 'entity_name']).reindex(full_index, fill_value=0).reset_index()

    # Convert Day to datetime for plotting
    daily_volume['Day'] = daily_volume['Day'].dt.to_timestamp()
    daily_volume = daily_volume.sort_values(['Day', 'entity_name'])

    # Identify the top N entities by total volume
    top_entities = (daily_volume.groupby('entity_name')['DocumentCount'].sum()
                    .nlargest(top_n * 2)  # Get more entities than needed to handle errors
                    .index
                    .tolist())

    valid_entities = []
    for entity in top_entities:
        try:
            # Check if the entity can be processed without error
            entity_data = daily_volume[daily_volume['entity_name'] == entity]
            # If successful, add to valid_entities
            valid_entities.append(entity)
            if len(valid_entities) == top_n:
                break
        except Exception as e:
            print(f"Error processing entity {entity}: {e}")
            continue

    # Filter the DataFrame to include only the valid top N entities
    top_entities_data = daily_volume[daily_volume['entity_name'].isin(valid_entities)]

    # Create a mapping of entity names to texts and motivations
    entity_texts_motivations = df_filtered.groupby(['entity_name', 'Day']).apply(lambda x: x[['text', 'motivation', 'entity_name']].to_dict('records')).to_dict()

    # Function to add line breaks
    def add_line_breaks(text, max_len=50):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > max_len:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " "
                current_line += word
        lines.append(current_line)
        return "<br>".join(lines)
    
    def unmask_motivation(df=pd.DataFrame(), motivation_col='motivation'):
        import re

        if type(df[motivation_col]) == str:
            unmasked_string = re.sub(r'Target Company_\d{1,2}', df['entity_name'], df[motivation_col])
        else:
            unmasked_string = None

        return unmasked_string

    # Determine the number of rows needed
    rows = (top_n + 1) // 2

    # Create subplots
    fig = make_subplots(rows=rows, cols=2, subplot_titles=valid_entities, shared_xaxes=True, shared_yaxes=False)

    # Define a colormap with up to 20 distinct colors
    colors_top = 7
    
    if top_n>7 :
        colors_top = top_n
    color_palette = plt.get_cmap('tab10', colors_top).colors
    color_palette = [mcolors.rgb2hex(color) for color in color_palette]
    color_map = dict(zip(valid_entities, color_palette[:len(valid_entities)]))

    line_size = 3
    marker_size = 12

    # Plot each entity's daily document volume
    for i, entity_id in enumerate(valid_entities):
        entity_data = top_entities_data[top_entities_data['entity_name'] == entity_id]
        row = i // 2 + 1
        col = i % 2 + 1

        # Create hover text for each point
        hover_text = []
        marker_dates = []
        marker_counts = []
        for date, count in zip(entity_data['Day'], entity_data['DocumentCount']):
            if (entity_id, date.to_period('D')) in entity_texts_motivations:
                records = entity_texts_motivations[(entity_id, date.to_period('D'))]
                hover_text.append(
                    "<br><br>".join(
                        [
                            f"<b>Text</b>: {add_line_breaks(record['text'])}<br><br><b>Motivation</b>: {add_line_breaks((unmask_motivation(record)))}"
                            for record in records
                        ]
                    )
                )
                marker_dates.append(date)
                marker_counts.append(count)

        # Add line trace
        fig.add_trace(go.Scatter(
            x=entity_data['Day'], 
            y=entity_data['DocumentCount'], 
            mode='lines', 
            name=entity_id, 
            line=dict(color=color_map[entity_id], width=line_size)
        ), row=row, col=col)

        # Add marker trace for non-zero DocumentCount
        non_zero_data = entity_data.loc[entity_data['DocumentCount'] != 0]
        if not non_zero_data.empty:
            fig.add_trace(go.Scatter(
                x=non_zero_data['Day'], 
                y=non_zero_data['DocumentCount'], 
                mode='markers', 
                name=entity_id, 
                hovertext=hover_text, 
                hoverinfo="text", 
                marker=dict(color=color_map[entity_id], size=marker_size)
            ), row=row, col=col)

    # Update layout
    fig.update_layout(
        height=300 * rows,
        showlegend=False,
        template='plotly_white'
    )

    # Show the plot
    if dash:
        return fig
    else:
        fig.show()

    
def provider_user_net(df_filtered_ai_cost_reduction, df_filtered_ai_providers, dash=False):
    # Exclude 'motivation' column during the merge
    cols_to_merge = ['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id', 'entity_name', 'text', 'masked_text']
    merged_df = pd.merge(df_filtered_ai_cost_reduction[cols_to_merge], df_filtered_ai_providers[cols_to_merge], how='outer', indicator=True)

    # Filter out common rows
    df_filtered_ai_cost_reduction_without_common = df_filtered_ai_cost_reduction.copy(deep=True)
    df_filtered_ai_providers_without_common = df_filtered_ai_providers.copy(deep=True)

    df_filtered_ai_cost_reduction_without_common['entity_name'] = df_filtered_ai_cost_reduction_without_common['entity_name'] + '_user'
    df_filtered_ai_providers_without_common['entity_name'] = df_filtered_ai_providers_without_common['entity_name'] + '_provider'

    # Concatenate the dataframes
    df_filtered = pd.concat([df_filtered_ai_cost_reduction_without_common, df_filtered_ai_providers_without_common], ignore_index=True)

    # Create a network graph
    G = nx.Graph()

    # Initialize a dictionary to count co-mentions and store chunk text and motivations
    co_mention_counts = defaultdict(lambda: {'count': 0, 'texts': [], 'motivations': defaultdict(list)})

    # Define a function to truncate text around co-mentions
    def truncate_text(text, entity1, entity2, window=10000):
        text_lower = text.lower()
        idx1 = text_lower.find(entity1.lower())
        idx2 = text_lower.find(entity2.lower())

        if idx1 == -1 or idx2 == -1:
            return text[:window] + "..." if len(text) > window else text

        start = max(0, min(idx1, idx2) - window // 2)
        end = min(len(text), max(idx1, idx2) + len(entity2) + window // 2)
        return text[start:end] + ("..." if end < len(text) else "")

    # Process the DataFrame to identify co-mentions
    for sentence_id, group in df_filtered.groupby('sentence_id'):
        entities = group['entity_name'].unique()
        entity_id_map = dict(zip(group['entity_name'], group['entity_name']))
        if len(entities) > 1:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    # Ensure only provider-user pairs are considered
                    if ('_provider' in entities[i] and '_user' in entities[j]) or ('_user' in entities[i] and '_provider' in entities[j]):
                        pair = tuple(sorted([entities[i], entities[j]]))
                        co_mention_counts[pair]['count'] += 1
                        truncated_text = truncate_text(group.iloc[0]['text'], entities[i], entities[j])
                        co_mention_counts[pair]['texts'].append(truncated_text)
                        for entity in entities:
                            entity_motivations = group[group['entity_name'] == entity]['motivation'].tolist()
                            for idx, motivation in enumerate(entity_motivations):
                                entity_motivations[idx] = motivation.replace(f'Target Company_', entity_id_map[entity])
                                # Remove the '_user' and '_provider' tags from the motivation
                                entity_motivations[idx] = entity_motivations[idx].replace('_user', '').replace('_provider', '')
                            co_mention_counts[pair]['motivations'][entity].extend(entity_motivations)

    # Add edges to the graph based on co-mentions
    for pair, data in co_mention_counts.items():
        if data['count'] >= 1:  # Adjust threshold here if needed
            G.add_edge(pair[0], pair[1], weight=data['count'], texts=data['texts'], motivations=data['motivations'])

    # Check if the graph is empty
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("The graph is empty. Please check the data and conditions.")
    else:
        # Get the positions of the nodes using a spring layout
        pos = nx.spring_layout(G, k=0.1, seed=42, dim=3)

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_z = []

        for edge in G.edges(data=True):
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            line=dict(width=2, color='lightgrey'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node traces
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_annotations = []
        node_color = []

        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            clean_node_text = node.replace('_user', '').replace('_provider', '')
            node_text.append(clean_node_text)
            node_color.append('royalblue' if node.endswith('_user') else 'tomato')
            # Remove duplicate chunks and truncate text
            unique_texts = [chunk for edge in G.edges(node, data=True) for chunk in edge[2]['texts']]
            unique_texts = list(dict.fromkeys(unique_texts))  # Remove duplicates while preserving order
            formatted_texts = [f"<b>Text {i+1}:</b><br>" + '<br>'.join([text[j:j+80] for j in range(0, len(text), 80)]) for i, text in enumerate(unique_texts)]

            # Format motivations
            formatted_motivations = []
            motivations_added = set()
            for i, text in enumerate(unique_texts):
                for edge in G.edges(node, data=True):
                    motivations = edge[2]['motivations'][node]
                    for motivation in motivations:
                        if motivation not in motivations_added:
                            formatted_motivations.append(f"<b>Motivation {len(formatted_motivations)+1}:</b><br>" + '<br>'.join([motivation[j:j+80] for j in range(0, len(motivation), 80)]))
                            motivations_added.add(motivation)

            node_annotations.append('<br><br>'.join(formatted_texts) + '<br><br>' + '<br><br>'.join(formatted_motivations))

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers+text',
            text=node_text,
            hovertext=node_annotations,
            hoverinfo='text',
            marker=dict(
                color=node_color,
                size=8,
                opacity=0.8,
                line_width=1
            )
        )

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(

                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=20, r=20, t=40),  # Reduced left and right margins
                            annotations=[dict(
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 )],
                            scene=dict(
                                xaxis=dict(showgrid=True, zeroline=True, backgroundcolor='#D3D3D3', gridcolor='rgba(255, 255, 255, 0.1)'),
                                yaxis=dict(showgrid=True, zeroline=True, backgroundcolor='#D3D3D3', gridcolor='rgba(255, 255, 255, 0.1)'),
                                zaxis=dict(showgrid=True, zeroline=True, backgroundcolor='#D3D3D3', gridcolor='rgba(255, 255, 255, 0.1)'),
                            ),
                        )
        )

        if dash:
            return fig
        else:
            fig.show()
            
def obtain_company_topic_links(df,labels, topic_blacklist, bigdata):
    if isinstance(df['topics'].iloc[0], str):
        df['topics'] = df['topics'].apply(eval)
        exploded_topics = df.explode('topics')
    
    else:
        exploded_topics = df.explode('topics')
    
    df_filt = exploded_topics.loc[~(exploded_topics.topics.isin(topic_blacklist))].copy()
    
    df_count = df_filt.loc[df_filt.label.isin(['Y','N'])].groupby(['rp_entity_id', 'topics']).agg({
    'sentence_id': pd.Series.nunique,
    'label': lambda x: x.value_counts().idxmax(),
    'other_entities_map':'first'
}).reset_index().rename(columns={'sentence_id': 'count', 'label': 'major_label'})
    
    # df_count = lookup_sector_information(df_count, bigdata)
    
    df_count = df_count.merge(get_chunk_when_max_coverage(df.loc[df.label.isin(labels)]).drop('date',axis=1), on='rp_entity_id',how='left')
    
    entity_name_mapping = df.set_index('rp_entity_id')['entity_name'].to_dict()

    # Map entity_name back to grouped_df
    df_count['entity_name'] = df_count['rp_entity_id'].map(entity_name_mapping)

    df_count['motivation_unmasked'] = df_count.apply(unmask_motivation, axis=1)
    
    tmp = []
    entity_name_mapping = df.set_index('rp_entity_id')['entity_name'].to_dict()

    for label in labels:
        
    # Prepare the node size and color information
        entity_sizes = df.loc[df.label.eq(label)].groupby('rp_entity_id').agg({'sentence_id': pd.Series.nunique,}).reset_index()
        # entity_sizes=lookup_sector_information(entity_sizes, bigdata)
        # Map entity_name back to grouped_df
        entity_sizes['entity_name'] = entity_sizes['rp_entity_id'].map(entity_name_mapping)
        entity_sizes = entity_sizes.merge(get_chunk_when_max_coverage(df.loc[df.label.eq(label)]).drop('date',axis=1), on='rp_entity_id',how='left')
        entity_sizes['motivation_unmasked'] = entity_sizes.apply(unmask_motivation, axis=1)
        
        tmp.append(entity_sizes)
    
    entity_sizes = pd.concat(tmp, axis=0).reset_index(drop=True)

    topic_sizes = df_count.groupby(['topics']).agg({'entity_name':'nunique','text':'first'}).reset_index()
    
    # Determine color based on majority label
    def determine_color(label):
        return 'green' if label == labels[0] else 'red'

    entity_colors = df_count.groupby('entity_name')['major_label'].apply(lambda x: determine_color(x.mode()[0])).reset_index()
    topic_colors = df_count.groupby('topics')['major_label'].apply(lambda x: determine_color(x.mode()[0])).reset_index()
    
    return df_count, entity_sizes, topic_sizes, entity_colors, topic_colors

def generate_networkx_network(df_count, entity_sizes, topic_sizes, entity_colors, topic_colors):
    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes for rp_entity_ids
    for idx, row in entity_sizes.iterrows():
        motivation_with_breaks = add_line_breaks(row['motivation_unmasked']).replace('<br>', '\n')
        text_with_breaks = add_line_breaks(row['text']).replace('<br>', '\n')

        G.add_node(row['entity_name'], label=row['entity_name'].replace(" ", "\n"), value=row['sentence_id'], size=row['sentence_id']*10,
                   text=row['text'],motivation=row['motivation_unmasked'],
                   title=f"Entity: {row['entity_name']}\n\nNumber of chunks: {row['sentence_id']}\n\nText: {text_with_breaks}\n\nMotivation: {motivation_with_breaks}",
                   shape='circle', color=entity_colors.loc[entity_colors['entity_name'] == row['entity_name'], 'major_label'].values[0], physics=False)

    # Add nodes for topics
    for idx, row in topic_sizes.loc[topic_sizes.entity_name.gt(2)].iterrows():

        text_with_breaks = add_line_breaks(row['text']).replace('<br>', '\n')

        G.add_node(row['topics'], size=row['entity_name']*4,value=row['entity_name'],
                   text = f"<b>Text:</b> {row['text']}<br><br><b>Number of chunks:</b> {row['entity_name']}",motivation=None,
                   title=f"Topic: {row['topics']}\n\nNumber of chunks: {row['entity_name']}\n\nText: {text_with_breaks}",
                   shape = 'diamond',color='blue', physics=False) #determine_color(topic_colors.loc[topic_colors['topics'] == row['topics'], 'major_label'].values[0]))

    # Add edges
    for idx, row in df_count.loc[df_count.topics.isin(topic_sizes.loc[topic_sizes.entity_name.gt(2)].topics.unique())].iterrows():
        G.add_edge(row['entity_name'], row['topics']) ##not sure I need text info here

    # Get 3D positions for the nodes
    pos = nx.spring_layout(G, k=0.1, iterations=10, seed=1)
    # Remove nodes without edges
    nodes_to_remove = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)
    
    return G


            
def generate_pyvis_chart(G, legend):
    # Get 3D positions for the nodes
    pos = nx.spring_layout(G, k=0.1, iterations=10, seed=1)
    # Remove nodes without edges
    nodes_to_remove = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)

    from pyvis.network import Network
    net = Network('1000px', '1600px',notebook=True, cdn_resources='in_line', neighborhood_highlight=True)
    net.from_nx(G)
    # # Customize hover colors
    # for node in net.nodes:
    #     node['color'] = {
    #         'background': node['color'],
    #         'border': node['color'],
    #         'highlight': {
    #             'background': node['color'],  # Color when node is hovered
    #             'border': node['color']          # Border color when node is hovered
    #         }
    #     }

    # Add Legend Nodes
    step = 130
    x = -600
    y = -600
    num_legend_nodes = 3

    # Function to convert color name to RGBA with transparency
    def to_rgba(color, alpha=0.5):
        colors = {
            'red': 'rgba(255, 0, 0, {})'.format(alpha),
            'green': 'rgba(0, 100, 0, {})'.format(alpha),
            'blue': 'rgba(0, 0, 255, {})'.format(alpha)
        }
        return colors.get(color, color)  # Default to given color if not in dict


    legend_nodes = []
    for legend_node, topic, color, shape, size in zip(range(num_legend_nodes), ['Company Positively Affected', 'Company Negatively Affected', 'Topic'],['green','red', 'blue',], ['circle','circle','diamond'], [5,5,45]):
        legend_nodes.append(
            (
                net.num_nodes() + legend_node, 
                {
                    'id': topic, 
                    'label': str(topic),
                    'size': size, 
                    'fixed': True, # So that we can move the legend nodes around to arrange them better
                    'physics': False, 
                    'x': x, 
                    'y': f'{y + legend_node*step}px',
                    'shape': shape, 
                    'widthConstraint': 90, 
                    'font': {'size': 18},
                    'color': to_rgba(color, 0.6)}) #{'background': color,'border': color,'highlight': {'background': color,'border': color}}})
        )

    for node in legend_nodes:
        net.add_node(node[0], **node[1])
    net.barnes_hut(overlap=1, damping=0,spring_strength=0.6, gravity=-2000,
                  spring_length=250)
    net.toggle_physics(False)
    net.set_options('{"layout":{"randomSeed":1}}')

    return net.generate_html()


