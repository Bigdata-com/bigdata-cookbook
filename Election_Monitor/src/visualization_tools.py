import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import seaborn as sns
import pandas as pd
from collections import defaultdict
import re
from functools import reduce
import sys

def lookup_sector_information(df_filtered, bigdata_credentials):
    bigdata = bigdata_credentials
    
    entity_key_to_sector = {}
    entity_key_to_group = {}
    entity_key_to_sector_consolidated = {}
    entity_key_to_name = {}
    
    # Get unique entity IDs from the DataFrame
    unique_entity_ids = df_filtered['rp_entity_id'].unique().tolist()

    # Iterate through entity keys in batches
    for entity_batch in chunked_iterable(unique_entity_ids, max_size_kb=8):
        # Lookup entity details for the current batch
        entities_lookup = bigdata.knowledge_graph.get_entities(entity_batch)

        # Process the returned entities
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

# Helper function to divide the entity_keys into batches of a given size (8KB limit)
def chunked_iterable(iterable, max_size_kb):
    from itertools import islice
    it = iter(iterable)
    max_size_bytes = max_size_kb * 1024
    batch = []
    current_size = 0

    for item in it:
        item_size = sys.getsizeof(item)
        if current_size + item_size > max_size_bytes - 100:
            yield batch
            batch = []
            current_size = 0

        batch.append(item)
        current_size += item_size

    if batch:
        yield batch


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

def get_chunk_when_max_coverage(df_filtered):
    
    entity_day_counts = df_filtered.groupby(['rp_entity_id', 'Date']).sentence_id.nunique().reset_index(name='counts')

    # Identify the day with the maximum mentions for each entity
    max_entity_day = entity_day_counts.loc[entity_day_counts.groupby('rp_entity_id')['counts'].idxmax()]

    # Merge back with the original dataframe to get the text value
    result = pd.merge(max_entity_day, df_filtered, on=['rp_entity_id', 'Date'], how='left').drop(columns='counts')
    text_cols = ['Headline','Quote']
    if 'Motivation' in result.columns:
        text_cols.append('Motivation')
        #result['Motivation'] = result.apply(unmask_motivation, axis=1)
    
    result = result.groupby(['rp_entity_id', 'Date'])[text_cols].last().reset_index()
    
    return result

##beta!
def unmask_motivation(row=pd.DataFrame(), motivation_col='motivation'):
    
    if type(row[motivation_col])==str:
        unmasked_string = re.sub(r'Target Company_\d{1,2}', row['entity_name'], row[motivation_col])
        if 'other_entities_map' in row.index:
            if row['other_entities_map']:
                if isinstance(row['other_entities_map'],str):
                    for key, name in eval(row['other_entities_map']):
                        unmasked_string = unmasked_string.replace(f'Other Company_{key}', str(name))
                elif isinstance(row['other_entities_map'],float):
                    pass
                else:
                    for key, name in row['other_entities_map']:
                        unmasked_string = unmasked_string.replace(f'Other Company_{key}', str(name))
    else:
        unmasked_string = None
    
    return unmasked_string

def top_companies_sector(df_filtered, bigdata_credentials, sector_column): 
    
    entity_news_count = defaultdict(int)

    df_filtered = lookup_sector_information(df_filtered, bigdata_credentials)
    
    df_news_count = df_filtered.groupby(['rp_entity_id']).agg({'sentence_id': pd.Series.nunique}).reset_index()

    df_news_count.rename(columns={'sentence_id': 'NewsCount'}, inplace=True)

    chunk_text_df = get_chunk_when_max_coverage(df_filtered)
    df_news_count = df_news_count.merge(df_filtered[['rp_entity_id', 'entity_name', sector_column]], how='left', on='rp_entity_id')
    
    df_news_count = df_news_count.merge(chunk_text_df[['rp_entity_id','Headline','Quote', 'Motivation']], on='rp_entity_id', how='left')
    
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
        vertical_spacing=0.4,
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}] * n_cols for _ in range(n_rows)],
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
            f"<b>Entity:</b> {row.entity_name}<br><br><b>Headline</b>:{add_line_breaks(row.Headline)}<br><br><b>Motivation</b>: {add_line_breaks(row.Motivation)}<br><br><b>Text</b>: {add_line_breaks(row.Quote)}"
            for idx, row in sector_data.iterrows()
        ]

        fig.add_trace(
            go.Bar(
                x= sector_data['entity_name'],
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
            #autotickangles=[25, 35, 45, 60],
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
        title_text=f"Top Companies by {sector_column.replace('_', ' ').capitalize()}",
    )

    # Show the interactive plot
    fig.show()
    
## Obtain volume exposure to a theme and identify a basket of companies
def identify_basket_of_companies(df, dfs_labels,df_names,bigdata_credentials, sector_column,basket_size, start_date, end_date):
    
    df_filt = df.loc[df.label.isin(dfs_labels)].copy().reset_index(drop=True)
    
    df_filt = df_filt.loc[df_filt.Date.ge(start_date)&df_filt.Date.le(end_date)]
    
    total_vol = df_filt.groupby('rp_entity_id')['sentence_id'].nunique().rename('total_exposure').reset_index()
    overall_vol = df.groupby('rp_entity_id')['sentence_id'].nunique().rename('total_volume').reset_index()
    
    grouped_df = df_filt.groupby(['rp_entity_id', 'label']).size().unstack(fill_value=0).reset_index()
    grouped_df = lookup_sector_information(grouped_df, bigdata_credentials)
    grouped_df.columns.name = None
    
    grouped_df = grouped_df.merge(total_vol, on='rp_entity_id', how='left')
    grouped_df = grouped_df.merge(overall_vol, on='rp_entity_id', how='left')
    
    for label, name in zip(dfs_labels, df_names):
        grouped_df[f'{name}_pct'] = grouped_df[label]*100/grouped_df['total_exposure']
        
        chunk_text_df = get_chunk_when_max_coverage(df_filt.loc[df_filt.label.eq(label)])
        grouped_df = pd.concat([grouped_df.set_index('rp_entity_id'), chunk_text_df.rename(columns={'Headline':f'headline_{name}','Quote':f'text_{name}', 'Motivation':f'motivation_{name}'}).set_index(
            'rp_entity_id')[[f'headline_{name}', f'text_{name}',f'motivation_{name}']]],axis=1).reset_index()

    ##select top companies in basket
    top_exposed = grouped_df.nlargest(basket_size, ['total_exposure','total_volume'], keep='first').sort_values(f'total_exposure').reset_index(drop=True)
    top_exposed['start_date'] = start_date
    top_exposed['end_date'] = end_date

    return top_exposed.drop('total_volume',axis=1)

def get_daily_volume_by_label(df, dfs_labels,df_names,bigdata_credentials, sector_column, start_date, end_date):

    dfs_by_date = []
    
    for label,name in zip(dfs_labels, df_names):
        
        df = df.sort_values(by=['Date'])
        df['Motivation'] = df.apply(unmask_motivation, motivation_col=f'Motivation', axis=1)
        df.Date = pd.to_datetime(df.Date)
        df_label = df.loc[df.label.eq(label)].copy().reset_index(drop=True)
        
        vol_by_date = df_label.groupby(['Date','rp_entity_id']).agg({'sentence_id':'nunique','Headline':'last', 'Quote':'last', 'Motivation':'last'}).reset_index()

        # Get all unique dates and entity names
        all_dates = pd.date_range(start=start_date,end=end_date, freq='D')
        all_entities = df_label['rp_entity_id'].unique()

        # Create a full index of all combinations of dates and entity names
        full_index = pd.MultiIndex.from_product([all_dates, all_entities], names=['Date', 'rp_entity_id'])

        # Reindex the original DataFrame to this full index
        vol_by_date_full = vol_by_date.set_index(['Date', 'rp_entity_id']).reindex(full_index).reset_index()

        # Fill missing values with zero
        vol_by_date_full['sentence_id'] = vol_by_date_full['sentence_id'].fillna(0)
        vol_by_date_full.Date = pd.to_datetime(vol_by_date_full.Date)
        # Display the resulting DataFrame
        vol_by_date_full = lookup_sector_information(vol_by_date_full, bigdata_credentials)
        
        #vol_by_date_full['Motivation'] = vol_by_date_full.apply(unmask_motivation, motivation_col=f'Motivation', axis=1)
        vol_by_date_full = vol_by_date_full.rename(columns={'sentence_id':name, 'Headline':f'headline_{name}','Quote':f'text_{name}','Motivation':f'motivation_{name}'})
        
        dfs_by_date.append(vol_by_date_full[['Date', 'entity_name', 'rp_entity_id', sector_column, name,f'headline_{name}', f'text_{name}', f'motivation_{name}']])
        
    if len(dfs_by_date)>1:
        
        final_df = reduce(lambda df1,df2: pd.merge(df1,df2,on=['Date', 'entity_name','rp_entity_id', sector_column], how='outer'), dfs_by_date)
        final_df[[f'{name}' for name in df_names]] = final_df[[f'{name}' for name in df_names]].fillna(0)
        final_df['total_exposure'] = final_df[[f'{name}' for name in df_names]].abs().sum(axis=1)
    else:
        final_df = dfs_by_date[0]
        final_df[[f'{name}' for name in df_names]] = final_df[[f'{name}' for name in df_names]].fillna(0)
        final_df['total_exposure'] = final_df[[f'{name}' for name in df_names]].abs().sum(axis=1)
    
    return final_df

def display_basket(df):
    cols = []
    for i in df.columns:
        if len(i) == 1:
            cols.append(i)
    tmp = pd.DataFrame()
    for col in cols:
        tmp = pd.concat([tmp, df.sort_values(by=col, ascending=False).head(3)], axis=0)
    display(tmp.drop_duplicates().reset_index(drop=True))

## Create a stacked bar chart showing relative exposure to the theme, ideally positive/negative exposure
def plot_confidence_bar_chart(basket_df, title_theme, df_names, legend_labels):
    
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
        title='Positive and Negative Exposure Percentages by Entity',
        xaxis=dict(range=[0, 100], dtick=10),
        yaxis=dict(title='Entity Name'),
        template='plotly_white',
        showlegend=True,
    )
    
    fig.update_layout(yaxis = dict(
        tickfont = dict(
            size=18-(2*len(basket_df)//10))
    ))
    
    fig.update_yaxes(showticklabels=True, row=1, col=1)
    
    colors = ['green','red','grey','orange']
    
    for i, (name, leg_label, color) in enumerate(zip(df_names, legend_labels, colors[:len(df_names)])): ##automate color generator
        base_val = [0]*len(df['entity_name']) if i == 0 else df[[f'{name}_pct' for name in df_names[:i]]].abs().sum(axis=1)
        text_col = f'text_{name}'
        # Create custom hover text with text and motivation, adding dynamic line breaks for better readability
        hover_text = [
            f"<b>{leg_label}:</b> {round(row[f'{name}_pct'],2)}%<br><br><b>Entity:</b>{row['entity_name']}<br><br><b>Headline</b>: {add_line_breaks(row[f'headline_{name}'])}<br><br><b>Motivation</b>: {add_line_breaks(row[ f'motivation_{name}'])}<br><br><b>Text:</b> {add_line_breaks(row[text_col])}" for index, row in df.iterrows()]
        
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
        height=900,width=1400,
        title_text=f"{title_theme} Watchlist",
    )

    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # Show the plot
    fig.show()
    
## Plot net exposure and chunk text over time
def track_basket_sectors_over_time(df,df_names, companies_list,sector_column, sectors_list = []):
    # Create the figure
    fig = go.Figure()
    import matplotlib.colors as mcolors
    import math
    
    if not sectors_list:
        # Rank sectors by the total number of news mentions
        sectors_list = (df.groupby(sector_column, group_keys=False)['total_exposure']
                      .sum()
                      .sort_values(ascending=False)
                      .index)[:8]

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
                lambda row: f"<b>{row['entity_name']}</b><br><b>Number of {df_names[0].title()} Chunks:</b> {row[f'{df_names[0]}']}<br>"
                            f"<b>Number of {df_names[1].title()} Chunks:</b> {row[f'{df_names[1]}']}<br>"
                            f"<b>Headline:</b> {add_line_breaks(row[f'headline_{df_names[0]}'])}<br>"
                            f"<b>Motivation ({df_names[0].title()}):</b> {add_line_breaks(row[f'motivation_{df_names[0]}'])}<br>"            
                            f"<b>Text ({df_names[0].title()}):</b> {add_line_breaks(row[f'text_{df_names[0]}'])}<br>"
                            f"<b>Headline:</b> {add_line_breaks(row[f'headline_{df_names[1]}'])}<br>"            
                            f"<b>Motivation ({df_names[1].title()}):</b> {add_line_breaks(row[f'motivation_{df_names[1]}'])}<br>"
                            f"<b>Text ({df_names[1].title()}):</b> {add_line_breaks(row[f'text_{df_names[1]}'])}<br>", axis=1)

            # Ensure hovertext is not empty
            entity_data['hovertext'] = entity_data['hovertext'].apply(lambda x: x if x.strip() else "No details available")

            # Positive exposure
            fig.add_trace(go.Scatter(
                x=entity_data['Date'],
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
                    x=non_zero_data['Date'],
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
                    x=zero_nonnull_data['Date'],
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
            height=500,
            title_text=f"{group}",
            template='plotly_white'
        )

        # Show the plot
        fig.show()

def top_companies_time(df_filtered):
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

    # Identify the top 4 entities by total volume
    top_entities = (daily_volume.groupby('entity_name')['DocumentCount'].sum()
                    .nlargest(4)
                    .index
                    .tolist())

    # Filter the DataFrame to include only the top 4 entities
    top_entities_data = daily_volume[daily_volume['entity_name'].isin(top_entities)]

    # Create a mapping of entity names to texts and motivations
    entity_texts_motivations = df_filtered.groupby(['entity_name', 'Day']).apply(lambda x: x[['Headline','Quote', 'Motivation']].to_dict('records')).to_dict()

    # Create subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=top_entities, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.03, horizontal_spacing=0.1)

    # Plot each entity's daily document volume
    for i, entity_id in enumerate(top_entities):
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
                            f"<b>Headline</b>: {add_line_breaks(record['Headline'])}<br><br><b>Motivation</b>: {add_line_breaks(record['Motivation'])}<br><br><b>Text</b>: {add_line_breaks(record['Quote'])}"
                            for record in records
                        ]
                    )
                )
                marker_dates.append(date)
                marker_counts.append(count)

        # Add line trace
        fig.add_trace(go.Scatter(x=entity_data['Day'], y=entity_data['DocumentCount'], mode='lines', name=entity_id, line=dict(width=3)), row=row, col=col)

        # Add marker trace
        fig.add_trace(go.Scatter(x=marker_dates, y=marker_counts, mode='markers', name=entity_id, hovertext=hover_text, hoverinfo="text", marker=dict(size=10)), row=row, col=col)

    # Update layout
    fig.update_layout(height=2000, width=2000, title_text="Daily Document Volume by Top 4 Entities", showlegend=False)

    # Show the plot
    fig.show()

### display topic network        
def obtain_company_topic_links(df,labels, topic_blacklist, bigdata):
    # First, filter out COMP topics using the topics_type column
    if isinstance(df['topics'].iloc[0], str):
        df['topics'] = df['topics'].apply(eval)
        df['topics_type'] = df['topics_type'].apply(eval)
    
    # Create a copy to avoid modifying original data
    df_filtered = df.copy()
    
    # Filter out COMP entities from topics based on topics_type
    filtered_rows = []
    for idx, row in df_filtered.iterrows():
        topics = row['topics']
        topics_types = row['topics_type']
        
        # Filter out topics where type is COMP
        filtered_topics = []
        for topic, topic_type in zip(topics, topics_types):
            if topic_type != 'COMP':
                filtered_topics.append(topic)
        
        # Update the row with filtered topics
        row_copy = row.copy()
        row_copy['topics'] = filtered_topics
        filtered_rows.append(row_copy)
    
    # Create new dataframe with filtered topics
    df_filtered = pd.DataFrame(filtered_rows)
    
    # Now explode the filtered topics
    exploded_topics = df_filtered.explode('topics')
    
    # Apply blacklist filtering
    df_filt = exploded_topics.loc[~(exploded_topics.topics.isin(topic_blacklist))].copy()
    
    df_count = df_filt.loc[df_filt.label.isin(labels)].groupby(['rp_entity_id', 'topics']).agg({
    'sentence_id': pd.Series.nunique,
    'label': lambda x: x.value_counts().idxmax(),
    'other_entities_map':'first'
}).reset_index().rename(columns={'sentence_id': 'count', 'label': 'major_label'})
    
    df_count = lookup_sector_information(df_count, bigdata)
    
    df_count = df_count.merge(get_chunk_when_max_coverage(df.loc[df.label.isin(labels)]).drop('Date',axis=1), on='rp_entity_id',how='left')

    #df_count['motivation_unmasked'] = df_count.apply(unmask_motivation, axis=1)
    
    tmp = []
    for label in labels:
        
    # Prepare the node size and color information
        entity_sizes = df.loc[df.label.eq(label)].groupby('rp_entity_id').agg({'sentence_id': pd.Series.nunique,}).reset_index()
        entity_sizes=lookup_sector_information(entity_sizes, bigdata)
        entity_sizes = entity_sizes.merge(get_chunk_when_max_coverage(df.loc[df.label.eq(label)]).drop('Date',axis=1), on='rp_entity_id',how='left')
        #entity_sizes['motivation_unmasked'] = entity_sizes.apply(unmask_motivation, axis=1)
        
        tmp.append(entity_sizes)
    
    entity_sizes = pd.concat(tmp, axis=0).reset_index(drop=True)

    topic_sizes = df_count.groupby(['topics']).agg({'entity_name':'nunique','Headline':'first','Quote':'first'}).reset_index()
    
    # Determine color based on majority label
    def determine_color(label):
        return 'green' if label == labels[0] else 'red'

    # Assign colors based on the count of major labels (using the `count` column)
    def assign_color_based_on_count(group):
        # Find the row with the highest 'count' and use its major_label for color assignment
        max_count_row = group.loc[group['count'].idxmax()]
        return determine_color(max_count_row['major_label'])

    # Group by entity name and assign colors based on label counts
    entity_colors = df_count.groupby('entity_name').apply(assign_color_based_on_count).reset_index().rename(columns={0: 'major_label'})

    # Group by topics and assign colors based on label counts
    topic_colors = df_count.groupby('topics').apply(assign_color_based_on_count).reset_index().rename(columns={0: 'major_label'})

    return df_count, entity_sizes, topic_sizes, entity_colors, topic_colors


def generate_networkx_network(df_count, entity_sizes, topic_sizes, entity_colors, topic_colors):
    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes for rp_entity_ids
    for idx, row in entity_sizes.iterrows():
        motivation_with_breaks = add_line_breaks(row['Motivation']).replace('<br>', '\n')
        text_with_breaks = add_line_breaks(row['Quote']).replace('<br>', '\n')
        headline_with_breaks = add_line_breaks(row['Headline']).replace('<br>', '\n') 

        # Get entity color with fallback
        entity_color_match = entity_colors.loc[entity_colors['entity_name'] == row['entity_name'], 'major_label']
        entity_color = entity_color_match.values[0] if len(entity_color_match) > 0 else 'gray'
        
        G.add_node(row['entity_name'], label=row['entity_name'].replace(" ", "\n"), value=row['sentence_id'], size=row['sentence_id']*10,
                   headline=row['Headline'],
                   text=row['Quote'],motivation=row['Motivation'],
                   title=f"Entity: {row['entity_name']}\n\nNumber of chunks: {row['sentence_id']}\n\nHeadline: {headline_with_breaks}\n\nMotivation: {motivation_with_breaks}\n\nText: {text_with_breaks}",
                   shape='circle', color=entity_color, physics=False)

    # Add nodes for topics
    for idx, row in topic_sizes.loc[topic_sizes.entity_name.gt(2)].iterrows():

        text_with_breaks = add_line_breaks(row['Quote']).replace('<br>', '\n')

        G.add_node(row['topics'], size=row['entity_name']*4,value=row['entity_name'],
                   headline=row['Headline'],
                   text = f"<b>Headline:</b> {row['Headline']}<br><br><b>Text:</b> {row['Quote']}<br><br><b>Number of chunks:</b> {row['entity_name']}",motivation=None,
                   title=f"Topic: {row['topics']}\n\nNumber of chunks: {row['entity_name']}\n\nHeadline: {row['Headline']}\n\nText: {text_with_breaks}",
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

def generate_plotly_network_chart(G, legend=[], interactive=True):
    
    if interactive:
        # Interactive plotly version (existing code)
        # Get 3D positions for the nodes
        pos = nx.spring_layout(G, dim=2, k=0.1, iterations=10, seed=1)

        # Create Plotly traces for nodes and edges
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1])
            edge_y.extend([y0, y1])


        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color='rgb(211, 211, 211)'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_color = []
        node_size = []
        node_annotations = []
        node_shape = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append(G.nodes[node]['color'])
            node_size.append(G.nodes[node]['size'])  # Adjust size for visibility,
            node_shape.append(G.nodes[node]['shape'])
            # Remove duplicate chunks and truncate text
            # for edge in G.edges(node, data=True):
            #     print(edge)
            #     print(edge[2]['text'])
        #     unique_texts = [f"<b>Text:</b> {add_line_breaks(node['text'])} <br>"]
        #     unique_texts = list(dict.fromkeys(unique_texts))  # Remove duplicates while preserving order

        #     # Format motivations
        #     formatted_motivations = []
        #     motivations_added = set()
        #     for i, text in enumerate(unique_texts):
        #         for edge in G.edges(node, data=True):
        #             motivations = edge[2]['motivation']
        #             formatted_motivations.append(f"<b>Motivation:</b> {add_line_breaks(motivations)}<br>")
        #             motivations_added.add(motivations)

            if G.nodes[node]['motivation']:
                node_annotations.append(f"<b>Entity:</b> {node} <br><br><b>Headline:</b> {G.nodes[node]['headline']} <br><br><b>Motivation:</b> {add_line_breaks(G.nodes[node]['motivation'])}" + '<br><br>' + f"<b>Text:</b> {add_line_breaks(G.nodes[node]['text'])}")
            else:
                node_annotations.append(f"<b>Topic:</b> {node} <br><br>{add_line_breaks(G.nodes[node]['text'])}" + '<br><br>')

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=[f'<b>{x.replace(" ", "<br>")}</b>' for x in node_text],
            mode='markers+text',
            hovertext=node_annotations,
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=0, color='gray'),
                symbol=node_shape,
            )
        )
        color_scatters = []
        for color, name in zip(['green','red', 'blue'],legend):
            tmp = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=100,
                color=color,
                symbol='circle'
            ),
            name=name, showlegend=True)

            color_scatters.append(tmp)

        legend_annotations = [] #dict(showarrow=False, x=0.1, y=1, text='<b>Legend</b>', xanchor='left', xref='paper', yref='paper')
        for i, (color, symbol, name) in enumerate(zip(['green','red', 'blue'],['&#9679;', '&#9679;','&#9670;'], legend)):
            tmp = dict(showarrow=False, x=0.1, y=1-(i+1)*0.02, text=f'<span style="font-size:30px; color:{color};">{symbol}</span>&nbsp;&nbsp;&nbsp;<span style="font-size:18px;">{name}</span>', xanchor='left', xref='paper', yref='paper')
            legend_annotations.append(tmp)

        # Create the Plotly figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(template='simple_white',
                        title=dict(font=dict(size=16)),
                        showlegend=False,
                        legend=dict(
                                itemsizing='constant'
                            ),
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=legend_annotations,
                        scene=dict(
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False),
                            zaxis=dict(showgrid=False, zeroline=False),
                        ),
                        height=1400,  # Increase height
                        width=1600   # Increase width
                    )
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.show()
    
    else:
        # Static matplotlib version
        pos = nx.spring_layout(G, k=0.1, iterations=10, seed=1)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=2, alpha=0.8)
        
        # Separate nodes by type and color
        entity_nodes = []
        topic_nodes = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            if node_data['shape'] == 'circle':  # Entity nodes
                entity_nodes.append(node)
            else:  # Topic nodes (diamond)
                topic_nodes.append(node)
        
        # Draw entity nodes (circles) with their colors
        for node in entity_nodes:
            node_data = G.nodes[node]
            color = node_data['color']
            size = node_data['size'] * 3  # Scale for matplotlib
            
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, 
                                 node_size=size, node_shape='o', ax=ax, alpha=0.8)
        
        # Draw topic nodes (diamonds) in blue
        if topic_nodes:
            topic_sizes = [G.nodes[node]['size'] * 3 for node in topic_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_color='blue',
                                 node_size=topic_sizes, node_shape='D', ax=ax, alpha=0.8)
        
        # Add node labels
        labels = {}
        for node in G.nodes():
            # Truncate long names for better display
            label = node.replace(" ", "\n") if len(node) > 15 else node
            labels[node] = label
        
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight='bold')
        
        # Create custom legend
        legend_elements = []
        colors = ['green', 'red', 'blue']
        shapes = ['o', 'o', 'D']  # circle, circle, diamond
        
        for i, (color, shape, name) in enumerate(zip(colors, shapes, legend)):
            if shape == 'D':
                # For diamond shape, use different marker
                legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=color,
                                                markersize=15, label=name))
            else:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                markersize=15, label=name))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=12)
        
        # Customize plot appearance
        ax.set_title('Network Analysis', fontsize=24, fontweight='bold', pad=20)
        ax.axis('off')  # Hide axes
        
        # Set equal aspect ratio and tight layout
        ax.set_aspect('equal')
        plt.tight_layout()
        
        # Add some padding around the network
        x_values = [pos[node][0] for node in G.nodes()]
        y_values = [pos[node][1] for node in G.nodes()]
        x_margin = (max(x_values) - min(x_values)) * 0.1
        y_margin = (max(y_values) - min(y_values)) * 0.1
        
        ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
        ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
        
        plt.show()

def generate_pyvis_chart(G, legend):
    # Get 3D positions for the nodes
    pos = nx.spring_layout(G, k=0.1, iterations=10, seed=2)
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
    net.barnes_hut(overlap=1,) # damping=0,spring_strength=0.1,spring_length=250,) #gravity=-1000)
    net.toggle_physics(False)
    net.set_options('{"layout":{"randomSeed":1}}')

    return net.show('interactive_network.html')