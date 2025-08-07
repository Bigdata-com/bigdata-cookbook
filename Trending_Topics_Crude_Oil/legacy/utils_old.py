from . import keywords_functions
from .label_extractor import extract_label
import pandas as pd
import asyncio
import os
from bigdata import Bigdata
from bigdata.query import Similarity
from bigdata.models.search import DocumentType, SortBy
from bigdata.daterange import RollingDateRange, AbsoluteDateRange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
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

def create_similarity_query(sentences):
    if not sentences:
        raise ValueError("The list of sentences must not be empty.")

    query = Similarity(sentences[0])

    for sentence in sentences[1:]:
        query &= Similarity(sentence)
    
    return query

def create_dates_lists(start_date, end_date, freq='W'):
    from datetime import datetime
    ## Convert start_date and end_date to datetime objects if they are not already
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    # Initialize lists
    start_list = []
    end_list = []
    # Generate date ranges
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    for i in range(len(date_range) - 1):
        start_list.append(date_range[i].strftime('%Y-%m-%d 00:00:00'))
        end_list.append((date_range[i + 1] - pd.Timedelta(days=1)).strftime('%Y-%m-%d 23:59:59'))
    # Handle the last range
    start_list.append(date_range[-1].strftime('%Y-%m-%d 00:00:00'))
    end_list.append(end_date.strftime('%Y-%m-%d 23:59:59'))
    return start_list, end_list

# Define the mask processing function
def mask_entity_coordinates(input_df, column_masked_text, mask_target, mask_other, entity_key_to_name):
    for idx, row in input_df.iterrows():
        text = row['text']
        entities = row['entities']
        entities.sort(key=lambda x: x['start'], reverse=True)  # Sort entities by start position in reverse order
        masked_text = text

        for entity in entities:
            entity_name = entity_key_to_name.get(entity['key'])
            start, end = entity['start'], entity['end']
            if entity['key'] == row['rp_entity_id']:
                masked_text = masked_text[:start] + mask_target + masked_text[end:]
            else:
                masked_text = masked_text[:start] + mask_other + masked_text[end:]

        input_df.at[idx, column_masked_text] = masked_text
    return input_df

def clean_text(text):
    text = text.replace('{', '').replace('}', '')
    return text

def clean_text_masked(text):
    text = text.replace('{', '').replace('}', '')
    return text

def process_sentences_masking(sentences: pd.DataFrame, entity_key_to_name, target_entity_mask: str,  other_entity_mask: str, masking: bool = False) -> pd.DataFrame:
    sentences['text'] = sentences['text'].str.replace('{','',regex=False)
    sentences['text'] = sentences['text'].str.replace('}','',regex=False)

    if masking:
        sentencesdf = mask_entity_coordinates(input_df = sentences, 
                                column_masked_text = 'masked_text',
                                mask_target = target_entity_mask,
                                mask_other = other_entity_mask,
                                entity_key_to_name = entity_key_to_name
                               )

        sentencesdf['masked_text'] = sentencesdf['masked_text'].apply(clean_text_masked)

        sentencesdf = sentencesdf[sentencesdf.masked_text!='to_remove']
    else:
        sentencesdf = sentences

    sentencesdf['text'] = sentencesdf['text'].apply(clean_text)
    sentencesdf = sentencesdf[sentencesdf.text!='to_remove']

    return sentencesdf.reset_index(drop = True)

def plot_labels_distribution(df, labels_names_dict):
    
    total_records = len(df)

    # Create the main plot
    plt.figure(figsize=(12, 6))

    # Main plot

    rating_c = df['label'].value_counts(normalize=False)
    rating_c = rating_c.sort_index()

    # Create the side plot for percentages
    plt.subplot(1, 2, 1)
    ax1=sns.barplot(x=rating_c.index, y=rating_c.values, palette='viridis', hue=rating_c.index)
    plt.title('Distribution of Labels (Total Records: {})'.format(total_records))
    plt.xlabel('Labels')
    plt.ylabel('Count')
    ax1.set_xticks(range(len(rating_c.index)))
    ax1.set_xticklabels([labels_names_dict[item.get_text()] for item in ax1.get_xticklabels()])


    # Add percentages to the top of the bars
    for index, value in enumerate(rating_c.values):
        plt.text(index, value, f'{value:.0f}', ha='center', va='bottom')


    # Calculate percentages
    rating_percentages = df['label'].value_counts(normalize=True) * 100
    rating_percentages = rating_percentages.sort_index()

    # Create the side plot for percentages
    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x=rating_percentages.index, y=rating_percentages.values, palette='viridis',hue=rating_percentages.index)
    plt.title('Percentage Distribution of Labels')
    plt.xlabel('Labels')
    plt.ylabel('Percentage')
    ax2.set_xticks(range(len(rating_percentages.index)))
    ax2.set_xticklabels([labels_names_dict[item.get_text()] for item in ax2.get_xticklabels()])

    # Add percentages to the top of the bars
    for index, value in enumerate(rating_percentages.values):
        plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



def run_similarity_search(query, start_date, end_date, bigdata_credentials, freq = 'MS', limit_documents = 1000):
    sentences_final = []
    bigdata = bigdata_credentials
    start_list, end_list = create_dates_lists(start_date, end_date, freq=freq)

    for period in tqdm(range(1,len(start_list)+1,1)): #,desc='Loading and saving datasets for '+'US ML'+' universe'):
        start_date = start_list[period-1]
        end_date = end_list[period-1]
        date_range = AbsoluteDateRange(start_date, end_date)
        search = bigdata.content_search.new_from_query(
            query=query,
            date_range= date_range,
            scope=DocumentType.NEWS,
            sortby=SortBy.RELEVANCE
        )

        # Limit the number of documents to retrieve
        results = search.limit_documents(limit_documents)

        # Collect unique entity keys
        entity_keys = set()
        for result in results:
            for chunk in result.chunks:
                for entity in chunk.entities:
                    entity_keys.add(entity.key)

        # Lookup entity details by their keys
        entities_lookup = bigdata.knowledge_graph.get_entities(list(entity_keys))

        # Create a mapping of entity keys to names and types
        entity_key_to_name = {}
        for entity in entities_lookup:
            if hasattr(entity, 'id') and hasattr(entity, 'name') and entity.entity_type == 'COMP':
                entity_key_to_name[entity.id] = entity.name

        # Create lists to store the DataFrame rows
        data_rows = []

        # Filter and store the results with known entity names and coordinates
        for result in results:
            chunk_texts_with_entities = []
            for chunk_index, chunk in enumerate(result.chunks):
                chunk_entities = []
                for entity in chunk.entities:
                    entity_name = entity_key_to_name.get(entity.key, 'Unknown')
                    if entity_name != 'Unknown':
                        chunk_entities.append({
                            'key': entity.key,
                            'name': entity_name,
                            'start': entity.start,
                            'end': entity.end
                        })
                if chunk_entities:
                    chunk_texts_with_entities.append((chunk.text, chunk_entities, chunk_index))

            for chunk_text, chunk_entities, chunk_index in chunk_texts_with_entities:
                for entity in chunk_entities:
                    other_entities = [e for e in chunk_entities if e['name'] != entity['name']]
                    data_rows.append({
                        'timestamp_utc': result.timestamp,
                        'rp_document_id': result.id,
                        'sentence_id': f"{result.id}-{chunk_index}",
                        'rp_entity_id': entity['key'],
                        'entity_name': entity['name'],
                        'text': chunk_text,
                        'other_entities': ', '.join([e['name'] for e in other_entities]),
                        'entities': chunk_entities
                    })

        # Create the DataFrame
        df = pd.DataFrame(data_rows)

        # Remove duplicate rows
        df = df.drop_duplicates(subset=['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id'])

        # Process the sentences to add masked text
        sentences = process_sentences_masking(df, entity_key_to_name, target_entity_mask='Target Company', other_entity_mask='Other Company', masking=True, )

        sentences_final.append(sentences)

    sentences = pd.concat(sentences_final)
    
    return sentences



async def process_sentences(sentences, sentence_column_2, masked_2, system_prompt, path_training, n_expected_response_tokens, batch_size, model, open_ai_credentials):
    sentence_final = []
    for i in range(0, sentences.shape[0], 100):
        start_idx = i
        end_idx = min(i + 100, sentences.shape[0])

        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                sentences_labels_2 = await extract_label(
                    sentences=sentences[start_idx:end_idx],
                    sentence_column=sentence_column_2,
                    masked=masked_2,  # defines whether the name of the entity is hidden (masked) from the LLM.
                    system_prompt=system_prompt,
                    path_call_hash_location=path_training,
                    path_result_hash_location=path_training,
                    n_expected_response_tokens=n_expected_response_tokens,
                    batch_size=batch_size,
                    concurrency=200,  # defines the number of calls made simultaneously
                    openai_api_key=open_ai_credentials,
                    parameters={
                        'model': model,  # gpt-4-0125-preview
                        'temperature': 0,
                        'response_format': {'type': 'json_object'}
                    }
                )
                sentences_labels_2 = sentences_labels_2
                sentence_final.append(sentences_labels_2)
                break  # Break out of retry loop if successful

            except Exception as e:
                print(f"Error on attempt {attempt + 1} for batch {start_idx} to {end_idx}: {e}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(100)  # Wait for 100 seconds before retrying
                else:
                    print(f"Failed after {retry_attempts} attempts for batch {start_idx} to {end_idx}")


    return pd.concat(sentence_final)

# Call the async function using an event loop

def run_prompt(sentences, sentence_column_2, masked_2, system_prompt_ai_classification, path_training, n_expected_response_tokens, batch_size, model, open_ai_credentials):
    df_ai_classif = asyncio.run(
        process_sentences(
            sentences=sentences,
            sentence_column_2=sentence_column_2,
            masked_2=masked_2,
            system_prompt=system_prompt_ai_classification,
            path_training=path_training,
            n_expected_response_tokens=n_expected_response_tokens,
            batch_size=batch_size,
            model = model,
            open_ai_credentials = open_ai_credentials
        )
    )
    df_ai_classif.reset_index(inplace = True, drop = True)
    return df_ai_classif

def top_companies_sector(df_filtered, bigdata_credentials): 
    bigdata = bigdata_credentials
    entity_news_count = defaultdict(int)

    # Lookup entity details by their keys
    entities_lookup = bigdata.knowledge_graph.get_entities(df_filtered['rp_entity_id'].unique().tolist())

    # Create a mapping of entity keys to names and sectors
    entity_key_to_name = {}
    entity_key_to_sector = {}
    for entity in entities_lookup:
        if hasattr(entity, 'id') and hasattr(entity, 'name') and entity.entity_type == 'COMP':
            entity_key_to_name[entity.id] = entity.name
            if hasattr(entity, 'sector'):
                entity_key_to_sector[entity.id] = entity.sector

    # Map sectors to the DataFrame
    df_filtered['sector'] = df_filtered['rp_entity_id'].map(entity_key_to_sector)

    df_news_count = df_filtered.groupby(['rp_entity_id']).agg({'sentence_id': pd.Series.nunique}).reset_index()

    df_news_count.rename(columns={'sentence_id': 'NewsCount'}, inplace=True)

    df_news_count = df_news_count.merge(df_filtered[['rp_entity_id', 'entity_name', 'sector', 'text', 'motivation']], how='left', on='rp_entity_id')
    #df_news_count = df_news_count.loc[~df_news_count['sector'].isin(['Technology'])]

    # Drop duplicates to ensure each company is counted only once
    df_news_count = df_news_count.drop_duplicates(subset=['rp_entity_id'])

    # Get the top 5 companies by sector
    top_5_by_sector = df_news_count.groupby('sector', group_keys=False).apply(lambda x: x.nlargest(5, 'NewsCount')).reset_index(drop=True)

    # Rank sectors by the total number of news mentions
    sector_ranking = (top_5_by_sector.groupby('sector')['NewsCount']
                      .sum()
                      .sort_values(ascending=False)
                      .index)

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

    # Determine the number of rows needed for 2 columns
    n_cols = 5
    n_rows = (len(sector_ranking) + n_cols - 1) // n_cols

    # Prepare the layout for subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,  # Adjust the number of columns to accommodate the bar plots
        subplot_titles=[f"{sector} Sector" for sector in sector_ranking],  # Title for each bar plot
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}] * n_cols for _ in range(n_rows)]
    )

    # Flatten the sector ranking for easy iteration
    sector_list = list(sector_ranking)

    for i, sector in enumerate(sector_list):
        row = i // n_cols + 1
        col = (i % n_cols) + 1

        # Plot bar chart for top 5 companies in the sector
        sector_data = top_5_by_sector[top_5_by_sector['sector'] == sector]

        # Create custom hover text with text and motivation, adding dynamic line breaks for better readability
        hover_text = [
            f"<b>Entity:</b> {row.entity_name}<br><br><b>Text</b>: {add_line_breaks(row.text)}<br><br><b>Motivation</b>: {add_line_breaks(row.motivation)}"
            for row in sector_data.itertuples()
        ]

        fig.add_trace(
            go.Bar(
                x=sector_data['entity_name'],
                y=sector_data['NewsCount'],
                name=f"{sector} Sector",
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
        height=800 * n_rows,
        title_text="Top Companies by Sector",
    )

    # Show the interactive plot
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
    entity_texts_motivations = df_filtered.groupby(['entity_name', 'Day']).apply(lambda x: x[['text', 'motivation']].to_dict('records')).to_dict()

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
                            f"<b>Text</b>: {add_line_breaks(record['text'])}<br><br><b>Motivation</b>: {add_line_breaks(record['motivation'])}"
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
    
def provider_user_net(df_filtered_ai_cost_reduction, df_filtered_ai_providers):
    
    # Exclude 'motivation' column during the merge
    cols_to_merge = ['timestamp_utc', 'rp_document_id', 'sentence_id', 'rp_entity_id', 'entity_name', 'text', 'masked_text']
    merged_df = pd.merge(df_filtered_ai_cost_reduction[cols_to_merge], df_filtered_ai_providers[cols_to_merge], how='outer', indicator=True)

    # Filter out common rows
    df_filtered_ai_cost_reduction_without_common = df_filtered_ai_cost_reduction.copy(deep = True)
    df_filtered_ai_providers_without_common = df_filtered_ai_providers.copy(deep = True)

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
            line=dict(width=4, color='gray'),
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
            node_text.append(node)
            node_color.append('blue' if node.endswith('_user') else 'red')
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
                size=15,
                line_width=2
            )
        )

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Network of Companies Mentioned in AI Cost-Cutting Context',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=20, r=20, t=40),  # Reduced left and right margins
                            annotations=[dict(
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 )],
                            scene=dict(
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False),
                                zaxis=dict(showgrid=False, zeroline=False),
                            ),
                            height=1600,  # Increase height
                            width=2000   # Increase width
                        )
        )

        fig.show()