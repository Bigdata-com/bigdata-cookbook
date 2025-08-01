import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def lookup_sector_information(df, bigdata_cred):
    """
    Placeholder for sector lookup function
    In the original this calls a bigdata service
    """
    # For now, just return the dataframe as-is
    # You can implement the actual lookup if needed
    return df

def provider_adopter_net(df_filtered_adopters,
                      df_filtered_providers, bigdata_cred,
                      min_occurrences=1,
                      dash=False,
                      interactive=True):
    """
    Create provider-adopter network analysis based on co-mentions in documents.
    Identical to the original provider_user_net function with adaptations for liquid cooling workflow.
    
    Parameters:
    -----------
    df_filtered_adopters : pd.DataFrame
        DataFrame with adopter companies (label "A")
    df_filtered_providers : pd.DataFrame  
        DataFrame with provider companies (label "P")
    bigdata_cred : object
        Credentials for bigdata service (can be None)
    min_occurrences : int, default=1
        Minimum number of occurrences for a company to be included in the analysis
    dash : bool, default=False
        If True, return figure object. If False, display the plot
    interactive : bool, default=True
        If True, create interactive 3D Plotly visualization. If False, create static 2D matplotlib plot
        
    Returns:
    --------
    plotly.graph_objects.Figure or matplotlib.figure.Figure or None
        Network visualization figure (Plotly if interactive=True, matplotlib if interactive=False)
    """
    
    # Exclude 'motivation' column during the merge
    cols_to_merge = [
        'Date', 'Document ID', 'sentence_id', 'rp_entity_id',
        'Company', 'Quote', 'masked_text'
    ]
    merged_df = pd.merge(df_filtered_adopters[cols_to_merge],
                         df_filtered_providers[cols_to_merge],
                         how='outer',
                         indicator=True)

    df_filtered_adopters = lookup_sector_information(df_filtered_adopters, bigdata_cred)
    df_filtered_providers = lookup_sector_information(df_filtered_providers, bigdata_cred)
    
    # Filter companies by minimum occurrences
    if min_occurrences > 1:
        
        # Count occurrences for adopters
        adopter_counts = df_filtered_adopters['Company'].value_counts()
        valid_adopters = adopter_counts[adopter_counts >= min_occurrences].index
        df_filtered_adopters = df_filtered_adopters[df_filtered_adopters['Company'].isin(valid_adopters)]
        
        # Count occurrences for providers  
        provider_counts = df_filtered_providers['Company'].value_counts()
        valid_providers = provider_counts[provider_counts >= min_occurrences].index
        df_filtered_providers = df_filtered_providers[df_filtered_providers['Company'].isin(valid_providers)]

    # Filter out common rows
    df_filtered_adopters_without_common = df_filtered_adopters.copy(
        deep=True)
    df_filtered_providers_without_common = df_filtered_providers.copy(
        deep=True)

    df_filtered_adopters_without_common[
        'Company'] = df_filtered_adopters_without_common[
            'Company'] + '_adopter'
    df_filtered_providers_without_common[
        'Company'] = df_filtered_providers_without_common[
            'Company'] + '_provider'

    # Concatenate the dataframes
    df_filtered = pd.concat([
        df_filtered_adopters_without_common,
        df_filtered_providers_without_common
    ],
                            ignore_index=True)

    # Create a network graph
    G = nx.Graph()

    # Initialize a dictionary to count co-mentions and store chunk text and motivations
    co_mention_counts = defaultdict(lambda: {
        'count': 0,
        'texts': [],
        'motivations': defaultdict(list),
        'headlines': defaultdict(list)
    })

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
        entities = group['Company'].unique()
        entity_id_map = dict(zip(group['Company'], group['Company']))
        if len(entities) > 1:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    # Ensure only provider-adopter pairs are considered
                    if ('_provider' in entities[i] and '_adopter'
                            in entities[j]) or ('_adopter' in entities[i] and
                                                '_provider' in entities[j]):
                        pair = tuple(sorted([entities[i], entities[j]]))
                        co_mention_counts[pair]['count'] += 1
                        truncated_text = truncate_text(group.iloc[0]['Quote'],
                                                       entities[i],
                                                       entities[j])
                        co_mention_counts[pair]['texts'].append(truncated_text)
                        for entity in entities:
                            entity_motivations = group[
                                group['Company'] ==
                                entity]['Motivation'].tolist()
                            entity_headlines = group[
                                group['Company'] ==
                                entity]['Headline'].tolist()
                            for idx, (motivation, headline) in enumerate(zip(
                                    entity_motivations, entity_headlines)):
                                entity_motivations[idx] = motivation.replace(
                                    f'Target Company_', entity_id_map[entity])
                                # Remove the '_adopter' and '_provider' tags from the motivation
                                entity_motivations[idx] = entity_motivations[
                                    idx].replace('_adopter',
                                                 '').replace('_provider', '')
                            co_mention_counts[pair]['motivations'][entity].extend(entity_motivations)
                            co_mention_counts[pair]['headlines'][entity].extend(entity_headlines)

    # Add edges to the graph based on co-mentions
    for pair, data in co_mention_counts.items():
        if data['count'] >= 1:  # Adjust threshold here if needed
            G.add_edge(pair[0],
                       pair[1],
                       weight=data['count'],
                       texts=data['texts'],
                       motivations=data['motivations'],
                       headlines = data['headlines'])

    # Check if the graph is empty
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("The graph is empty. Please check the data and conditions.")
        return None
    
    if interactive:
        # Interactive 3D Plotly version
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

        edge_trace = go.Scatter3d(x=edge_x,
                                  y=edge_y,
                                  z=edge_z,
                                  line=dict(width=4, color='darkgrey'),
                                  hoverinfo='none',
                                  mode='lines',
                                  showlegend=False,
                                  name="")

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
            clean_node_text = node.replace('_adopter',
                                           '').replace('_provider', '')
            node_text.append(clean_node_text)
            node_color.append(
                'royalblue' if node.endswith('_adopter') else 'tomato')
            # Remove duplicate chunks and truncate text
            unique_texts = [
                chunk for edge in G.edges(node, data=True)
                for chunk in edge[2]['texts']
            ]
            unique_texts = list(dict.fromkeys(
                unique_texts))  # Remove duplicates while preserving order
            formatted_texts = [
                f"<b>Text {i+1}:</b><br>" +
                '<br>'.join([text[j:j + 80] for j in range(0, len(text), 80)])
                for i, text in enumerate(unique_texts)
            ]

            # Format motivations
            formatted_motivations = []
            motivations_added = set()
            for i, text in enumerate(unique_texts):
                for edge in G.edges(node, data=True):
                    motivations = edge[2]['motivations'][node]
                    for motivation in motivations:
                        if motivation not in motivations_added:
                            formatted_motivations.append(
                                f"<b>Motivation {len(formatted_motivations)+1}:</b><br>"
                                + '<br>'.join([
                                    motivation[j:j + 80]
                                    for j in range(0, len(motivation), 80)
                                ]))
                            motivations_added.add(motivation)
            
            formatted_headlines = []
            headlines_added = set()
            for i, text in enumerate(unique_texts):
                for edge in G.edges(node, data=True):
                    headlines = edge[2]['headlines'][node]
                    for headline in headlines:
                        if headline not in headlines_added:
                            formatted_headlines.append(
                                f"<b>Headline {len(formatted_headlines)+1}:</b><br>"
                                + '<br>'.join([
                                    headline[j:j + 80]
                                    for j in range(0, len(headline), 80)
                                ]))
                            headlines_added.add(headline)

            # Add company name as first field in hover, followed by existing information
            company_header = f"<b>Company:</b> {clean_node_text}<br><br>"
            content_sections = [formatted_headline+'<br><br>'+formatted_motivation+'<br><br>'+formatted_text for formatted_headline, formatted_text, formatted_motivation in zip(formatted_headlines, formatted_texts, formatted_motivations)]
            hover_content = company_header + '<br><br>'.join(content_sections)
            node_annotations.append(hover_content)

        node_trace = go.Scatter3d(x=node_x,
                                  y=node_y,
                                  z=node_z,
                                  mode='markers+text',
                                  text=node_text,
                                  hovertext=node_annotations,
                                  hoverinfo='text',
                                  marker=dict(color=node_color,
                                              size=8,
                                              opacity=0.8,
                                              line_width=1),
                                  showlegend=False,
                                  name="")

        # Create legend traces (invisible points just for legend)
        legend_adopter = go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(color='royalblue', size=10),
            name='Adopters',
            showlegend=True
        )
        
        legend_provider = go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(color='tomato', size=10),
            name='Providers',
            showlegend=True
        )

        # Create the figure
        fig = go.Figure(
            data=[edge_trace, node_trace, legend_adopter, legend_provider],
            layout=go.Layout(
                title="Provider-Adopter Network",
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20,
                            t=40),  # Reduced left and right margins
                scene=dict(
                    xaxis=dict(showgrid=True,
                               zeroline=True,
                               backgroundcolor='#D3D3D3',
                               gridcolor='rgba(255, 255, 255, 0.1)'),
                    yaxis=dict(showgrid=True,
                               zeroline=True,
                               backgroundcolor='#D3D3D3',
                               gridcolor='rgba(255, 255, 255, 0.1)'),
                    zaxis=dict(showgrid=True,
                               zeroline=True,
                               backgroundcolor='#D3D3D3',
                               gridcolor='rgba(255, 255, 255, 0.1)'),
                ),height=800,  # Increase height
                width=1000   # Increase width
            ))

        if dash:
            return fig
        else:
            fig.show()
            return fig
    
    else:
        # Static 2D matplotlib version
        # Get the positions of the nodes using a spring layout (2D)
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgrey', width=3, alpha=0.6)
        
        # Separate nodes by type
        adopter_nodes = [node for node in G.nodes() if node.endswith('_adopter')]
        provider_nodes = [node for node in G.nodes() if node.endswith('_provider')]
        
        # Draw adopter nodes (blue)
        if adopter_nodes:
            adopter_pos = {node: pos[node] for node in adopter_nodes}
            nx.draw_networkx_nodes(G, adopter_pos, nodelist=adopter_nodes, 
                                 node_color='royalblue', node_size=300, alpha=0.8, ax=ax)
        
        # Draw provider nodes (red)
        if provider_nodes:
            provider_pos = {node: pos[node] for node in provider_nodes}
            nx.draw_networkx_nodes(G, provider_pos, nodelist=provider_nodes, 
                                 node_color='tomato', node_size=300, alpha=0.8, ax=ax)
        
        # Draw labels (clean node names)
        labels = {node: node.replace('_adopter', '').replace('_provider', '') for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
        
        # Set title and clean up axes
        ax.set_title("Provider-Adopter Network", fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='royalblue', label='Adopters'),
            Patch(facecolor='tomato', label='Providers')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if dash:
            return fig
        else:
            plt.show()
            return fig

