"""
API Helper Functions for Bigdata.com Co-mentions API

This module provides utility functions for the Co-mentions API,
including network graph visualization.
"""

from typing import List, Dict, Any


def create_comentions_network_graph(
    session,
    kg_entities_endpoint: str,
    center_name: str,
    center_id: str,
    connected_entities: List[dict],
    category_name: str,
    text: str,
    max_nodes: int = 20
):
    """
    Create a network graph with center entity and connected entities.
    
    Args:
        session: requests.Session with authentication
        kg_entities_endpoint: KG entities endpoint URL
        center_name: Name of the center entity
        center_id: ID of the center entity
        connected_entities: List of connected entity dicts
        category_name: Name of the category (for title)
        text: Query text (for title)
        max_nodes: Maximum number of connected nodes to show
    
    Returns:
        plotly Figure or None
    """
    import plotly.graph_objects as go
    import numpy as np
    
    if not connected_entities:
        return None
    
    # Limit nodes and sort by total_chunks_count
    entities = sorted(connected_entities, key=lambda x: x.get('total_chunks_count', 0), reverse=True)[:max_nodes]
    
    # Resolve entity names
    entity_ids = [e['id'] for e in entities]
    response = session.post(kg_entities_endpoint, json={"values": entity_ids})
    
    if response.status_code != 200:
        return None
    
    resolved = response.json().get('results', {})
    
    # Create node positions (center + circular arrangement)
    n_connected = len(entities)
    angles = np.linspace(0, 2 * np.pi, n_connected, endpoint=False)
    
    # Center node
    node_x = [0]
    node_y = [0]
    node_text = [f"<b>{center_name}</b><br>({center_id})"]
    node_size = [50]
    node_color = ['#FF6B6B']  # Red for center
    
    # Connected nodes
    edge_x = []
    edge_y = []
    
    for i, entity in enumerate(entities):
        # Position on circle
        x = 2 * np.cos(angles[i])
        y = 2 * np.sin(angles[i])
        node_x.append(x)
        node_y.append(y)
        
        # Get resolved name
        entity_info = resolved.get(entity['id'], {})
        name = entity_info.get('name', entity['id'])
        chunks = entity.get('total_chunks_count', 0)
        headlines = entity.get('total_headlines_count', 0)
        
        node_text.append(f"<b>{name}</b><br>ID: {entity['id']}<br>Chunks: {chunks:,}<br>Headlines: {headlines:,}")
        
        # Size based on chunk count (normalized)
        size = 15 + min(35, chunks / 100)
        node_size.append(size)
        node_color.append('#4ECDC4')  # Teal for connected
        
        # Edge from center to this node
        edge_x.extend([0, x, None])
        edge_y.extend([0, y, None])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='none'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        text=[center_name[:15]] + [resolved.get(e['id'], {}).get('name', e['id'])[:12] for e in entities],
        textposition='bottom center',
        textfont=dict(size=9),
        hovertext=node_text,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f'🔗 {category_name.capitalize()} Connected to {center_name}<br><sup>Query: "{text}" | Top {len(entities)} by chunk count</sup>',
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        plot_bgcolor='white'
    )
    
    return fig
