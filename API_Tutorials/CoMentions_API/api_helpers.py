"""Helper functions for the Bigdata.com Co-mentions API."""

from __future__ import annotations

import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from requests import Session


def create_comentions_network_graph(
    session: Session,
    kg_entities_endpoint: str,
    center_name: str,
    center_id: str,
    connected_entities: list[dict[str, Any]],
    category_name: str,
    text: str,
    max_nodes: int = 20,
) -> Figure | None:
    """Create a static network graph that renders reliably on GitHub."""
    if not connected_entities:
        return None

    filtered_entities = [entity for entity in connected_entities if entity.get("id") != center_id]
    entities = sorted(
        filtered_entities,
        key=lambda entity: entity.get("total_chunks_count", 0),
        reverse=True,
    )[:max_nodes]
    if not entities:
        return None

    # Resolve entity names
    entity_ids = [entity["id"] for entity in entities]
    response = session.post(kg_entities_endpoint, json={"values": entity_ids})

    if response.status_code != 200:
        return None

    resolved: dict[str, dict[str, Any]] = response.json().get("results", {})

    # Create node positions (center + circular arrangement)
    n_connected = len(entities)
    angles = np.linspace(0, 2 * np.pi, n_connected, endpoint=False)

    node_x = 2 * np.cos(angles)
    node_y = 2 * np.sin(angles)
    chunk_counts = [int(entity.get("total_chunks_count", 0)) for entity in entities]
    # Some categories report only headline counts, leaving every chunk count at zero.
    max_chunks = max(chunk_counts) or 1
    node_sizes = [350 + (1_100 * chunks / max_chunks) for chunks in chunk_counts]

    fig, axis = plt.subplots(figsize=(12, 8))
    for x_coordinate, y_coordinate in zip(node_x, node_y, strict=True):
        axis.plot([0, x_coordinate], [0, y_coordinate], color="#9CA3AF", linewidth=1, zorder=1)

    axis.scatter(node_x, node_y, s=node_sizes, color="#4ECDC4", edgecolor="white", zorder=2)
    axis.scatter([0], [0], s=1_800, color="#FF6B6B", edgecolor="white", zorder=3)
    axis.annotate(
        f"{center_name}\n({center_id})",
        (0, 0),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=4,
    )

    for x_coordinate, y_coordinate, entity, chunks in zip(
        node_x,
        node_y,
        entities,
        chunk_counts,
        strict=True,
    ):
        entity_id = str(entity["id"])
        name = str(resolved.get(entity_id, {}).get("name", entity_id))
        headlines = int(entity.get("total_headlines_count", 0))
        label = f"{textwrap.fill(name, 16)}\n{chunks:,} chunks | {headlines:,} headlines"
        axis.annotate(label, (x_coordinate, y_coordinate), ha="center", va="center", fontsize=7)

    query_label = text or "No topic filter"
    axis.set_title(
        f"{category_name.capitalize()} connected to {center_name}\n"
        f'Query: "{query_label}" | Top {len(entities)} by chunk count',
        pad=20,
    )
    axis.set_xlim(-2.8, 2.8)
    axis.set_ylim(-2.5, 2.5)
    axis.set_aspect("equal")
    axis.axis("off")
    fig.tight_layout()
    return fig
