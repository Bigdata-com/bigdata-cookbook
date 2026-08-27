"""Embed PNG fallbacks in Plotly notebook outputs for static viewers (e.g. GitHub)."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

import nbformat
import plotly.graph_objects as go
import plotly.io as pio


def _figure_from_plotly_payload(payload: Any) -> go.Figure:
    if isinstance(payload, str):
        return pio.from_json(payload)
    if isinstance(payload, dict):
        return go.Figure(payload)
    raise TypeError(f"Unsupported Plotly payload type: {type(payload)!r}")


def embed_png_in_output(output: dict[str, Any]) -> bool:
    if output.get("output_type") != "display_data":
        return False
    data = output.get("data")
    if not isinstance(data, dict):
        return False
    if "image/png" in data and data["image/png"]:
        return False
    plotly_key = "application/vnd.plotly.v1+json"
    if plotly_key not in data:
        return False

    fig = _figure_from_plotly_payload(data[plotly_key])
    png_bytes = pio.to_image(fig, format="png", scale=2)
    data["image/png"] = base64.b64encode(png_bytes).decode("ascii")
    return True


def embed_plotly_png(nb: nbformat.NotebookNode) -> int:
    embedded = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if isinstance(output, dict) and embed_png_in_output(output):
                embedded += 1
    return embedded


def embed_plotly_png_file(notebook_path: Path) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    embedded = embed_plotly_png(nb)
    nbformat.write(nb, notebook_path)
    print(f"embedded plotly png: {notebook_path} (outputs={embedded})")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: embed_plotly_png.py <notebook.ipynb>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: notebook not found: {path}", file=sys.stderr)
        return 1
    embed_plotly_png_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
