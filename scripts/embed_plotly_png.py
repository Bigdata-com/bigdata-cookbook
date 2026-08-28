"""Make Plotly notebook outputs render on GitHub.

GitHub's notebook viewer does not execute Plotly JavaScript. It only shows
static ``image/png`` (or SVG). This script:

1. Embeds a kaleido PNG into each ``application/vnd.plotly.v1+json`` output.
2. Drops orphan Plotly HTML-only outputs (no figure JSON / PNG) that blank out
   GitHub and bloat notebooks with an inlined plotly.js bundle.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

import nbformat
import plotly.graph_objects as go
import plotly.io as pio

PLOTLY_JSON = "application/vnd.plotly.v1+json"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def _figure_from_plotly_payload(payload: Any) -> go.Figure:
    if isinstance(payload, str):
        return pio.from_json(payload)
    if isinstance(payload, dict):
        return go.Figure(payload)
    raise TypeError(f"Unsupported Plotly payload type: {type(payload)!r}")


def _is_plotly_html(html: str) -> bool:
    sample = html[:12_000].lower()
    return "plotly.js" in sample or "plotlyconfig" in sample or "plotly-graph-div" in sample


def embed_png_in_output(output: dict[str, Any]) -> bool:
    if output.get("output_type") != "display_data":
        return False
    data = output.get("data")
    if not isinstance(data, dict):
        return False
    if data.get("image/png"):
        return False
    if PLOTLY_JSON not in data:
        return False

    fig = _figure_from_plotly_payload(data[PLOTLY_JSON])
    png_bytes = pio.to_image(fig, format="png", scale=2)
    data["image/png"] = base64.b64encode(png_bytes).decode("ascii")
    return True


def is_orphan_plotly_html(output: dict[str, Any]) -> bool:
    """True for Plotly HTML dumps with neither figure JSON nor a static image."""
    if output.get("output_type") != "display_data":
        return False
    data = output.get("data")
    if not isinstance(data, dict):
        return False
    if PLOTLY_JSON in data or data.get("image/png") or data.get("image/svg+xml"):
        return False
    html = _as_text(data.get("text/html"))
    return bool(html) and _is_plotly_html(html)


def fix_notebook(nb: nbformat.NotebookNode) -> tuple[int, int]:
    embedded = 0
    removed = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        outputs = list(cell.get("outputs", []))
        kept: list[Any] = []
        for output in outputs:
            if isinstance(output, dict) and is_orphan_plotly_html(output):
                removed += 1
                continue
            if isinstance(output, dict) and embed_png_in_output(output):
                embedded += 1
            kept.append(output)
        cell["outputs"] = kept
    return embedded, removed


def fix_notebook_file(notebook_path: Path) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    embedded, removed = fix_notebook(nb)
    nbformat.write(nb, notebook_path)
    print(
        f"fixed plotly github render: {notebook_path} "
        f"(png_embedded={embedded}, orphan_html_removed={removed})"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: embed_plotly_png.py <notebook.ipynb>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: notebook not found: {path}", file=sys.stderr)
        return 1
    fix_notebook_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
