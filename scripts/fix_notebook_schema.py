"""Fix common nbformat issues so notebooks render on GitHub.

Repairs stream outputs missing ``name`` (stdout/stderr), ensures kernelspec /
language_info metadata, adds missing cell ids, then validates with nbformat.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import nbformat
from nbformat.validator import iter_validate

DEFAULT_KERNELSPEC: dict[str, str] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

DEFAULT_LANGUAGE_INFO: dict[str, Any] = {
    "name": "python",
    "mimetype": "text/x-python",
    "pygments_lexer": "ipython3",
    "nbconvert_exporter": "python",
    "file_extension": ".py",
}


def _fix_stream_output(output: dict[str, Any]) -> bool:
    if output.get("output_type") != "stream":
        return False
    if "name" in output and output["name"] in {"stdout", "stderr"}:
        return False
    output["name"] = "stdout"
    return True


def _fix_metadata(nb: nbformat.NotebookNode) -> bool:
    changed = False
    metadata = nb.metadata
    kernelspec = metadata.get("kernelspec")
    if not isinstance(kernelspec, dict):
        metadata["kernelspec"] = dict(DEFAULT_KERNELSPEC)
        changed = True
    else:
        for key, value in DEFAULT_KERNELSPEC.items():
            if key not in kernelspec or not kernelspec[key]:
                kernelspec[key] = value
                changed = True

    language_info = metadata.get("language_info")
    if not isinstance(language_info, dict):
        metadata["language_info"] = dict(DEFAULT_LANGUAGE_INFO)
        changed = True
    elif "name" not in language_info or not language_info["name"]:
        language_info["name"] = "python"
        changed = True
    return changed


def _ensure_cell_ids(nb: nbformat.NotebookNode) -> int:
    added = 0
    for cell in nb.cells:
        if cell.get("id"):
            continue
        cell["id"] = uuid.uuid4().hex[:8]
        added += 1
    if added or any(cell.get("id") for cell in nb.cells):
        nb.nbformat_minor = max(int(getattr(nb, "nbformat_minor", 4) or 4), 5)
    return added


def fix_notebook(nb: nbformat.NotebookNode) -> tuple[int, int, int]:
    """Return counts of (stream_outputs_fixed, metadata_fields_fixed, cell_ids_added)."""
    stream_fixed = 0
    metadata_fixed = 0
    if _fix_metadata(nb):
        metadata_fixed += 1

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if isinstance(output, dict) and _fix_stream_output(output):
                stream_fixed += 1

    ids_added = _ensure_cell_ids(nb)
    return stream_fixed, metadata_fixed, ids_added


def validate_notebook(nb: nbformat.NotebookNode) -> None:
    errors = list(iter_validate(nb))
    if errors:
        message = "\n".join(str(error) for error in errors[:5])
        raise ValueError(f"Notebook validation failed:\n{message}")


def fix_notebook_file(notebook_path: Path) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    stream_fixed, metadata_fixed, ids_added = fix_notebook(nb)
    validate_notebook(nb)
    nbformat.write(nb, notebook_path)
    print(
        f"fixed schema: {notebook_path} "
        f"(stream_outputs={stream_fixed}, metadata={metadata_fixed}, cell_ids={ids_added})"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: fix_notebook_schema.py <notebook.ipynb>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: notebook not found: {path}", file=sys.stderr)
        return 1
    try:
        fix_notebook_file(path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
