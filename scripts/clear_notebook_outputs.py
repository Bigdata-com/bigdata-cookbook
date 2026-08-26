"""Clear code-cell outputs so nbconvert can read legacy notebooks with invalid output metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def clear_outputs(notebook_path: Path) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    nbformat.write(nb, notebook_path)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: clear_notebook_outputs.py <notebook.ipynb>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    clear_outputs(path)
    print(f"cleared outputs: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
