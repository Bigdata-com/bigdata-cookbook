#!/usr/bin/env python3
"""Smoke test for API_Tutorials/Workflow_example (SDK-free REST + smart-batching).

Uses 5 entities and a 14-day window. Exits 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

load_dotenv(WORKFLOW_DIR / ".env")
load_dotenv(WORKFLOW_DIR.parents[2] / ".env")

ENTITY_LIMIT = 5
START_DATE = "2026-04-01"
END_DATE = "2026-04-14"
SMOKE_QUERY = "data center expansion and cloud infrastructure investment"


def _check_no_sdk_imports() -> None:
    import_patterns = (
        "from bigdata import",
        "import bigdata",
        "from bigdata_client",
        "import bigdata_client",
        "from bigdata_research_tools",
        "import bigdata_research_tools",
    )
    for py_file in (WORKFLOW_DIR / "src").rglob("*.py"):
        for line_no, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            for pattern in import_patterns:
                if pattern in stripped:
                    raise RuntimeError(f"SDK import in {py_file.name}:{line_no}: {stripped!r}")


def main() -> int:
    print("=== Workflow_example smoke test ===")
    failures: list[str] = []

    # 1. Static: no SDK in src/
    try:
        _check_no_sdk_imports()
        print("PASS  no SDK imports in src/")
    except Exception as exc:
        failures.append(f"no-sdk: {exc}")
        print(f"FAIL  no SDK imports: {exc}")

    # 2. Import surface (notebooks' shared imports)
    try:
        from src import BigDataSession  # noqa: F401
        from src.bigdata_rest import BigdataRestClient, load_universe  # noqa: F401
        from src.labeler.screener_labeler import Labeler  # noqa: F401
        from src.mindmap import generate_risk_tree  # noqa: F401
        from src.search_helper import run_universe_search  # noqa: F401

        print("PASS  shared imports")
    except Exception as exc:
        failures.append(f"imports: {exc}")
        print(f"FAIL  shared imports: {exc}")
        return 1

    # 3. REST session
    try:
        from src import BigDataSession

        session = BigDataSession()
        assert session.auth_mode == "api_key"
        print(f"PASS  BigDataSession REST auth ({session.api_base_url})")
    except Exception as exc:
        failures.append(f"session: {exc}")
        print(f"FAIL  BigDataSession: {exc}")
        return 1

    # 4. Universe load (5 entities)
    try:
        from src.bigdata_rest import company_ids_from_universe, load_universe

        universe = load_universe(WORKFLOW_DIR / "id_name_mapping_tiny.csv").head(ENTITY_LIMIT)
        company_ids = company_ids_from_universe(universe)
        assert 3 <= len(company_ids) <= ENTITY_LIMIT
        print(f"PASS  universe loaded ({len(company_ids)} entities)")
    except Exception as exc:
        failures.append(f"universe: {exc}")
        print(f"FAIL  universe: {exc}")
        return 1

    # 5. Smart-batching search (basket_filtered_entities=True via search_helper)
    try:
        from src.search_helper import run_universe_search

        id_to_name = dict(zip(universe["RP_ENTITY_ID"], universe["COMPANY_NAME"]))
        df = run_universe_search(
            company_ids,
            [SMOKE_QUERY],
            start_date=START_DATE,
            end_date=END_DATE,
            scope="news",
            chunk_percentage=0.02,
            requests_per_minute=350,
            id_to_name=id_to_name,
        )
        print(f"PASS  smart-batching search ({len(df)} chunk rows, {START_DATE}..{END_DATE})")
    except Exception as exc:
        failures.append(f"search: {exc}")
        print(f"FAIL  smart-batching search: {exc}")

    # 6. Optional LLM stubs (only if OPENAI_API_KEY set)
    import os

    if os.getenv("OPENAI_API_KEY"):
        try:
            from src.mindmap import generate_risk_tree

            tree = generate_risk_tree(main_theme="data center growth", max_depth=1, max_children=2)
            terminals = tree.get_terminal_label_summaries()
            assert terminals
            print(f"PASS  generate_risk_tree ({len(terminals)} terminal nodes)")
        except Exception as exc:
            failures.append(f"mindmap: {exc}")
            print(f"FAIL  generate_risk_tree: {exc}")
    else:
        print("SKIP  LLM mindmap (OPENAI_API_KEY not set)")

    if failures:
        print(f"\n=== FAIL ({len(failures)} check(s)) ===")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
