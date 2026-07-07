"""Tests for MCP universe input coercion and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.mcp_workflow import (
    BUILTIN_UNIVERSE_MODES,
    _coerce_universe_input,
    _normalize_universe_dataframe,
    _selected_universe,
    create_run,
    validate_universe,
)
from src.screener import UNIVERSE_ID_COLUMN, load_universe


def test_coerce_bare_csv_path_string() -> None:
    coerced = _coerce_universe_input("mag7.csv")
    assert coerced is not None
    assert coerced["mode"] == "csv_path"
    assert coerced["path"].endswith("mag7.csv")
    assert Path(coerced["path"]).is_file()


def test_coerce_dict_with_path_but_no_mode() -> None:
    coerced = _coerce_universe_input({"path": "mag7.csv"})
    assert coerced is not None
    assert coerced["mode"] == "csv_path"
    _, manifest = _normalize_universe_dataframe(coerced)
    assert manifest["row_count"] == 7
    assert manifest["mode"] == "csv_path"


def test_coerce_json_string_universe() -> None:
    payload = '{"path": "mag7.csv", "id_column": "RP_ENTITY_ID"}'
    coerced = _coerce_universe_input(payload)
    assert coerced is not None
    assert coerced["mode"] == "csv_path"
    _, manifest = _normalize_universe_dataframe(coerced)
    assert manifest["row_count"] == 7


def test_coerce_inline_entity_ids_without_mode() -> None:
    coerced = _coerce_universe_input({"entity_ids": ["E09E2B", "D8442A"]})
    assert coerced is not None
    assert coerced["mode"] == "inline_entity_ids"
    _, manifest = _normalize_universe_dataframe(coerced)
    assert manifest["row_count"] == 2


def test_coerce_builtin_universe_mode_tokens() -> None:
    for mode in BUILTIN_UNIVERSE_MODES:
        coerced = _coerce_universe_input(mode)
        assert coerced == {"mode": mode}


def test_coerce_legacy_rp_company_id_column(tmp_path: Path) -> None:
    import pandas as pd

    legacy_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        {
            "RP_COMPANY_ID": ["E09E2B"],
            "COMPANY_NAME": ["NVIDIA Corp."],
        }
    ).to_csv(legacy_path, index=False)
    universe_df = load_universe(legacy_path)
    assert list(universe_df.columns) == ["RP_ENTITY_ID", "COMPANY_NAME"]
    assert universe_df.iloc[0][UNIVERSE_ID_COLUMN] == "E09E2B"


def test_load_universe_europe_ml_caps_csv() -> None:
    universe_df = load_universe("europe_ml_caps.csv")
    assert UNIVERSE_ID_COLUMN in universe_df.columns
    assert len(universe_df) > 0


def test_selected_universe_prefers_explicit_over_config() -> None:
    config = {"universe_input": {"mode": "default_global_all_caps"}}
    selected = _selected_universe({"path": "mag7.csv"}, config)
    _, manifest = _normalize_universe_dataframe(selected)
    assert manifest["row_count"] == 7


def test_validate_universe_builtin_sample_xnas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setattr("src.mcp_workflow.DEFAULT_RUNS_ROOT", runs_root)

    create_run(
        main_theme="Test theme",
        analyst_focus="Smoke test",
        run_id="sample_xnas_mode_test",
        universe={"mode": "default_global_all_caps"},
    )
    response = validate_universe("sample_xnas_mode_test", universe="sample_xnas")
    assert response["universe_summary"]["mode"] == "sample_xnas"
    assert response["universe_summary"]["row_count"] > 100


def test_create_run_uses_mag7_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setattr("src.mcp_workflow.DEFAULT_RUNS_ROOT", runs_root)

    response = create_run(
        main_theme="Test risk",
        analyst_focus="Quick smoke test",
        run_id="mag7_mcp_universe_test",
        universe={"path": "mag7.csv"},
    )
    assert response["universe_summary"]["row_count"] == 7
    assert response["universe_summary"]["mode"] == "csv_path"


def test_validate_universe_string_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setattr("src.mcp_workflow.DEFAULT_RUNS_ROOT", runs_root)

    create_run(
        main_theme="Test risk",
        analyst_focus="Quick smoke test",
        run_id="mag7_validate_string",
        universe={"mode": "default_global_all_caps"},
    )
    response = validate_universe("mag7_validate_string", universe="mag7.csv")
    assert response["universe_summary"]["row_count"] == 7
    assert response["universe_summary"]["mode"] == "csv_path"
