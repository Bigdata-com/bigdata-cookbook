"""Orchestration for the client news monitor PoC."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.client_monitor.config import resolve_search_category
from src.client_monitor.digest import (
    build_alerts_with_stories,
    build_retrieval_digest,
    build_run_summary,
    write_alerts_with_stories_csv,
    write_json,
    write_jsonl,
)
from src.client_monitor.mas import BaselineStore, ensure_baselines, score_topic
from src.client_monitor.novelty import SeenHeadlineStore, mark_syndication
from src.client_monitor.plans import build_search_plan, save_search_plan
from src.client_monitor.query import QuerySpec, build_entity_wide_spec, build_query_spec
from src.client_monitor.retrieval import (
    dedupe_rows_by_document,
    flatten_documents,
    plan_entity_ids,
    plan_entity_only_search,
    run_plan_search,
)
from src.client_monitor.taxonomy import build_taxonomy_index, build_topic_filter, load_taxonomy
from src.client_monitor.topics import ENTITY_WIDE_MONITOR_TOPIC, MONITOR_TOPICS, SearchMode
from src.client_monitor.universe import UNIVERSE_ID_COLUMN, UNIVERSE_NAME_COLUMN, load_universe
from src.client_monitor.volumes import fetch_entity_volumes
from src.client_monitor.window import TimeWindow, build_time_window, parse_window_end

logger = logging.getLogger(__name__)

DEFAULT_REQUESTS_PER_MINUTE = 350


@dataclass(frozen=True)
class MonitorConfig:
    """Runtime configuration for one client monitor run."""

    universe_path: Path
    taxonomy_path: Path
    output_dir: Path
    window: TimeWindow
    search_modes: tuple[SearchMode, ...]
    category_profile: str
    search_category: dict[str, Any]
    chunk_percentage: float
    limit_entities: int
    max_chunks_per_basket: int | None
    seen_headlines_db: Path | None
    force_baseline_refresh: bool
    requests_per_minute: int
    skip_mas: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe_path": str(self.universe_path),
            "taxonomy_path": str(self.taxonomy_path),
            "output_dir": str(self.output_dir),
            "window_start": self.window.start_iso,
            "window_end": self.window.end_iso,
            "window_minutes": self.window.minutes,
            "search_modes": [mode.value for mode in self.search_modes],
            "category_profile": self.category_profile,
            "search_category": self.search_category,
            "chunk_percentage": self.chunk_percentage,
            "limit_entities": self.limit_entities,
            "max_chunks_per_basket": self.max_chunks_per_basket,
            "seen_headlines_db": str(self.seen_headlines_db) if self.seen_headlines_db else None,
            "force_baseline_refresh": self.force_baseline_refresh,
            "requests_per_minute": self.requests_per_minute,
            "skip_mas": self.skip_mas,
        }


def _load_entity_universe(config: MonitorConfig) -> tuple[list[str], dict[str, str]]:
    universe_df = load_universe(config.universe_path)
    if config.limit_entities > 0:
        universe_df = universe_df.head(config.limit_entities)
    entity_ids = universe_df[UNIVERSE_ID_COLUMN].astype(str).tolist()
    id_to_name = dict(
        zip(
            universe_df[UNIVERSE_ID_COLUMN].astype(str),
            universe_df[UNIVERSE_NAME_COLUMN].astype(str),
            strict=True,
        )
    )
    return entity_ids, id_to_name


def _build_specs(
    *,
    search_mode: SearchMode,
    taxonomy_index: dict[str, list[str]],
    window: TimeWindow,
    category: dict[str, Any],
) -> dict[str, QuerySpec]:
    if search_mode.is_entity_wide():
        return {
            ENTITY_WIDE_MONITOR_TOPIC: build_entity_wide_spec(window=window, category=category),
        }
    specs: dict[str, QuerySpec] = {}
    for topic in MONITOR_TOPICS:
        topic_filter = build_topic_filter(taxonomy_index[topic.key])
        specs[topic.key] = build_query_spec(
            monitor_topic=topic.key,
            search_mode=search_mode,
            document_voice_text=topic.document_voice_text,
            topic_filter=topic_filter,
            window=window,
            category=category,
        )
    return specs


def run_topic_mode(
    *,
    config: MonitorConfig,
    search_mode: SearchMode,
    entity_ids: list[str],
    id_to_name: dict[str, str],
    taxonomy_index: dict[str, list[str]],
    baseline_store: BaselineStore,
    seen_store: SeenHeadlineStore | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run volume + retrieval + MAS for one search mode across all monitor topics."""
    specs = _build_specs(
        search_mode=search_mode,
        taxonomy_index=taxonomy_index,
        window=config.window,
        category=config.search_category,
    )

    chunk_rows: list[dict[str, Any]] = []
    mas_rows: list[dict[str, Any]] = []
    topic_stats: dict[str, Any] = {}

    for topic_key, spec in specs.items():
        topic_start = time.perf_counter()

        logger.info("Fetching volumes for %s (%s)", topic_key, search_mode.value)
        v_now = fetch_entity_volumes(spec, entity_ids)

        logger.info("Ensuring baselines for %s (%s)", topic_key, search_mode.value)
        lambda_by_entity = ensure_baselines(
            spec,
            entity_ids,
            baseline_store,
            force_refresh=config.force_baseline_refresh,
        )

        plan = build_search_plan(
            spec,
            entity_ids,
            v_now,
            max_chunks_per_basket=config.max_chunks_per_basket,
        )
        plan_path = config.output_dir / "plans" / f"{topic_key}_{search_mode.value}.json"
        save_search_plan(plan, plan_path)

        plan_entities = plan_entity_ids(plan)
        documents = run_plan_search(
            plan,
            chunk_percentage=config.chunk_percentage,
            requests_per_minute=config.requests_per_minute,
        )
        rows = flatten_documents(
            documents,
            spec=spec,
            plan_entity_ids=plan_entities,
            id_to_name=id_to_name,
        )
        chunk_rows.extend(rows)

        topic_mas = score_topic(spec, entity_ids, v_now, lambda_by_entity, id_to_name)
        mas_rows.extend(topic_mas)

        topic_stats[topic_key] = {
            "entities_with_volume": len(v_now),
            "baskets": len(plan.get("baskets") or []),
            "expected_chunks": plan.get("chunk_upper_bound_estimate", 0),
            "retrieved_rows": len(rows),
            "elapsed_seconds": round(time.perf_counter() - topic_start, 2),
        }

    chunk_rows = mark_syndication(chunk_rows)
    if seen_store is not None:
        chunk_rows = seen_store.mark_cross_run(chunk_rows)

    return chunk_rows, mas_rows, topic_stats


def run_skip_mas_entity_batch(
    *,
    config: MonitorConfig,
    entity_ids: list[str],
    id_to_name: dict[str, str],
    seen_store: SeenHeadlineStore | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Entity-only smart batch over all companies — no themes, no MAS scoring."""
    start = time.perf_counter()
    spec = build_entity_wide_spec(window=config.window, category=config.search_category)
    plan = plan_entity_only_search(
        entity_ids,
        window=config.window,
        category=config.search_category,
        requests_per_minute=config.requests_per_minute,
    )
    expected = int(plan.get("chunk_upper_bound_estimate") or 0)
    target = int(expected * config.chunk_percentage)
    basket_count = len(plan.get("baskets") or [])
    logger.info(
        "skip-mas entity batch: %d entities, %d baskets, chunk_percentage=%.0f%% "
        "(expected=%s → target≈%s)",
        len(entity_ids),
        basket_count,
        config.chunk_percentage * 100.0,
        f"{expected:,}",
        f"{target:,}",
    )
    save_search_plan(plan, config.output_dir / "plans" / "entity_wide_skip_mas.json")

    plan_entities = plan_entity_ids(plan)
    documents = run_plan_search(
        plan,
        chunk_percentage=config.chunk_percentage,
        requests_per_minute=config.requests_per_minute,
    )
    chunk_rows = flatten_documents(
        documents,
        spec=spec,
        plan_entity_ids=plan_entities,
        id_to_name=id_to_name,
    )
    chunk_rows = mark_syndication(chunk_rows)
    if seen_store is not None:
        chunk_rows = seen_store.mark_cross_run(chunk_rows)

    stats = {
        "entities": len(entity_ids),
        "baskets": basket_count,
        "expected_chunks": expected,
        "chunk_percentage": config.chunk_percentage,
        "target_chunks": target,
        "retrieved_rows": len(chunk_rows),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    return chunk_rows, stats


def run_monitor(config: MonitorConfig) -> dict[str, Any]:
    """Execute the full client monitor pipeline."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "config.json", config.to_dict())

    entity_ids, id_to_name = _load_entity_universe(config)
    seen_store = (
        SeenHeadlineStore(config.seen_headlines_db)
        if config.seen_headlines_db is not None
        else None
    )

    all_chunk_rows: list[dict[str, Any]] = []
    all_mas_rows: list[dict[str, Any]] = []
    mode_stats: dict[str, Any] = {}
    timings: dict[str, float] = {}

    if config.skip_mas:
        chunk_rows, batch_stats = run_skip_mas_entity_batch(
            config=config,
            entity_ids=entity_ids,
            id_to_name=id_to_name,
            seen_store=seen_store,
        )
        all_chunk_rows = chunk_rows
        mode_stats["entity_only_skip_mas"] = batch_stats
        timings["skip_mas_entity_batch"] = batch_stats["elapsed_seconds"]
    else:
        taxonomy = load_taxonomy(config.taxonomy_path)
        taxonomy_index = build_taxonomy_index(taxonomy)
        baseline_store = BaselineStore(config.output_dir / "mas_baselines.db")

        for search_mode in config.search_modes:
            mode_start = time.perf_counter()
            chunk_rows, mas_rows, topic_stats = run_topic_mode(
                config=config,
                search_mode=search_mode,
                entity_ids=entity_ids,
                id_to_name=id_to_name,
                taxonomy_index=taxonomy_index,
                baseline_store=baseline_store,
                seen_store=seen_store,
            )
            all_chunk_rows.extend(chunk_rows)
            all_mas_rows.extend(mas_rows)
            mode_stats[search_mode.value] = {
                "topic_stats": topic_stats,
                "chunk_rows": len(chunk_rows),
                "mas_rows": len(mas_rows),
                "elapsed_seconds": round(time.perf_counter() - mode_start, 2),
            }
            timings[f"mode_{search_mode.value}"] = round(time.perf_counter() - mode_start, 2)

    all_chunk_rows = dedupe_rows_by_document(all_chunk_rows)

    write_jsonl(config.output_dir / "retrieval_chunks.jsonl", all_chunk_rows)
    write_json(
        config.output_dir / "retrieval_digest.json",
        build_retrieval_digest(all_chunk_rows),
    )

    mas_df = pd.DataFrame(all_mas_rows)
    mas_path = config.output_dir / "mas_scores.csv"
    mas_df.to_csv(mas_path, index=False)

    alerts_with_stories = (
        []
        if config.skip_mas
        else build_alerts_with_stories(
            chunk_rows=all_chunk_rows,
            mas_rows=all_mas_rows,
        )
    )
    write_alerts_with_stories_csv(
        config.output_dir / "alerts_with_stories.csv",
        alerts_with_stories,
    )

    summary = build_run_summary(
        config=config.to_dict(),
        chunk_rows=all_chunk_rows,
        mas_rows=all_mas_rows,
        alerts_with_stories=alerts_with_stories,
        timings=timings,
        mode_stats=mode_stats if (config.skip_mas or len(config.search_modes) > 1) else None,
        skip_mas=config.skip_mas,
    )
    write_json(config.output_dir / "run_summary.json", summary)
    return summary


def build_config(
    *,
    universe_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    window_end: str | None,
    window_minutes: int,
    search_mode: str,
    compare_modes: bool,
    category_profile: str,
    chunk_percentage: float,
    limit_entities: int,
    max_chunks_per_basket: int | None,
    seen_headlines_db: Path | None,
    force_baseline_refresh: bool,
    requests_per_minute: int,
    skip_mas: bool = False,
) -> MonitorConfig:
    """Build ``MonitorConfig`` from CLI arguments."""
    if skip_mas and compare_modes:
        msg = "--skip-mas cannot be combined with --compare-modes"
        raise ValueError(msg)
    end = parse_window_end(window_end)
    window = build_time_window(window_end=end, window_minutes=window_minutes)
    if skip_mas:
        modes = (SearchMode.ENTITY_ONLY,)
    elif compare_modes:
        modes = SearchMode.all_modes()
    else:
        modes = (SearchMode.parse(search_mode),)
    resolved_profile, search_category = resolve_search_category(category_profile)
    return MonitorConfig(
        universe_path=universe_path,
        taxonomy_path=taxonomy_path,
        output_dir=output_dir,
        window=window,
        search_modes=modes,
        category_profile=resolved_profile,
        search_category=search_category,
        chunk_percentage=chunk_percentage,
        limit_entities=limit_entities,
        max_chunks_per_basket=max_chunks_per_basket,
        seen_headlines_db=seen_headlines_db,
        force_baseline_refresh=force_baseline_refresh,
        requests_per_minute=requests_per_minute,
        skip_mas=skip_mas,
    )


def default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"client_poc_{stamp}"


def load_environment() -> None:
    """Load .env and validate BIGDATA_API_KEY."""
    load_dotenv(Path.cwd() / ".env", override=False)
    if not os.getenv("BIGDATA_API_KEY"):
        msg = "BIGDATA_API_KEY is not set; add it to your .env or environment"
        raise SystemExit(msg)
