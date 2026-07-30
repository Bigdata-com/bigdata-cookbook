"""Fast offline match of MissedStories against an Edge datafile CSV."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ravenpackapi import RPApi

logger = logging.getLogger("edge_match_offline")


def normalize_title(text: str) -> str:
    cleaned = str(text).lower()
    cleaned = re.sub(r"\s*\|\s*.*$", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.95
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(Path.cwd() / ".env", override=False)
    import os

    out = Path("runs/edge_missed_recovery")
    mapping = pd.read_csv(out / "entity_mapping.csv")
    missed = pd.read_csv("MissedStories.csv", sep=";")
    missed.columns = [c.strip() for c in missed.columns]
    missed = missed.loc[:, [c for c in missed.columns if c]]
    missed["Ticker"] = missed["Ticker"].astype(str).str.strip().str.upper()
    missed["Pub Date"] = pd.to_datetime(missed["Pub Date"], errors="coerce")

    logger.info("Loading datafile…")
    edge = pd.read_csv(
        out / "datafile_raw.csv",
        usecols=lambda c: str(c).lower()
        in {
            "timestamp_utc",
            "entity_name",
            "title",
            "source_name",
            "rp_document_id",
            "rp_entity_id",
        },
    )
    edge.columns = [str(c).strip().lower() for c in edge.columns]
    logger.info("Loaded %d edge rows", len(edge))

    entity_to_ticker = dict(
        zip(mapping["rp_entity_id"].astype(str), mapping["ticker"].astype(str), strict=True)
    )
    edge["ticker"] = edge["rp_entity_id"].astype(str).map(entity_to_ticker)
    edge = edge.dropna(subset=["ticker", "title"]).copy()
    edge["title_norm"] = edge["title"].map(normalize_title)

    # Exact / containment prefilter via normalized title sets per ticker
    by_ticker: dict[str, pd.DataFrame] = {
        str(t): g.reset_index(drop=True) for t, g in edge.groupby("ticker", sort=False)
    }

    threshold = 0.72
    match_rows: list[dict[str, object]] = []
    matched_doc_ids: set[str] = set()

    for i, story in missed.iterrows():
        ticker = str(story["Ticker"]).upper()
        headline = str(story["Headline"])
        hnorm = normalize_title(headline)
        candidates = by_ticker.get(ticker)
        best_score = 0.0
        best: dict[str, object] = {}
        if candidates is not None and not candidates.empty:
            # Prefer exact / containment first
            exact = candidates[candidates["title_norm"] == hnorm]
            if not exact.empty:
                row = exact.iloc[0]
                best_score = 1.0
                best = row.to_dict()
            else:
                # Containment either direction
                contain_mask = candidates["title_norm"].map(
                    lambda t: bool(t) and (hnorm in t or t in hnorm)
                )
                contain = candidates[contain_mask]
                pool = contain if not contain.empty else candidates
                # Cap scoring work for huge tickers
                if len(pool) > 5000:
                    # token overlap prefilter: require >=50% of headline tokens
                    tokens = set(hnorm.split())
                    if tokens:
                        min_hit = max(1, len(tokens) // 2)

                        def _keep(t: str) -> bool:
                            return len(tokens & set(str(t).split())) >= min_hit

                        mask = pool["title_norm"].map(_keep)
                        pool = pool[mask]
                    pool = pool.head(5000)
                for _, cand in pool.iterrows():
                    score = title_similarity(headline, str(cand["title"]))
                    if score > best_score:
                        best_score = score
                        best = cand.to_dict()
        matched = best_score >= threshold
        doc_id = str(best.get("rp_document_id") or "") if matched else ""
        if matched and doc_id:
            matched_doc_ids.add(doc_id)
        match_rows.append(
            {
                "ticker": ticker,
                "missed_headline": headline,
                "publisher": story.get("Publisher"),
                "pub_date": story.get("Pub Date"),
                "matched": matched,
                "match_score": round(best_score, 3),
                "edge_title": best.get("title", "") if matched else "",
                "edge_source": best.get("source_name", "") if matched else "",
                "edge_timestamp_utc": best.get("timestamp_utc", "") if matched else "",
                "rp_document_id": doc_id,
                "edge_url": "",
            }
        )
        if (i + 1) % 50 == 0:
            logger.info("Matched %d / %d stories…", i + 1, len(missed))

    api_key = os.getenv("RAVENPACK_API_KEY") or os.getenv("RP_API_KEY")
    if api_key and matched_doc_ids:
        api = RPApi(api_key=api_key, product="edge")
        api.log_curl_commands = False
        url_cache: dict[str, str] = {}
        for doc_id in sorted(matched_doc_ids):
            try:
                url_cache[doc_id] = str(api.get_document_url(doc_id) or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("URL fail %s: %s", doc_id, exc)
                url_cache[doc_id] = ""
            time.sleep(0.02)
        for row in match_rows:
            doc_id = str(row.get("rp_document_id") or "")
            if doc_id:
                row["edge_url"] = url_cache.get(doc_id, "")

    matches = pd.DataFrame(match_rows)
    matches.to_csv(out / "missed_match_report.csv", index=False)
    matched_n = int(matches["matched"].sum())
    summary = {
        "mode": "recover_offline",
        "missed_stories": int(len(missed)),
        "matched": matched_n,
        "match_rate": round(matched_n / len(missed), 4) if len(missed) else 0.0,
        "match_threshold": threshold,
        "edge_rows": int(len(edge)),
        "provider": "MRVR",
        "note": "MRVR-only; premium publishers may be out of scope",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Done — matched %d / %d (%.1f%%)", matched_n, len(missed), 100 * summary["match_rate"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
