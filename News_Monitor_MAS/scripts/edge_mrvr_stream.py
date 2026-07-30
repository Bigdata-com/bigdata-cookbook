#!/usr/bin/env python3
"""Sample: subscribe to real-time Edge MRVR JSON streaming.

Uses the Edge **feed** host (not the analytics query host)::

    GET https://feed-edge.ravenpack.com/1.0/json/{dataset_id}?keep_alive=t

The HTTP 200 response stays open and emits newline-delimited JSON records.
``keep_alive=t`` sends a bare newline after ~30s of silence so clients can detect
a healthy connection (reset if silent for >60s).

Important: streaming applies only the **dataset definition filters**. Entity
universe and ``rp_provider_id=MRVR`` must be baked into the dataset (this script
creates that dataset). Ad-hoc filters used by historical ``pull`` are not sent
on the stream.

Examples::

    uv run python scripts/edge_mrvr_stream.py \\
      --tickers AAPL,MSFT \\
      --duration-seconds 60 \\
      --output-dir runs/edge_stream_smoke

    uv run python scripts/edge_mrvr_stream.py \\
      --universe us_sml.csv --limit-entities 20 \\
      --min-entity-relevance 90 \\
      --duration-seconds 120
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from ravenpackapi import Dataset

# Allow ``uv run python scripts/edge_mrvr_stream.py`` from repo root.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from edge_mrvr_stories import (  # noqa: E402
    DATASET_NAME,
    INTERNAL_FIELDS,
    PROVIDER_MRVR,
    build_filters,
    load_api_key,
    make_api,
    resolve_entity_mapping,
    write_json,
)

logger = logging.getLogger("edge_mrvr_stream")

FEED_BASE = "https://feed-edge.ravenpack.com/1.0"


def create_stream_dataset(
    *,
    entity_ids: list[str],
    min_entity_relevance: float | None,
    dataset_id_path: Path,
) -> str:
    """Create an Edge granular dataset with universe filters baked in."""
    api = make_api()
    filters = build_filters(entity_ids, min_entity_relevance=min_entity_relevance)
    ds = api.create_dataset(
        Dataset(
            name=f"{DATASET_NAME}-stream",
            product="edge",
            product_version="1.0",
            frequency="granular",
            fields=INTERNAL_FIELDS,
            filters=filters,
        )
    )
    dataset_id_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_id_path.write_text(str(ds.id), encoding="utf-8")
    logger.info("Created stream dataset %s (%d entities)", ds.id, len(entity_ids))
    return str(ds.id)


def stream_records(
    *,
    dataset_id: str,
    api_key: str,
    keep_alive: bool,
    duration_seconds: float | None,
    max_records: int | None,
    output_jsonl: Path | None,
) -> dict[str, Any]:
    """Subscribe to the NDJSON feed until duration/max_records/Ctrl-C."""
    query = urlencode({"keep_alive": "t" if keep_alive else "f"})
    url = f"{FEED_BASE}/json/{dataset_id}?{query}"
    req = Request(url, headers={"api_key": api_key}, method="GET")

    started = time.monotonic()
    n_records = 0
    n_keepalives = 0
    sample: dict[str, Any] | None = None
    out_fh = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_fh = output_jsonl.open("w", encoding="utf-8")

    logger.info("Connecting to %s", url.replace(api_key, "***"))
    try:
        with urlopen(req, timeout=90) as resp:  # noqa: S310 — Edge HTTPS feed
            status = getattr(resp, "status", None) or resp.getcode()
            logger.info("Stream open HTTP %s", status)
            while True:
                if duration_seconds is not None and (time.monotonic() - started) >= duration_seconds:
                    logger.info("Duration limit reached (%.0fs)", duration_seconds)
                    break
                if max_records is not None and n_records >= max_records:
                    logger.info("Record limit reached (%d)", max_records)
                    break

                raw = resp.readline()
                if raw == b"":
                    logger.warning("Stream closed by server (empty read)")
                    break

                line = raw.decode("utf-8", errors="replace")
                if line in {"\n", "\r\n", ""}:
                    n_keepalives += 1
                    logger.debug("keep-alive newline")
                    continue

                text = line.strip()
                if not text:
                    n_keepalives += 1
                    continue

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON line: %s", text[:120])
                    continue

                n_records += 1
                if sample is None and isinstance(payload, dict):
                    sample = payload
                if out_fh is not None:
                    out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    out_fh.flush()
                if n_records == 1 or n_records % 25 == 0:
                    name = payload.get("ENTITY_NAME") or payload.get("entity_name")
                    title = payload.get("TITLE") or payload.get("title") or ""
                    logger.info("record=%d entity=%s title=%s", n_records, name, str(title)[:80])
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit(f"Stream HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise SystemExit(f"Stream connection failed: {exc}") from exc
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if out_fh is not None:
            out_fh.close()

    elapsed = time.monotonic() - started
    return {
        "dataset_id": dataset_id,
        "feed_url": f"{FEED_BASE}/json/{dataset_id}",
        "keep_alive": keep_alive,
        "records": n_records,
        "keepalives": n_keepalives,
        "elapsed_seconds": round(elapsed, 2),
        "sample": sample,
        "output_jsonl": str(output_jsonl) if output_jsonl else None,
        "finished_at": datetime.now(UTC).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--universe", type=Path, default=None)
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--entity-ids", type=str, default=None)
    parser.add_argument("--limit-entities", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Reuse an existing Edge dataset id (must already include MRVR + entity filters)",
    )
    parser.add_argument(
        "--min-entity-relevance",
        type=float,
        default=None,
        help="Bake entity_relevance floor into the stream dataset filters",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=60.0,
        help="Stop after N seconds (default 60; use 0 to run until Ctrl-C / max-records)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on JSON records then exit",
    )
    parser.add_argument(
        "--no-keep-alive",
        action="store_true",
        help="Disable keep_alive=t (not recommended)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("runs") / f"edge_stream_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = make_api()
    mapping = resolve_entity_mapping(
        api,
        output_dir=args.output_dir,
        universe=args.universe,
        tickers=args.tickers,
        entity_ids=args.entity_ids,
        limit_entities=args.limit_entities,
        reuse_mapping=True,
    )
    entity_ids = sorted(set(mapping["rp_entity_id"].astype(str).tolist()))
    if not entity_ids:
        raise SystemExit("No entities resolved for stream universe")

    dataset_id_path = args.output_dir / "dataset_id.txt"
    if args.dataset_id:
        dataset_id = args.dataset_id.strip()
        dataset_id_path.write_text(dataset_id, encoding="utf-8")
        logger.info("Reusing dataset %s", dataset_id)
    else:
        dataset_id = create_stream_dataset(
            entity_ids=entity_ids,
            min_entity_relevance=args.min_entity_relevance,
            dataset_id_path=dataset_id_path,
        )

    duration = None if args.duration_seconds and args.duration_seconds <= 0 else args.duration_seconds
    summary = stream_records(
        dataset_id=dataset_id,
        api_key=load_api_key(),
        keep_alive=not args.no_keep_alive,
        duration_seconds=duration,
        max_records=args.max_records,
        output_jsonl=args.output_dir / "stream_records.jsonl",
    )
    summary["mapped_entities"] = len(entity_ids)
    summary["min_entity_relevance"] = args.min_entity_relevance
    write_json(args.output_dir / "run_summary.json", summary)
    logger.info(
        "stream done — records=%d keepalives=%d elapsed=%.1fs → %s",
        summary["records"],
        summary["keepalives"],
        summary["elapsed_seconds"],
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
