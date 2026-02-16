"""
Batch API Client for Bigdata.com Search.

Submits queries to the Batch Search API, polls for completion, and downloads results.
Used by the Batch_Search_API cookbook notebook.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

DEFAULT_BASE_URL = "https://api.bigdata.com"
DEFAULT_POLL_INTERVAL = 15
DEFAULT_TIMEOUT = 300


class MetricsTracker:
    """Track and aggregate metrics for batch API operations."""

    def __init__(self, batch_id: str, num_queries: int) -> None:
        self.batch_id = batch_id
        self.num_queries = num_queries
        self.start_time = time.time()
        self.api_timings: Dict[str, Any] = {
            "create_job": None,
            "upload_file": None,
            "status_checks": [],
            "download_results": None,
        }
        self.status_history: list[Dict[str, Any]] = []
        self.status_durations: Dict[str, float] = {
            "pending": 0.0,
            "processing": 0.0,
            "completed": 0.0,
            "failed": 0.0,
        }
        self.current_status: Optional[str] = None
        self.status_start_time: Optional[float] = None
        self.poll_count = 0
        self.input_file_size = 0
        self.output_file_size = 0
        self.upload_size = 0
        self.download_size = 0

    def record_api_call(
        self, operation: str, duration: float, size: Optional[int] = None
    ) -> None:
        if operation == "create_job":
            self.api_timings["create_job"] = duration
        elif operation == "upload_file":
            self.api_timings["upload_file"] = duration
            if size is not None:
                self.upload_size = size
        elif operation == "status_check":
            self.api_timings["status_checks"].append(duration)
            self.poll_count += 1
        elif operation == "download_results":
            self.api_timings["download_results"] = duration
            if size is not None:
                self.download_size = size

    def record_status_change(self, new_status: str) -> None:
        now = time.time()
        if self.current_status and self.status_start_time is not None:
            duration = now - self.status_start_time
            if self.current_status in self.status_durations:
                self.status_durations[self.current_status] += duration
        if new_status != self.current_status:
            self.status_history.append(
                {"status": new_status, "timestamp": now, "elapsed": now - self.start_time}
            )
            self.current_status = new_status
            self.status_start_time = now

    def set_file_sizes(self, input_size: int, output_size: int = 0) -> None:
        self.input_file_size = input_size
        self.output_file_size = output_size

    @staticmethod
    def _format_duration(seconds: float) -> str:
        td = timedelta(seconds=int(seconds))
        parts = []
        if td.days > 0:
            parts.append(f"{td.days}d")
        hours, remainder = divmod(td.seconds, 3600)
        if hours > 0:
            parts.append(f"{hours}h")
        minutes, secs = divmod(remainder, 60)
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        return " ".join(parts)


class BatchAPIClient:
    """Client for Bigdata.com Batch Search API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.organization_id = organization_id
        self.user_id = user_id
        self.client = httpx.Client(timeout=DEFAULT_TIMEOUT)

    def create_batch_job(self) -> Tuple[str, str, float]:
        url = f"{self.base_url}/v1/search/batches"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        body = {
            "organizationId": self.organization_id,
            "userId": self.user_id,
        }
        start = time.time()
        response = self.client.post(url, headers=headers, json=body)
        response.raise_for_status()
        duration = time.time() - start
        data = response.json()
        if isinstance(data.get("body"), str):
            try:
                data = json.loads(data["body"])
            except json.JSONDecodeError:
                pass
        if isinstance(data.get("data"), dict):
            data = data["data"]
        batch_id = data.get("batchId") or data.get("batch_id") or data.get("id")
        presigned_url = (
            data.get("presignedUrl")
            or data.get("presigned_url")
            or data.get("uploadUrl")
            or data.get("upload_url")
        )
        if not batch_id or not presigned_url:
            raise ValueError(
                f"Missing batchId or presignedUrl. Keys: {list(data.keys())}"
            )
        return batch_id, presigned_url, duration

    def upload_queries_file(self, presigned_url: str, jsonl_content: bytes) -> float:
        start = time.time()
        response = httpx.put(
            presigned_url, headers={"Content-Type": "application/jsonl"}, content=jsonl_content
        )
        response.raise_for_status()
        return time.time() - start

    def check_batch_status(self, batch_id: str) -> Tuple[Dict[str, Any], float]:
        url = f"{self.base_url}/v1/search/batches/{batch_id}"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        body = {"organizationId": self.organization_id, "userId": self.user_id}
        start = time.time()
        response = self.client.post(url, headers=headers, json=body)
        if response.status_code == 404:
            response = self.client.get(url, headers={"X-API-KEY": self.api_key})
        response.raise_for_status()
        duration = time.time() - start
        return response.json(), duration

    def download_results(
        self, download_url: str, timeout: int = DEFAULT_TIMEOUT
    ) -> Tuple[bytes, float]:
        start = time.time()
        response = httpx.get(download_url, timeout=timeout)
        response.raise_for_status()
        return response.content, time.time() - start

    def close(self) -> None:
        self.client.close()


def poll_until_complete(
    client: BatchAPIClient,
    batch_id: str,
    metrics: MetricsTracker,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    max_wait: int = 3600,
) -> Dict[str, Any]:
    """Poll batch job until completion or failure."""
    start_time = time.time()
    print(f"Polling for batch job completion (batch_id: {batch_id})...")
    print(f"Poll interval: {poll_interval} seconds\n")
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise TimeoutError(f"Batch job did not complete within {max_wait} seconds")
        try:
            status_data, duration = client.check_batch_status(batch_id)
            metrics.record_api_call("status_check", duration)
            status = status_data.get("status", "unknown")
            metrics.record_status_change(status)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status.upper()} | "
                f"Elapsed: {MetricsTracker._format_duration(elapsed)}"
            )
            if status == "completed":
                print("\n✓ Batch job completed successfully!")
                return status_data
            if status == "failed":
                msg = (
                    status_data.get("errorMessage")
                    or status_data.get("error_message")
                    or "Unknown error"
                )
                raise RuntimeError(f"Batch job failed: {msg}")
            time.sleep(poll_interval)
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            print(f"⚠ Error checking status: {e}. Retrying in {poll_interval}s...")
            time.sleep(poll_interval)
