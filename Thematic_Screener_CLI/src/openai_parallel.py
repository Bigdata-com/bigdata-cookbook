"""Parallel OpenAI chat completion executor with rate limiting and retries.

This module exposes a small, well-typed API to fan out many OpenAI chat
completion requests concurrently while respecting OpenAI's rate limits
(requests-per-minute and, optionally, tokens-per-minute). It is intended
to replace ad-hoc, sequential `client.chat.completions.parse` loops in
notebooks/scripts when many independent prompts must be processed.

Design choices
--------------
- Async-first: the core executor is built on `asyncio` and `AsyncOpenAI`
  because OpenAI calls are I/O bound and benefit greatly from concurrency.
- Sliding-window rate limiter: more accurate than a simple per-second cap
  because it tracks the most recent N seconds of usage.
- Bounded concurrency: a semaphore caps in-flight requests independently
  of the RPM ceiling, protecting against memory/CPU spikes when many
  long-running requests overlap.
- Retries with jittered exponential backoff on rate limit and transient
  API errors. Non-retriable client errors fail fast.
- Sync wrapper (`run_chat_requests_parallel`) for notebook ergonomics.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import random
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

Message: TypeAlias = dict[str, str]


def sampling_params_for_model(model: str, **params: Any) -> dict[str, Any]:
    """Return sampling kwargs that ``model`` accepts.

    ``gpt-5.6-luna`` only supports default sampling, so temperature, top_p,
    seed, and penalty fields are omitted for luna models.
    """
    if "luna" in model.lower():
        return {}
    return {key: value for key, value in params.items() if value is not None}


ResponseFormat: TypeAlias = dict[str, Any]


def _make_progress_bar(total: int, desc: str | None) -> Any:
    """Return a ``tqdm`` progress bar suited to the current environment.

    Uses ``tqdm.auto`` so the notebook widget is selected inside Jupyter and
    the terminal bar is used otherwise. Raises :class:`OpenAIParallelError`
    with installation instructions if ``tqdm`` is not available.
    """
    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without tqdm
        raise OpenAIParallelError(
            "show_progress=True requires the `tqdm` package. "
            "Install it with `pip install tqdm`."
        ) from exc
    return tqdm(total=total, desc=desc or "OpenAI requests", unit="req")


class OpenAIParallelError(Exception):
    """Base exception raised by the parallel OpenAI executor."""


@dataclass(slots=True, frozen=True)
class RateLimitConfig:
    """Rate-limiting configuration for the parallel executor.

    Attributes
    ----------
    requests_per_minute:
        Maximum number of API requests allowed in any rolling 60-second
        window. Mirrors OpenAI's RPM tier limit.
    tokens_per_minute:
        Optional cap on total tokens (input + estimated output) in any
        rolling 60-second window. When ``None`` no token limiter is used.
    max_concurrent_requests:
        Maximum number of requests that may be in-flight simultaneously.
        Independent of the RPM cap; protects against resource spikes.
    """

    requests_per_minute: int = 500
    tokens_per_minute: int | None = None
    max_concurrent_requests: int = 20

    def __post_init__(self) -> None:
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if self.tokens_per_minute is not None and self.tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive when set")
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")


@dataclass(slots=True)
class ChatRequest:
    """A single chat completion request to dispatch.

    The ``request_id`` is echoed back on the response so callers can
    correlate results with their inputs (e.g. a sentence_id).
    """

    request_id: str
    messages: list[Message]
    model: str = "gpt-5.6-luna"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 42
    response_format: ResponseFormat | None = None
    max_completion_tokens: int | None = None
    estimated_tokens: int = 0


@dataclass(slots=True)
class ChatResponse:
    """The outcome of a single chat completion request."""

    request_id: str
    content: str | None
    raw_response: Any = None
    error: Exception | None = None
    attempts: int = 1
    elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.content is not None


class SlidingWindowRateLimiter:
    """Asyncio sliding-window rate limiter.

    Allows at most ``max_events`` events in any ``window_seconds`` rolling
    window. Each call to :meth:`acquire` records a usage of ``weight``
    units (default 1) and blocks until enough capacity is free.

    Notes
    -----
    This implementation prefers correctness over throughput: callers are
    serialized by an internal lock when checking the window. Throughput
    is still high in practice because the critical section is O(window)
    deque maintenance only.
    """

    def __init__(self, max_events: int, window_seconds: float = 60.0) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._max_events = max_events
        self._window = window_seconds
        self._timestamps: deque[tuple[float, int]] = deque()
        self._current_load = 0
        self._lock = asyncio.Lock()

    async def acquire(self, weight: int = 1) -> None:
        if weight <= 0:
            raise ValueError("weight must be positive")
        if weight > self._max_events:
            raise ValueError(
                f"Single event weight {weight} exceeds limiter capacity {self._max_events}"
            )
        while True:
            wait_for: float
            async with self._lock:
                now = time.monotonic()
                self._evict_expired(now)
                if self._current_load + weight <= self._max_events:
                    self._timestamps.append((now, weight))
                    self._current_load += weight
                    return
                oldest_ts, _ = self._timestamps[0]
                wait_for = self._window - (now - oldest_ts)
            await asyncio.sleep(max(wait_for, 0.01))

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0][0] <= cutoff:
            _, weight = self._timestamps.popleft()
            self._current_load -= weight


_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


class ParallelOpenAIClient:
    """Execute many OpenAI chat completion requests concurrently.

    Combines a sliding-window rate limiter (RPM, and optionally TPM),
    a concurrency-limiting semaphore, and retries with jittered
    exponential backoff on transient errors.
    """

    def __init__(
        self,
        rate_limit: RateLimitConfig | None = None,
        client: AsyncOpenAI | None = None,
        max_retries: int = 5,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if base_backoff_seconds <= 0:
            raise ValueError("base_backoff_seconds must be positive")
        if max_backoff_seconds < base_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= base_backoff_seconds")

        self._config = rate_limit or RateLimitConfig()
        self._client = client or AsyncOpenAI()
        self._owns_client = client is None
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_requests)
        self._rpm_limiter = SlidingWindowRateLimiter(
            max_events=self._config.requests_per_minute,
            window_seconds=60.0,
        )
        self._tpm_limiter: SlidingWindowRateLimiter | None = None
        if self._config.tokens_per_minute is not None:
            self._tpm_limiter = SlidingWindowRateLimiter(
                max_events=self._config.tokens_per_minute,
                window_seconds=60.0,
            )
        self._max_retries = max_retries
        self._base_backoff = base_backoff_seconds
        self._max_backoff = max_backoff_seconds

    async def aclose(self) -> None:
        """Close the underlying AsyncOpenAI client if we own it."""
        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> ParallelOpenAIClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def submit(self, request: ChatRequest) -> ChatResponse:
        """Execute a single request, respecting limits and retrying transient errors."""
        return await self._execute(request)

    async def run_batch(
        self,
        requests: Sequence[ChatRequest],
        return_exceptions: bool = True,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> list[ChatResponse]:
        """Run all ``requests`` concurrently and return responses in input order.

        When ``return_exceptions`` is True (the default), failures are
        captured on the :class:`ChatResponse` and never raised, so the
        caller always gets a result for every input.

        Parameters
        ----------
        show_progress:
            When True, display a ``tqdm`` progress bar that advances each
            time a request finishes (success or failure). Uses ``tqdm.auto``
            so it renders as a notebook widget in Jupyter and as a terminal
            bar elsewhere.
        progress_desc:
            Optional description shown next to the progress bar.
        """
        if not requests:
            return []

        total = len(requests)
        progress = _make_progress_bar(total=total, desc=progress_desc) if show_progress else None

        async def execute_with_progress(req: ChatRequest) -> ChatResponse:
            try:
                return await self._execute(req)
            finally:
                if progress is not None:
                    progress.update(1)

        tasks = [asyncio.create_task(execute_with_progress(req)) for req in requests]
        try:
            if return_exceptions:
                return await asyncio.gather(*tasks)
            results: list[ChatResponse] = []
            for task in tasks:
                response = await task
                if response.error is not None:
                    raise OpenAIParallelError(
                        f"Request {response.request_id!r} failed after "
                        f"{response.attempts} attempts: {response.error}"
                    ) from response.error
                results.append(response)
            return results
        finally:
            if progress is not None:
                progress.close()

    async def _execute(self, request: ChatRequest) -> ChatResponse:
        start = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._dispatch(request)
            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                logger.warning(
                    "Retriable error on attempt %s/%s for request %s: %s",
                    attempt,
                    self._max_retries,
                    request.request_id,
                    exc,
                )
                if attempt == self._max_retries:
                    break
                await asyncio.sleep(self._compute_backoff(attempt))
                continue
            except APIError as exc:
                logger.error(
                    "Non-retriable API error for request %s: %s",
                    request.request_id,
                    exc,
                )
                return self._failure(request, exc, attempt, start)
            except Exception as exc:  # noqa: BLE001 - last-resort capture
                logger.exception(
                    "Unexpected error for request %s", request.request_id
                )
                return self._failure(request, exc, attempt, start)
            else:
                response.attempts = attempt
                response.elapsed_seconds = time.monotonic() - start
                return response

        return self._failure(
            request,
            last_error or OpenAIParallelError("Exhausted retries with no error captured"),
            self._max_retries,
            start,
        )

    async def _dispatch(self, request: ChatRequest) -> ChatResponse:
        async with self._semaphore:
            await self._rpm_limiter.acquire()
            if self._tpm_limiter is not None and request.estimated_tokens > 0:
                await self._tpm_limiter.acquire(weight=request.estimated_tokens)

            create_kwargs: dict[str, Any] = {
                "model": request.model,
                "messages": request.messages,
                **sampling_params_for_model(
                    request.model,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    seed=request.seed,
                ),
            }
            if request.response_format is not None:
                create_kwargs["response_format"] = request.response_format
            if request.max_completion_tokens is not None:
                create_kwargs["max_completion_tokens"] = request.max_completion_tokens
            completion = await self._client.chat.completions.create(**create_kwargs)

        content = completion.choices[0].message.content if completion.choices else None
        return ChatResponse(
            request_id=request.request_id,
            content=content,
            raw_response=completion,
        )

    def _compute_backoff(self, attempt: int) -> float:
        """Jittered exponential backoff: base * 2^(attempt-1), capped, +/-20% jitter."""
        nominal = min(self._base_backoff * (2 ** (attempt - 1)), self._max_backoff)
        jitter = nominal * 0.2 * (2 * random.random() - 1)
        return max(nominal + jitter, 0.0)

    @staticmethod
    def _failure(
        request: ChatRequest,
        error: Exception,
        attempts: int,
        start: float,
    ) -> ChatResponse:
        return ChatResponse(
            request_id=request.request_id,
            content=None,
            raw_response=None,
            error=error,
            attempts=attempts,
            elapsed_seconds=time.monotonic() - start,
        )


def run_chat_requests_parallel(
    requests: Sequence[ChatRequest],
    rate_limit: RateLimitConfig | None = None,
    max_retries: int = 5,
    base_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 30.0,
    allow_nested_loop: bool = True,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> list[ChatResponse]:
    """Synchronous wrapper over :class:`ParallelOpenAIClient` for notebook use.

    Creates and tears down an :class:`AsyncOpenAI` client internally. For
    long-lived applications prefer instantiating :class:`ParallelOpenAIClient`
    directly and reusing it across batches.

    Parameters
    ----------
    allow_nested_loop:
        When ``True`` (the default) and a running event loop is detected
        (e.g. inside MCP hosts or Jupyter), the batch runs in a worker thread
        with its own event loop. Set to ``False`` to raise instead.
    show_progress:
        When True (the default), display a ``tqdm`` progress bar that
        advances each time a request completes. Renders as a notebook
        widget in Jupyter and as a terminal bar elsewhere. Set to False
        to disable.
    progress_desc:
        Optional description shown next to the progress bar.
    """

    async def runner() -> list[ChatResponse]:
        async with ParallelOpenAIClient(
            rate_limit=rate_limit,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        ) as client:
            return await client.run_batch(
                requests,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )

    def run_in_new_loop() -> list[ChatResponse]:
        return asyncio.run(runner())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run_in_new_loop()

    if not allow_nested_loop:
        raise OpenAIParallelError(
            "run_chat_requests_parallel was called from a running event loop "
            "with allow_nested_loop=False; await ParallelOpenAIClient.run_batch "
            "directly instead."
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(run_in_new_loop).result()
