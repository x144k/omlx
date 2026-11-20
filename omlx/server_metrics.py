# SPDX-License-Identifier: Apache-2.0
"""
Server-level metrics for the oMLX admin dashboard.

Provides a thread-safe singleton that aggregates serving metrics
across all engines/models. Session metrics reset on server start,
while all-time metrics persist across restarts via JSON file.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .usage_history import UsageHistory

logger = logging.getLogger(__name__)

# Interval between periodic saves of all-time stats (seconds)
_SAVE_INTERVAL = 300


class ServerMetrics:
    """
    Global server-level metrics for the Status dashboard.

    Thread-safe: uses threading.Lock since scheduler runs in ThreadPoolExecutor.
    Tracks cumulative totals and average speeds across all requests,
    with optional per-model breakdown.

    Supports two scopes:
    - "session": resets on server restart (default, backward-compatible)
    - "alltime": persisted across restarts via stats_path JSON file
    """

    def __init__(self, stats_path: Optional[Path] = None):
        self._lock = threading.Lock()
        self._stats_path = stats_path
        self.usage_history: UsageHistory | None = None
        self._history_record_failed = False

        # Session totals (reset on server restart or clear)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_requests: int = 0
        self.total_prefill_duration: float = 0.0
        self.total_generation_duration: float = 0.0
        # Current-window counters -- initialised here so get_snapshot() never
        # hits AttributeError on a fresh server before the first request lands
        # (a startup health/stats probe can call get_snapshot() first). The old
        # code set these only in record_request_complete()/clear_metrics().
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0
        self._live_prompt_tokens: int = 0
        self._live_completion_tokens: int = 0
        self._per_model: Dict[str, Dict[str, Any]] = {}

        # Preflight rejections — split by reason so operators can tell
        # "user fed an oversize prompt" (hard_limit) apart from "system
        # under memory pressure, admission paused" (admission_paused).
        # This is the only operator-visible signal for the prefill-peak
        # memory guard; the guard was previously dead and shipping it
        # without telemetry repeats that mistake.
        self.preflight_rejections: Dict[str, int] = {
            "hard_limit": 0,
            "admission_paused": 0,
        }

        # All-time totals (persisted across restarts)
        self._alltime_prompt_tokens: int = 0
        self._alltime_completion_tokens: int = 0
        self._alltime_cached_tokens: int = 0
        self._alltime_requests: int = 0
        self._alltime_prefill_duration: float = 0.0
        self._alltime_generation_duration: float = 0.0
        self._alltime_per_model: Dict[str, Dict[str, Any]] = {}

        # Live current-window prompt tokens: the FULL conversation length (all
        # messages incl. tool results) as of the latest prompt build, tracked
        # separately from _last_prompt_tokens (the last *completed generation's*
        # prompt). The gap between them is this turn's tool-output tokens, which
        # land after the last generation but before the next prompt is built --
        # exactly what a per-generation gauge otherwise misses. current_total_tokens
        # uses the max so the live context gauge matches the GUI (which counts the
        # full message list), not the frozen last-generation count. See #4671.
        self._live_prompt_tokens: int = 0

        self._start_time = time.time()
        self._last_save_time = time.time()

        # Load persisted all-time stats
        if stats_path:
            self._load_alltime()

    @staticmethod
    def _new_model_counters() -> Dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "requests": 0,
            "prefill_duration": 0.0,
            "generation_duration": 0.0,
        }

    def _load_alltime(self) -> None:
        """Load all-time stats from disk. Called once during __init__."""
        if not self._stats_path or not self._stats_path.exists():
            return
        try:
            with open(self._stats_path) as f:
                data = json.load(f)
            self._alltime_prompt_tokens = int(data.get("total_prompt_tokens", 0))
            self._alltime_completion_tokens = int(
                data.get("total_completion_tokens", 0)
            )
            self._alltime_cached_tokens = int(data.get("total_cached_tokens", 0))
            self._alltime_requests = int(data.get("total_requests", 0))
            self._alltime_prefill_duration = float(
                data.get("total_prefill_duration", 0.0)
            )
            self._alltime_generation_duration = float(
                data.get("total_generation_duration", 0.0)
            )
            per_model = data.get("per_model", {})
            for model_id, counters in per_model.items():
                self._alltime_per_model[model_id] = {
                    "prompt_tokens": int(counters.get("prompt_tokens", 0)),
                    "completion_tokens": int(counters.get("completion_tokens", 0)),
                    "cached_tokens": int(counters.get("cached_tokens", 0)),
                    "requests": int(counters.get("requests", 0)),
                    "prefill_duration": float(counters.get("prefill_duration", 0.0)),
                    "generation_duration": float(
                        counters.get("generation_duration", 0.0)
                    ),
                }
            logger.info("Loaded all-time stats from %s", self._stats_path)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError, OSError) as e:
            logger.warning("Failed to load all-time stats from %s: %s", self._stats_path, e)

    def save_alltime(self) -> None:
        """Save all-time stats to disk. Thread-safe."""
        if not self._stats_path:
            return
        with self._lock:
            data = {
                "total_prompt_tokens": self._alltime_prompt_tokens,
                "total_completion_tokens": self._alltime_completion_tokens,
                "total_cached_tokens": self._alltime_cached_tokens,
                "total_requests": self._alltime_requests,
                "total_prefill_duration": self._alltime_prefill_duration,
                "total_generation_duration": self._alltime_generation_duration,
                "per_model": dict(self._alltime_per_model),
            }
            self._last_save_time = time.time()
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._stats_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self._stats_path)
        except OSError as e:
            logger.warning("Failed to save all-time stats to %s: %s", self._stats_path, e)

    def _maybe_save_alltime(self) -> None:
        """Save all-time stats if enough time has passed. Called within lock."""
        if not self._stats_path:
            return
        now = time.time()
        if now - self._last_save_time >= _SAVE_INTERVAL:
            # Release lock before I/O
            self._lock.release()
            try:
                self.save_alltime()
            finally:
                self._lock.acquire()

    def record_request_complete(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        prefill_duration: float = 0.0,
        generation_duration: float = 0.0,
        model_id: str = "",
        request_duration: float | None = None,
    ) -> None:
        """Record a completed request. Thread-safe."""
        with self._lock:
            # Session counters
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cached_tokens += cached_tokens
            self.total_requests += 1
            # Current-window (last completed request) tokens.
            self._last_prompt_tokens = prompt_tokens
            self._last_completion_tokens = completion_tokens
            self.total_prefill_duration += prefill_duration
            self.total_generation_duration += generation_duration

            # All-time counters
            self._alltime_prompt_tokens += prompt_tokens
            self._alltime_completion_tokens += completion_tokens
            self._alltime_cached_tokens += cached_tokens
            self._alltime_requests += 1
            self._alltime_prefill_duration += prefill_duration
            self._alltime_generation_duration += generation_duration

            # Per-model counters (session)
            if model_id:
                if model_id not in self._per_model:
                    self._per_model[model_id] = self._new_model_counters()
                m = self._per_model[model_id]
                m["prompt_tokens"] += prompt_tokens
                m["completion_tokens"] += completion_tokens
                m["cached_tokens"] += cached_tokens
                m["requests"] += 1
                m["prefill_duration"] += prefill_duration
                m["generation_duration"] += generation_duration

                # Per-model counters (all-time)
                if model_id not in self._alltime_per_model:
                    self._alltime_per_model[model_id] = self._new_model_counters()
                am = self._alltime_per_model[model_id]
                am["prompt_tokens"] += prompt_tokens
                am["completion_tokens"] += completion_tokens
                am["cached_tokens"] += cached_tokens
                am["requests"] += 1
                am["prefill_duration"] += prefill_duration
                am["generation_duration"] += generation_duration

            # Periodic save
            self._maybe_save_alltime()

        if self.usage_history is not None:
            try:
                self.usage_history.record(
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    prefill_duration=prefill_duration,
                    generation_duration=generation_duration,
                    request_duration=request_duration,
                )
            except Exception:
                if not self._history_record_failed:
                    logger.warning("Usage history recording failed; serving continues")
                    self._history_record_failed = True

    def close(self) -> None:
        """Flush cumulative stats and historical aggregates at shutdown."""
        self.save_alltime()
        if self.usage_history is not None:
            self.usage_history.close()

    def record_preflight_rejection(self, reason: str) -> None:
        """Increment the preflight-rejection counter for ``reason``.

        Unknown reasons are bucketed under ``"other"`` rather than
        silently dropped so an operator notices when a new reject path
        is added without updating the metric.
        """
        with self._lock:
            if reason not in self.preflight_rejections:
                reason = "other"
                self.preflight_rejections.setdefault("other", 0)
            self.preflight_rejections[reason] += 1

    def _build_snapshot(
        self,
        prompt: int,
        completion: int,
        cached: int,
        requests: int,
        prefill_dur: float,
        gen_dur: float,
        uptime: float,
    ) -> Dict[str, Any]:
        """Build a metrics snapshot dict from raw values."""
        actual_processed = prompt - cached
        avg_prefill_tps = (
            actual_processed / prefill_dur if prefill_dur > 0 else 0.0
        )
        avg_generation_tps = completion / gen_dur if gen_dur > 0 else 0.0
        cache_efficiency = (cached / prompt * 100) if prompt > 0 else 0.0

        return {
            "total_tokens_served": prompt + completion,
            "total_cached_tokens": cached,
            "cache_efficiency": round(cache_efficiency, 1),
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "total_requests": requests,
            "avg_prefill_tps": round(avg_prefill_tps, 1),
            "avg_generation_tps": round(avg_generation_tps, 1),
            "uptime_seconds": round(uptime, 1),
        }

    def get_snapshot(
        self, model_id: str = "", scope: str = "session"
    ) -> Dict[str, Any]:
        """Get current metrics snapshot. Thread-safe.

        Args:
            model_id: If provided and tracked, return per-model metrics.
                      Otherwise return global aggregate.
            scope: "session" for current session, "alltime" for persisted totals.
        """
        with self._lock:
            now = time.time()
            uptime = now - self._start_time

            if scope == "alltime":
                if model_id:
                    m = self._alltime_per_model.get(model_id)
                    if m:
                        return self._build_snapshot(
                            m["prompt_tokens"],
                            m["completion_tokens"],
                            m["cached_tokens"],
                            m["requests"],
                            m["prefill_duration"],
                            m["generation_duration"],
                            uptime,
                        )
                    return self._build_snapshot(0, 0, 0, 0, 0.0, 0.0, uptime)
                return self._build_snapshot(
                    self._alltime_prompt_tokens,
                    self._alltime_completion_tokens,
                    self._alltime_cached_tokens,
                    self._alltime_requests,
                    self._alltime_prefill_duration,
                    self._alltime_generation_duration,
                    uptime,
                )

            # scope == "session" (default)
            if model_id:
                m = self._per_model.get(model_id)
                if m:
                    return self._build_snapshot(
                        m["prompt_tokens"],
                        m["completion_tokens"],
                        m["cached_tokens"],
                        m["requests"],
                        m["prefill_duration"],
                        m["generation_duration"],
                        uptime,
                    )
                return self._build_snapshot(0, 0, 0, 0, 0.0, 0.0, uptime)

            snap = self._build_snapshot(
                self.total_prompt_tokens,
                self.total_completion_tokens,
                self.total_cached_tokens,
                self.total_requests,
                self.total_prefill_duration,
                self.total_generation_duration,
                uptime,
            )
            snap["current_prompt_tokens"] = self._last_prompt_tokens
            snap["current_completion_tokens"] = self._last_completion_tokens
            snap["current_total_tokens"] = (
                max(self._last_prompt_tokens, self._live_prompt_tokens)
                + self._live_completion_tokens
            )
            return snap

    def clear_metrics(self) -> None:
        """Clear session metrics. Thread-safe."""
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_cached_tokens = 0
            self.total_requests = 0
            # Current-window (last completed request) is the live context gauge
            # the web-admin dashboard shows, distinct from the cumulative
            # session totals below, which sum across every request and exceed
            # the ceiling on long sessions (a prior context-gauge accounting bug).
            self._last_prompt_tokens = 0
            self._last_completion_tokens = 0
            self._live_prompt_tokens = 0
            self._live_completion_tokens = 0
            self.total_prefill_duration = 0.0
            self.total_generation_duration = 0.0
            self._per_model.clear()

    def reset_live_context(self) -> None:
        """Reset only the live current-window context counters (NOT the
        cumulative usage totals). Called on scheduler deep_reset / engine
        switch so the #4671 live context gauge reflects the rebuilt
        conversation instead of a monotonic pre-reset high-water mark.
        Cumulative all-time usage is preserved."""
        with self._lock:
            self._last_prompt_tokens = 0
            self._last_completion_tokens = 0
            self._live_prompt_tokens = 0
            self._live_completion_tokens = 0

    def set_live_prompt(self, tokens: int) -> None:
        """Record the full conversation length at a prompt build.

        num_prompt_tokens is the tokenized length of the *entire* current
        conversation (system prompt + all messages + every tool result appended
        so far) -- the same quantity the web-admin GUI renders as context usage.
        Tracking it here lets the live context gauge (#4671 current_total_tokens)
        match the GUI even for tool-output tokens that arrive after the last
        completed generation, which a per-generation prompt/completion gauge
        would otherwise miss until the next generation's prompt is rebuilt.
        Thread-safe; monotonic within a session.
        """
        with self._lock:
            if tokens > self._live_prompt_tokens:
                self._live_prompt_tokens = tokens

    def set_live_completion(self, tokens: int) -> None:
        """Record the current in-flight generation's completion-token count.

        Paired with set_live_prompt so current_total_tokens reflects the
        *live* turn (prompt + generated-so-far) -- the same quantity the
        web-admin GUI renders -- rather than the last *completed* generation.
        Reading the gauge in the window between a generation finishing and
        record_request_complete() firing, where the frozen _last_* counters
        still hold the previous turn, keeps it from under-reporting by this
        turn's completion tokens. Thread-safe; monotonic within a session."""
        with self._lock:
            if tokens > self._live_completion_tokens:
                self._live_completion_tokens = tokens

    def clear_alltime_metrics(self) -> None:
        """Clear all-time metrics and delete the persisted file. Thread-safe."""
        with self._lock:
            self._alltime_prompt_tokens = 0
            self._alltime_completion_tokens = 0
            self._alltime_cached_tokens = 0
            self._alltime_requests = 0
            self._alltime_prefill_duration = 0.0
            self._alltime_generation_duration = 0.0
            self._alltime_per_model.clear()
        if self._stats_path and self._stats_path.exists():
            try:
                self._stats_path.unlink()
            except OSError as e:
                logger.warning(
                    "Failed to delete stats file %s: %s", self._stats_path, e
                )


# Global singleton
_server_metrics: Optional[ServerMetrics] = None


def get_server_metrics() -> ServerMetrics:
    """Get the global ServerMetrics singleton."""
    global _server_metrics
    if _server_metrics is None:
        _server_metrics = ServerMetrics()
    return _server_metrics


def reset_server_metrics(
    stats_path: Optional[Path] = None, *, usage_history_enabled: bool = True
) -> None:
    """Reset metrics (called on server start).

    If a previous instance exists and has a stats_path, save before resetting.
    ``usage_history_enabled`` seeds the recorder from settings; the toggle can
    still be flipped at runtime through the admin API.
    """
    global _server_metrics
    if _server_metrics is not None:
        _server_metrics.close()
    _server_metrics = ServerMetrics(stats_path=stats_path)
    if stats_path is not None:
        from .usage_history import UsageHistory

        try:
            _server_metrics.usage_history = UsageHistory(
                stats_path.parent / "usage.sqlite3", enabled=usage_history_enabled
            )
        except Exception:
            logger.warning("Usage history initialization failed; serving continues")
