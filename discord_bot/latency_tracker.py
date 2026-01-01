"""
Latency Tracker - Measures execution time from button click to order fill

Tracks timing at each step of the goal handling pipeline to identify
bottlenecks and measure optimization improvements.

Usage:
    tracker = LatencyTracker()

    # Start tracking a goal event
    metrics = tracker.start("goal_123", "channel_456")

    # Record checkpoints
    metrics.mark_validation_done()
    metrics.mark_decisions_ready()
    metrics.mark_orders_built()
    metrics.mark_order_submitted("HOME WIN", sign_ms=50, post_ms=200, filled=True)
    metrics.mark_orders_submitted()
    metrics.mark_discord_confirmed()

    # Get summary
    print(metrics.summary())
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("latency_tracker")


def now_ms() -> float:
    """Get current time in milliseconds."""
    return time.time() * 1000


@dataclass
class OrderLatency:
    """Timing for a single order."""
    market: str
    sign_ms: float = 0
    post_ms: float = 0
    total_ms: float = 0
    filled: bool = False
    error: Optional[str] = None


@dataclass
class LatencyMetrics:
    """Tracks timing for a single goal event."""
    goal_id: str
    channel_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Timestamps (ms since epoch)
    t0_button_click: float = 0
    t1_validation_done: float = 0
    t2_decisions_ready: float = 0
    t3_orders_built: float = 0
    t4_orders_submitted: float = 0
    t5_discord_confirmed: float = 0

    # Per-order timing
    order_latencies: List[OrderLatency] = field(default_factory=list)

    # Pre-signing info
    used_presigned: bool = False
    presign_age_ms: float = 0  # How old was the pre-signed order

    # Context
    team_scored: str = ""
    score_after: str = ""
    orders_placed: int = 0
    orders_filled: int = 0

    def mark_validation_done(self) -> None:
        self.t1_validation_done = now_ms()

    def mark_decisions_ready(self) -> None:
        self.t2_decisions_ready = now_ms()

    def mark_orders_built(self) -> None:
        self.t3_orders_built = now_ms()

    def mark_orders_submitted(self) -> None:
        self.t4_orders_submitted = now_ms()

    def mark_discord_confirmed(self) -> None:
        self.t5_discord_confirmed = now_ms()

    def add_order_latency(
        self,
        market: str,
        sign_ms: float = 0,
        post_ms: float = 0,
        filled: bool = False,
        error: Optional[str] = None
    ) -> None:
        self.order_latencies.append(OrderLatency(
            market=market,
            sign_ms=sign_ms,
            post_ms=post_ms,
            total_ms=sign_ms + post_ms,
            filled=filled,
            error=error
        ))
        self.orders_placed += 1
        if filled:
            self.orders_filled += 1

    @property
    def validation_ms(self) -> float:
        """Time for validation (t0 -> t1)."""
        if self.t1_validation_done and self.t0_button_click:
            return self.t1_validation_done - self.t0_button_click
        return 0

    @property
    def decisions_ms(self) -> float:
        """Time to get buy decisions (t1 -> t2)."""
        if self.t2_decisions_ready and self.t1_validation_done:
            return self.t2_decisions_ready - self.t1_validation_done
        return 0

    @property
    def build_ms(self) -> float:
        """Time to build orders (t2 -> t3)."""
        if self.t3_orders_built and self.t2_decisions_ready:
            return self.t3_orders_built - self.t2_decisions_ready
        return 0

    @property
    def submit_ms(self) -> float:
        """Time to submit orders (t3 -> t4)."""
        if self.t4_orders_submitted and self.t3_orders_built:
            return self.t4_orders_submitted - self.t3_orders_built
        return 0

    @property
    def discord_ms(self) -> float:
        """Time for Discord confirmation (t4 -> t5)."""
        if self.t5_discord_confirmed and self.t4_orders_submitted:
            return self.t5_discord_confirmed - self.t4_orders_submitted
        return 0

    @property
    def critical_path_ms(self) -> float:
        """Time from click to orders submitted (t0 -> t4)."""
        if self.t4_orders_submitted and self.t0_button_click:
            return self.t4_orders_submitted - self.t0_button_click
        return 0

    @property
    def total_ms(self) -> float:
        """Total time from click to Discord confirmation."""
        if self.t5_discord_confirmed and self.t0_button_click:
            return self.t5_discord_confirmed - self.t0_button_click
        return 0

    @property
    def avg_order_sign_ms(self) -> float:
        """Average signing time per order."""
        if not self.order_latencies:
            return 0
        return sum(o.sign_ms for o in self.order_latencies) / len(self.order_latencies)

    @property
    def avg_order_post_ms(self) -> float:
        """Average POST time per order."""
        if not self.order_latencies:
            return 0
        return sum(o.post_ms for o in self.order_latencies) / len(self.order_latencies)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"**Latency Report** | {self.team_scored} goal | {self.score_after}",
            f"```",
            f"Critical Path: {self.critical_path_ms:.0f}ms (click → orders on Polymarket)",
            f"Total:         {self.total_ms:.0f}ms (click → Discord confirm)",
            f"",
            f"Breakdown:",
            f"  Validation:  {self.validation_ms:6.0f}ms",
            f"  Decisions:   {self.decisions_ms:6.0f}ms",
            f"  Build:       {self.build_ms:6.0f}ms",
            f"  Submit:      {self.submit_ms:6.0f}ms  ← {self.orders_placed} orders",
            f"  Discord:     {self.discord_ms:6.0f}ms",
        ]

        if self.order_latencies:
            lines.append(f"")
            lines.append(f"Per-Order Timing:")
            for ol in self.order_latencies:
                status = "FILLED" if ol.filled else ("ERR" if ol.error else "PENDING")
                presign = " (pre-signed)" if self.used_presigned else ""
                lines.append(f"  {ol.market:12}: sign={ol.sign_ms:.0f}ms post={ol.post_ms:.0f}ms [{status}]{presign}")

        if self.used_presigned:
            lines.append(f"")
            lines.append(f"Pre-signed order age: {self.presign_age_ms:.0f}ms")

        lines.append(f"```")
        lines.append(f"Orders: {self.orders_filled}/{self.orders_placed} filled")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "goal_id": self.goal_id,
            "channel_id": self.channel_id,
            "timestamp": self.timestamp,
            "team_scored": self.team_scored,
            "score_after": self.score_after,
            "used_presigned": self.used_presigned,
            "presign_age_ms": self.presign_age_ms,
            "timings": {
                "validation_ms": self.validation_ms,
                "decisions_ms": self.decisions_ms,
                "build_ms": self.build_ms,
                "submit_ms": self.submit_ms,
                "discord_ms": self.discord_ms,
                "critical_path_ms": self.critical_path_ms,
                "total_ms": self.total_ms,
            },
            "orders": {
                "placed": self.orders_placed,
                "filled": self.orders_filled,
                "avg_sign_ms": self.avg_order_sign_ms,
                "avg_post_ms": self.avg_order_post_ms,
                "details": [
                    {
                        "market": o.market,
                        "sign_ms": o.sign_ms,
                        "post_ms": o.post_ms,
                        "filled": o.filled,
                        "error": o.error,
                    }
                    for o in self.order_latencies
                ],
            },
        }


class LatencyTracker:
    """
    Tracks and persists latency metrics for goal events.

    Usage:
        tracker = LatencyTracker(log_dir="logs/latency")
        metrics = tracker.start("goal_123", "channel_456")
        # ... record checkpoints ...
        tracker.finish(metrics)
    """

    def __init__(self, log_dir: str = "logs/latency"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._active_metrics: Dict[str, LatencyMetrics] = {}
        self._history: List[LatencyMetrics] = []

    def start(self, goal_id: str, channel_id: str) -> LatencyMetrics:
        """Start tracking a new goal event."""
        metrics = LatencyMetrics(
            goal_id=goal_id,
            channel_id=channel_id,
            t0_button_click=now_ms(),
        )
        self._active_metrics[goal_id] = metrics
        logger.debug(f"Started tracking goal {goal_id}")
        return metrics

    def finish(self, metrics: LatencyMetrics) -> None:
        """Finish tracking and persist metrics."""
        # Remove from active
        self._active_metrics.pop(metrics.goal_id, None)

        # Add to history
        self._history.append(metrics)

        # Log summary
        logger.info(
            f"Goal {metrics.goal_id}: {metrics.critical_path_ms:.0f}ms critical path, "
            f"{metrics.orders_filled}/{metrics.orders_placed} filled"
        )

        # Persist to file
        self._persist(metrics)

    def _persist(self, metrics: LatencyMetrics) -> None:
        """Write metrics to log file."""
        log_file = self.log_dir / "latency_log.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist latency metrics: {e}")

    def get_stats(self, last_n: int = 10) -> dict:
        """Get aggregate statistics from recent events."""
        recent = self._history[-last_n:] if self._history else []
        if not recent:
            return {"count": 0}

        critical_paths = [m.critical_path_ms for m in recent if m.critical_path_ms > 0]
        totals = [m.total_ms for m in recent if m.total_ms > 0]

        return {
            "count": len(recent),
            "avg_critical_path_ms": sum(critical_paths) / len(critical_paths) if critical_paths else 0,
            "avg_total_ms": sum(totals) / len(totals) if totals else 0,
            "min_critical_path_ms": min(critical_paths) if critical_paths else 0,
            "max_critical_path_ms": max(critical_paths) if critical_paths else 0,
            "total_orders_placed": sum(m.orders_placed for m in recent),
            "total_orders_filled": sum(m.orders_filled for m in recent),
            "presigned_count": sum(1 for m in recent if m.used_presigned),
        }
