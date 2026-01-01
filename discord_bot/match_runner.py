"""
Match Runner - Background price fetching and model updates.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import time

from .poisson_model import PoissonPredictor, Probabilities

logger = logging.getLogger("match_runner")


@dataclass
class MatchConfig:
    """Match configuration."""
    name: str
    channel_id: str
    kick_off_cet: datetime
    home_team: str
    away_team: str
    home_win_token: str
    away_win_token: str
    draw_token: str
    budget_per_market: float = 500.0

    # Optional additional markets
    over_2_5_token: str = ""
    under_2_5_token: str = ""


@dataclass
class PreSignedOrders:
    """Container for pre-signed orders."""
    orders: Dict[str, Dict] = field(default_factory=dict)
    created_at: float = 0

    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000 if self.created_at else 0


@dataclass
class LiveMatchState:
    """State of a running match."""
    config: MatchConfig
    match_minute: int = 0
    home_goals: int = 0
    away_goals: int = 0
    predictor: PoissonPredictor = field(default_factory=PoissonPredictor)
    is_running: bool = False

    def get_buy_decisions(self, home_scores: bool) -> dict:
        return self.predictor.get_buy_decisions(home_scores)

    def update_score(self, home_goals: int, away_goals: int):
        self.home_goals = home_goals
        self.away_goals = away_goals


class AsyncMatchRunner:
    """Background runner that fetches prices and updates models."""

    def __init__(self, exchange=None):
        self.exchange = exchange
        self._matches: Dict[str, LiveMatchState] = {}
        self._task: Optional[asyncio.Task] = None
        self._status_callback = None

    def set_status_callback(self, callback) -> None:
        self._status_callback = callback

    def add_match(self, config: MatchConfig) -> None:
        self._matches[config.channel_id] = LiveMatchState(config=config)

    def get_match_state(self, channel_id: str) -> Optional[LiveMatchState]:
        return self._matches.get(channel_id)

    def get_buy_decisions(self, channel_id: str, home_scores: bool) -> Optional[dict]:
        state = self._matches.get(channel_id)
        if state and state.is_running:
            return state.get_buy_decisions(home_scores)
        return None

    def update_score(self, channel_id: str, home_goals: int, away_goals: int) -> None:
        if channel_id in self._matches:
            self._matches[channel_id].update_score(home_goals, away_goals)

    def get_presigned_orders(self, channel_id: str, home_scores: bool) -> Optional[PreSignedOrders]:
        # Pre-signing implementation omitted
        return None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _run_loop(self) -> None:
        """Main loop - fetches prices every minute."""
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Runner error: {e}")
            await asyncio.sleep(60)

    async def _tick(self) -> None:
        """Process all matches."""
        now = datetime.utcnow()
        for state in self._matches.values():
            self._update_match(state, now)

    def _update_match(self, state: LiveMatchState, now: datetime) -> None:
        """Update a single match. Implementation details omitted."""
        # Fetches prices, updates model, handles timing
        pass


def load_match_config_from_json(match_data: dict, teams_order: tuple) -> Optional[MatchConfig]:
    """Load match config from JSON. Implementation details omitted."""
    try:
        home_key, away_key = teams_order
        teams = match_data.get('teams', {})

        kick_off_str = match_data.get('kick_off_cet', '')
        if not kick_off_str:
            return None

        kick_off = datetime.fromisoformat(kick_off_str.replace(' ', 'T'))

        return MatchConfig(
            name=match_data.get('name', ''),
            channel_id=match_data.get('channel_id', ''),
            kick_off_cet=kick_off,
            home_team=home_key,
            away_team=away_key,
            home_win_token=teams.get(home_key, {}).get('token_id', ''),
            away_win_token=teams.get(away_key, {}).get('token_id', ''),
            draw_token=match_data.get('draw_token_id', ''),
        )
    except Exception:
        return None
