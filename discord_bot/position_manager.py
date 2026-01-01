"""
Position Manager - Handles position lifecycle after trades.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List, Callable
from enum import Enum

logger = logging.getLogger("position_manager")


class PositionState(Enum):
    BOUGHT = "bought"
    SELL_PENDING = "sell_pending"
    CLOSED = "closed"


@dataclass
class Position:
    token_id: str
    symbol: str
    market_name: str
    amount: float
    entry_price: float
    target_price: float
    entry_time: datetime
    state: PositionState = PositionState.BOUGHT
    channel_id: Optional[str] = None


class PositionManager:
    """Manages open positions and exit timing."""

    def __init__(self, exchange, notify_callback: Optional[Callable] = None):
        self.exchange = exchange
        self.notify_callback = notify_callback
        self._positions: Dict[str, List[Position]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    def add_position(self, token_id: str, symbol: str, market_name: str,
                     amount: float, buy_price: float, predicted_sell_price: float,
                     channel_id: str) -> Position:
        """Track a new position."""
        position = Position(
            token_id=token_id,
            symbol=symbol,
            market_name=market_name,
            amount=amount,
            entry_price=buy_price,
            target_price=predicted_sell_price,
            entry_time=datetime.utcnow(),
            channel_id=channel_id,
        )

        if channel_id not in self._positions:
            self._positions[channel_id] = []
        self._positions[channel_id].append(position)

        # Start exit task
        self._schedule_exit(position)
        return position

    def _schedule_exit(self, position: Position) -> None:
        """Schedule position exit. Implementation details omitted."""
        pass

    async def emergency_close(self, channel_id: str) -> int:
        """Close all positions for channel immediately."""
        # Implementation handles order cancellation and market sells
        return 0

    def has_open_positions(self, channel_id: str) -> bool:
        positions = self._positions.get(channel_id, [])
        return any(p.state != PositionState.CLOSED for p in positions)

    def get_open_positions(self, channel_id: str) -> List[Position]:
        positions = self._positions.get(channel_id, [])
        return [p for p in positions if p.state != PositionState.CLOSED]
