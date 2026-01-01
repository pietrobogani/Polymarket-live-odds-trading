"""
Polymarket Exchange Adapter

This module provides a PolymarketExchange class that implements the same interface
as the Exchange class, allowing the EnvelopeStrategy to trade on Polymarket
prediction markets.

Symbol format: "{token_id}:YES" or "{token_id}:NO"
Example: "0x1234...abcd:YES"
"""

import time
import requests
import pandas as pd
from typing import Any, Dict, List, Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs,
    BalanceAllowanceParams,
    AssetType,
    OpenOrderParams,
    MarketOrderArgs,
    OrderType,
)
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.constants import POLYGON


class PolymarketExchange:
    """
    Polymarket exchange adapter that implements the same interface as the Exchange class.

    Symbol format: "{token_id}:YES" or "{token_id}:NO"
    Example: "71321045679252212594626385532706912750332728571942532289631379312455583992563:YES"

    Usage:
        # Read-only (no trading)
        exchange = PolymarketExchange()

        # With authentication for trading
        exchange = PolymarketExchange(api_setup={
            "private_key": "0x...",
            "signature_type": 0,  # 0=EOA, 1=Email/Magic, 2=Browser proxy
        })
    """

    CLOB_HOST = "https://clob.polymarket.com"
    CHAIN_ID = POLYGON  # 137

    def __init__(self, *, api_setup: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize Polymarket client.

        Args:
            api_setup: Optional dict containing:
                - private_key: Ethereum private key for signing
                - signature_type: 0 (EOA), 1 (Email/Magic), or 2 (Browser proxy)
                - funder: (optional) Address holding funds for proxy wallets
        """
        self.exchange_name = "polymarket"

        if api_setup is None:
            # Read-only client
            self.client = ClobClient(self.CLOB_HOST)
            self.authenticated = False
        else:
            signature_type = api_setup.get("signature_type", 0)
            funder = api_setup.get("funder")

            # Build client kwargs
            client_kwargs = {
                "host": self.CLOB_HOST,
                "key": api_setup.get("private_key"),
                "chain_id": self.CHAIN_ID,
                "signature_type": signature_type,
            }

            # Only add funder if provided (needed for proxy wallets with signature_type=2)
            if funder:
                client_kwargs["funder"] = funder

            self.client = ClobClient(**client_kwargs)
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
            self.authenticated = True

            # Log the configuration
            print(f"Polymarket client initialized:")
            print(f"  Signature type: {signature_type}")
            print(f"  Funder (proxy): {funder or 'Not set (using EOA)'}")

        self._active_symbol = None
        self._markets_cache = {}

    def load_markets(self) -> Dict[str, Any]:
        """Load available markets from Polymarket."""
        try:
            markets = self.client.get_simplified_markets()
            for market in markets:
                condition_id = market.get("condition_id")
                if condition_id:
                    self._markets_cache[condition_id] = market
            return self._markets_cache
        except Exception as e:
            raise Exception(f"Failed to load markets: {e}")

    def _parse_symbol(self, symbol: str) -> tuple:
        """
        Parse symbol into (token_id, outcome).

        Args:
            symbol: Format "{token_id}:YES" or "{token_id}:NO"

        Returns:
            Tuple of (token_id, outcome)
        """
        if ":" not in symbol:
            raise ValueError(
                f"Invalid symbol format: {symbol}. Expected 'token_id:YES' or 'token_id:NO'"
            )
        parts = symbol.rsplit(":", 1)
        outcome = parts[1].upper()
        if outcome not in ("YES", "NO"):
            raise ValueError(f"Invalid outcome: {outcome}. Must be 'YES' or 'NO'")
        return parts[0], outcome

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current price for a symbol.

        Args:
            symbol: Trading symbol in format "token_id:YES" or "token_id:NO"

        Returns:
            Dict with symbol, last, bid, ask prices
        """
        token_id, outcome = self._parse_symbol(symbol)

        try:
            midpoint = self.client.get_midpoint(token_id)
            # Handle if midpoint returns a dict
            if isinstance(midpoint, dict):
                price = float(midpoint.get("mid", midpoint.get("price", 0.5)))
            else:
                price = float(midpoint)
        except Exception:
            # Fallback to get_price if midpoint fails
            try:
                side = BUY if outcome == "YES" else SELL
                price_response = self.client.get_price(token_id, side)
                if isinstance(price_response, dict):
                    price = float(price_response.get("price", 0.5))
                else:
                    price = float(price_response)
            except Exception:
                price = 0.5  # Default fallback

        return {
            "symbol": symbol,
            "last": price,
            "bid": price,
            "ask": price,
        }

    def fetch_min_amount_tradable(self, symbol: str) -> float:
        """
        Return minimum tradable amount.

        Polymarket minimum is typically 5 shares for most markets.
        """
        return 5.0

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        """Convert amount to string with appropriate precision."""
        return str(round(amount, 2))

    def price_to_precision(self, symbol: str, price: float) -> str:
        """Convert price to string with appropriate precision (0-1 range)."""
        return str(round(price, 4))

    def fetch_balance(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch balance information.

        Returns dict with USDC balance and optionally position tokens for the active symbol.

        Returns:
            Dict like {"USDC": {"total": 100.0}, "YES": {"total": 50.0}}
        """
        if not self.authenticated:
            return {"USDC": {"total": 0}}

        result = {}

        # Get USDC (collateral) balance
        try:
            collateral = self.client.get_balance_allowance(
                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            # USDC has 6 decimals
            result["USDC"] = {"total": float(collateral.get("balance", 0)) / 1e6}
        except Exception:
            result["USDC"] = {"total": 0}

        # If tracking a specific symbol, get that position's balance
        if self._active_symbol:
            token_id, outcome = self._parse_symbol(self._active_symbol)
            try:
                conditional = self.client.get_balance_allowance(
                    params=BalanceAllowanceParams(
                        asset_type=AssetType.CONDITIONAL,
                        token_id=token_id,
                    )
                )
                result[outcome] = {"total": float(conditional.get("balance", 0))}
            except Exception:
                result[outcome] = {"total": 0}

        return result

    def fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch open orders for a symbol.

        Args:
            symbol: Trading symbol in format "token_id:YES" or "token_id:NO"

        Returns:
            List of order dicts with id, symbol, side, price, amount, info
        """
        if not self.authenticated:
            return []

        self._active_symbol = symbol
        token_id, _ = self._parse_symbol(symbol)

        try:
            orders = self.client.get_orders(OpenOrderParams(market=token_id))

            # Convert to expected format
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    "id": order.get("id"),
                    "symbol": symbol,
                    "side": "buy" if order.get("side") == "BUY" else "sell",
                    "price": float(order.get("price", 0)),
                    "amount": float(order.get("size", 0)),
                    "info": {
                        "tradeSide": "open",
                        "amount": float(order.get("size", 0)),
                        "price": float(order.get("price", 0)),
                    },
                })
            return formatted_orders
        except Exception as e:
            raise Exception(f"Failed to fetch open orders: {e}")

    def fetch_closed_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch closed orders for a symbol."""
        # Polymarket doesn't have a direct closed orders endpoint in py-clob-client
        # Return empty list for now
        return []

    def cancel_order(self, id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an order by ID.

        Args:
            id: Order ID to cancel
            symbol: Trading symbol (used for consistency with Exchange interface)

        Returns:
            Cancellation result
        """
        if not self.authenticated:
            raise Exception("Authentication required to cancel orders")

        try:
            return self.client.cancel(id)
        except Exception as e:
            raise Exception(f"Failed to cancel order {id}: {e}")

    def fetch_recent_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data from Polymarket price history.

        Note: Volume is not available from Polymarket's price history endpoint,
        so the volume column will be set to 0.

        Args:
            symbol: Trading symbol in format "token_id:YES" or "token_id:NO"
            timeframe: Candle timeframe ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d")
            limit: Number of candles to fetch

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: DatetimeIndex named "timestamp"
        """
        token_id, _ = self._parse_symbol(symbol)
        self._active_symbol = symbol

        # Map timeframe to fidelity (in minutes)
        fidelity_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "1d": 1440,
        }
        fidelity = fidelity_map.get(timeframe, 60)

        # Calculate time range
        end_ts = int(time.time())
        interval_seconds = fidelity * 60
        start_ts = end_ts - (limit * interval_seconds)

        # Fetch price history
        try:
            response = requests.get(
                f"{self.CLOB_HOST}/prices-history",
                params={
                    "market": token_id,
                    "startTs": start_ts,
                    "endTs": end_ts,
                    "fidelity": fidelity,
                },
            )
            response.raise_for_status()
            history = response.json().get("history", [])
        except Exception as e:
            raise Exception(f"Failed to fetch OHLCV data for {symbol}: {e}")

        if not history:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Aggregate into OHLCV candles
        df = self._aggregate_to_ohlcv(history, interval_seconds)
        return df

    def _aggregate_to_ohlcv(self, history: List[Dict], interval_seconds: int) -> pd.DataFrame:
        """
        Aggregate price points into OHLCV candles.

        Args:
            history: List of {"t": timestamp, "p": price} dicts
            interval_seconds: Candle interval in seconds

        Returns:
            DataFrame with OHLCV columns
        """
        if not history:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Convert to DataFrame
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s")
        df["price"] = df["p"].astype(float)

        # Create time buckets
        df["bucket"] = df["timestamp"].dt.floor(f"{interval_seconds}s")

        # Aggregate
        ohlcv = df.groupby("bucket").agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
        ).reset_index()

        ohlcv["volume"] = 0  # Volume not available from Polymarket
        ohlcv.set_index("bucket", inplace=True)
        ohlcv.index.name = "timestamp"

        return ohlcv

    def place_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """
        Place a market order.

        Args:
            symbol: Trading symbol in format "token_id:YES" or "token_id:NO"
            side: "buy" or "sell"
            amount: Amount in USDC for market orders

        Returns:
            Order result
        """
        if not self.authenticated:
            raise Exception("Authentication required to place orders")

        token_id, _ = self._parse_symbol(symbol)
        poly_side = BUY if side.lower() == "buy" else SELL

        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=poly_side,
            )
            signed_order = self.client.create_market_order(order_args)
            result = self.client.post_order(signed_order, OrderType.FOK)
            return result
        except Exception as e:
            raise Exception(f"Failed to place market order: {e}")

    def place_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> Dict[str, Any]:
        """
        Place a limit order.

        Args:
            symbol: Trading symbol in format "token_id:YES" or "token_id:NO"
            side: "buy" or "sell"
            amount: Number of shares
            price: Limit price (must be between 0.01 and 0.99 for Polymarket)

        Returns:
            Order result
        """
        if not self.authenticated:
            raise Exception("Authentication required to place orders")

        token_id, _ = self._parse_symbol(symbol)
        poly_side = BUY if side.lower() == "buy" else SELL

        # Validate price is in valid range for Polymarket
        if not 0.01 <= price <= 0.99:
            raise ValueError(
                f"Price must be between 0.01 and 0.99 for Polymarket, got {price}"
            )

        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=amount,
                side=poly_side,
            )
            signed_order = self.client.create_order(order_args)
            result = self.client.post_order(signed_order, OrderType.GTC)
            return result
        except Exception as e:
            raise Exception(f"Failed to place limit order: {e}")

    def create_signed_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> Dict[str, Any]:
        """
        Create and sign a limit order WITHOUT posting it.

        Use this for pre-signing orders that will be posted later.
        Call post_presigned_order() to submit the signed order.

        Args:
            symbol: Trading symbol in format "token_id:YES" or "token_id:NO"
            side: "buy" or "sell"
            amount: Number of shares
            price: Limit price (must be between 0.01 and 0.99)

        Returns:
            Dict with signed_order object and metadata
        """
        if not self.authenticated:
            raise Exception("Authentication required to sign orders")

        token_id, _ = self._parse_symbol(symbol)
        poly_side = BUY if side.lower() == "buy" else SELL

        if not 0.01 <= price <= 0.99:
            raise ValueError(
                f"Price must be between 0.01 and 0.99 for Polymarket, got {price}"
            )

        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=amount,
                side=poly_side,
            )
            signed_order = self.client.create_order(order_args)
            return {
                "signed_order": signed_order,
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "token_id": token_id,
                "created_at": time.time(),
            }
        except Exception as e:
            raise Exception(f"Failed to create signed order: {e}")

    def post_presigned_order(self, presigned: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post a pre-signed order to Polymarket.

        Args:
            presigned: Dict returned by create_signed_order()

        Returns:
            Order result from Polymarket
        """
        if not self.authenticated:
            raise Exception("Authentication required to post orders")

        try:
            signed_order = presigned["signed_order"]
            result = self.client.post_order(signed_order, OrderType.GTC)
            return result
        except Exception as e:
            raise Exception(f"Failed to post pre-signed order: {e}")

    def fetch_trades(
        self, token_id: str, after_ts: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent trades for a token.

        Args:
            token_id: Token ID to fetch trades for
            after_ts: Unix timestamp - only return trades after this time
            limit: Maximum number of trades to return

        Returns:
            List of trade dicts with keys: id, price, size, side, timestamp
        """
        try:
            # Build query params
            params = {"asset_id": token_id}
            if after_ts:
                params["after"] = after_ts

            response = requests.get(
                f"{self.CLOB_HOST}/trades",
                params=params,
            )
            response.raise_for_status()
            trades = response.json()

            # Format trades
            formatted = []
            for trade in trades[:limit]:
                formatted.append({
                    "id": trade.get("id"),
                    "price": float(trade.get("price", 0)),
                    "size": float(trade.get("size", 0)),
                    "side": trade.get("side", "").lower(),
                    "timestamp": trade.get("match_time") or trade.get("timestamp"),
                })
            return formatted
        except Exception as e:
            # Don't raise, just return empty - volume is non-critical
            return []

    def get_volume_since(
        self, token_ids: List[str], after_ts: int
    ) -> Dict[str, float]:
        """
        Get total trade volume for tokens since a timestamp.

        Args:
            token_ids: List of token IDs
            after_ts: Unix timestamp - sum trades after this time

        Returns:
            Dict of token_id -> volume (in USD)
        """
        volumes = {}
        for token_id in token_ids:
            trades = self.fetch_trades(token_id, after_ts=after_ts)
            total_volume = sum(
                t["price"] * t["size"] for t in trades if t.get("price") and t.get("size")
            )
            volumes[token_id] = total_volume
        return volumes

    def roll_time(self):
        """No-op for live trading (used in backtesting only)."""
        return False

    def set_sandbox_mode(self, value: bool):
        """No-op - Polymarket doesn't have a sandbox mode."""
        pass
