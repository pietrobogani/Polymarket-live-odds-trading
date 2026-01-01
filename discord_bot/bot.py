"""
Discord Trading Bot for Polymarket

Listens for goal messages in a specific Discord channel and places
market orders on Polymarket.

Uses Poisson probability model to determine optimal buy decisions:
- Fetches 1X2 prices every minute during live matches
- Pre-computes "what-if" scenarios for goals
- Buys ALL markets where probability increases by >=5%
- Caps buy price at model-predicted post-goal probability (max 0.98)

Usage:
    python -m discord_bot.bot

Environment Variables:
    DISCORD_BOT_TOKEN: Your Discord bot token
    DISCORD_CHANNEL_ID: Channel ID to monitor
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime

import discord
from discord.ext import commands
from discord import ui

from . import config
from .parser import MessageParser, GoalEvent
from .match_runner import AsyncMatchRunner, MatchConfig, load_match_config_from_json, PreSignedOrders
from .position_manager import PositionManager
from .latency_tracker import LatencyTracker, LatencyMetrics, now_ms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("discord_bot")


class GoalButton(ui.Button):
    """A button that triggers a goal event when clicked."""

    def __init__(self, team_name: str, label: str, style: discord.ButtonStyle, bot: "TradingBot", channel_id: int):
        # custom_id makes the button persistent across bot restarts
        custom_id = f"goal_{channel_id}_{team_name.replace(' ', '_')}"
        super().__init__(label=label, style=style, custom_id=custom_id)
        self.team_name = team_name
        self.bot = bot
        self.channel_id = channel_id

    async def callback(self, interaction: discord.Interaction):
        """Handle button click - trigger goal for this team."""
        channel_id = interaction.channel_id

        # Create goal event
        goal_message = f"goal {self.team_name}"
        event = self.bot.parser.parse(goal_message, channel_id=channel_id)

        if event:
            # Acknowledge the button click immediately
            await interaction.response.send_message(
                f"**⚽ GOAL {self.team_name.upper()}!**",
                ephemeral=False
            )
            # Create a wrapper that allows sending to channel
            class ChannelWrapper:
                def __init__(self, channel):
                    self.channel = channel
                    self.author = interaction.user

            wrapper = ChannelWrapper(interaction.channel)
            wrapper.channel = interaction.channel

            # Process the goal
            await self.bot._handle_goal_event(event, wrapper)
        else:
            await interaction.response.send_message(
                f"Failed to parse goal for {self.team_name}",
                ephemeral=True
            )


class UndoButton(ui.Button):
    """A button that triggers goal disallowed (undo) when clicked."""

    def __init__(self, bot: "TradingBot", channel_id: int):
        custom_id = f"undo_{channel_id}"
        super().__init__(label="GOAL DISALLOWED", style=discord.ButtonStyle.danger, emoji="🚫", custom_id=custom_id)
        self.bot = bot
        self.channel_id = channel_id

    async def callback(self, interaction: discord.Interaction):
        """Handle undo button click."""
        channel_id = interaction.channel_id

        # Check if there's a goal to undo
        if channel_id not in self.bot.last_goal or self.bot.last_goal[channel_id] is None:
            await interaction.response.send_message("No goal to undo.", ephemeral=True)
            return

        # ESCAPE MECHANISM: Emergency close any open positions
        if self.bot.position_manager.has_open_positions(str(channel_id)):
            await interaction.response.send_message(
                "**EMERGENCY CLOSE: Selling all open positions...**",
                ephemeral=False
            )
            closed_count = await self.bot.position_manager.emergency_close(str(channel_id))
            await interaction.followup.send(f"Emergency closed {closed_count} position(s)")
        else:
            await interaction.response.defer()

        team_name = self.bot.last_goal[channel_id]
        scores = self.bot._get_current_score(channel_id)

        if team_name in scores and scores[team_name] > 0:
            scores[team_name] -= 1
            self.bot.last_goal[channel_id] = None
            self.bot.last_goal_time[channel_id] = 0
            score_str = self.bot._format_score_display(channel_id)
            await interaction.followup.send(f"Goal by {team_name} removed.\n**Score: {score_str}** (cooldown reset)")


class ScoreButton(ui.Button):
    """A button that shows current score."""

    def __init__(self, bot: "TradingBot", channel_id: int):
        custom_id = f"score_{channel_id}"
        super().__init__(label="📊 SCORE", style=discord.ButtonStyle.secondary, custom_id=custom_id)
        self.bot = bot
        self.channel_id = channel_id

    async def callback(self, interaction: discord.Interaction):
        """Show current score."""
        channel_id = interaction.channel_id
        score_str = self.bot._format_score_display(channel_id)
        await interaction.response.send_message(f"**Current Score: {score_str}**", ephemeral=True)


class GoalButtonView(ui.View):
    """A view containing goal buttons for a match."""

    def __init__(self, home_team: str, away_team: str, bot: "TradingBot", channel_id: int):
        # timeout=None makes buttons persistent (work even after bot restart)
        super().__init__(timeout=None)
        self.bot = bot
        self.channel_id = channel_id

        # Add goal buttons for each team
        self.add_item(GoalButton(
            team_name=home_team,
            label=f"⚽ {home_team.upper()}",
            style=discord.ButtonStyle.success,
            bot=bot,
            channel_id=channel_id
        ))
        self.add_item(GoalButton(
            team_name=away_team,
            label=f"⚽ {away_team.upper()}",
            style=discord.ButtonStyle.primary,
            bot=bot,
            channel_id=channel_id
        ))
        self.add_item(ScoreButton(bot, channel_id))
        self.add_item(UndoButton(bot, channel_id))


class TradingBot(commands.Bot):
    """
    Discord bot that listens for goal messages and executes Polymarket orders.

    Uses Poisson probability model to determine optimal buy decisions:
    - Fetches 1X2 prices every minute during live matches
    - Pre-computes "what-if" scenarios for both home/away goals
    - On goal: buys ALL markets where probability increases by >=5%
    - Caps buy price at model-predicted probability (max 0.98)
    - Respects per-market budget limits
    """

    def __init__(self):
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix="!", intents=intents)

        self.parser = MessageParser()
        self.exchange = None

        # Score tracking: channel_id -> {team_name: goals}
        self.match_scores = {}

        # Last goal tracking for undo: channel_id -> team_name
        self.last_goal = {}

        # Team order mapping: channel_id -> (home_team_key, away_team_key)
        self.team_order = {}

        # Duplicate goal protection: channel_id -> timestamp of last goal
        self.last_goal_time = {}
        self.GOAL_COOLDOWN_SECONDS = 30  # Minimum seconds between goals in same channel

        # Master channel for status updates
        self.master_channel = None

        self._setup_exchange()

        # Initialize match runner with exchange (will start in on_ready)
        self.match_runner = AsyncMatchRunner(self.exchange)

        # Initialize position manager for handling sell orders
        self.position_manager = PositionManager(self.exchange)

        # Initialize latency tracker for performance monitoring
        self.latency_tracker = LatencyTracker(log_dir="logs/latency")

    def _setup_exchange(self):
        """Initialize the Polymarket exchange connection (always needed for price fetching)."""
        if not config.EXECUTE_ORDERS:
            logger.info("Order execution is DISABLED (test mode)")

        try:
            # Load Polymarket credentials
            with open(config.SECRET_FILE, "r") as f:
                secrets = json.load(f)

            poly_config = secrets.get("polymarket", {})
            api_setup = {
                "private_key": poly_config.get("private_key"),
                "signature_type": poly_config.get("signature_type", 2),
                "funder": poly_config.get("funder"),
            }

            # Import and initialize exchange
            import sys
            import os

            # Add parent directory to path for imports
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            from utilities.polymarket_exchange import PolymarketExchange

            self.exchange = PolymarketExchange(api_setup=api_setup)
            logger.info("Polymarket exchange initialized successfully")

        except FileNotFoundError:
            logger.error(f"Secret file not found: {config.SECRET_FILE}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")

    async def on_ready(self):
        """Called when the bot is ready and connected."""
        logger.info(f"Bot connected as {self.user}")
        logger.info(f"Monitoring {len(self.parser.active_channels)} channels")
        logger.info(f"Order execution: {'ENABLED' if config.EXECUTE_ORDERS else 'DISABLED'}")
        logger.info(self.parser.get_matches_summary())

        # Set up master channel for status updates
        if config.MASTER_CHANNEL_ID:
            self.master_channel = self.get_channel(config.MASTER_CHANNEL_ID)
            if self.master_channel:
                logger.info(f"Master channel set: {self.master_channel.name} ({config.MASTER_CHANNEL_ID})")
                self.match_runner.set_status_callback(self._send_status_to_master)
            else:
                logger.warning(f"Master channel {config.MASTER_CHANNEL_ID} not found")
        else:
            self.master_channel = None
            logger.info("No master channel configured - status updates disabled")

        # Set up position manager notification callback
        self.position_manager.notify_callback = self._send_position_notification

        # Load match configurations and start background runner
        await self._load_match_configs()
        await self.match_runner.start()
        logger.info("Match runner started - fetching prices every minute")

        # Auto-send goal buttons to all match channels
        await self._send_buttons_to_all_channels()

    async def _send_position_notification(self, channel_id: str, message: str) -> None:
        """Send position-related notification to a channel."""
        try:
            channel = self.get_channel(int(channel_id))
            if channel:
                await channel.send(f"**[Position Manager]** {message}")
        except Exception as e:
            logger.error(f"Failed to send position notification: {e}")

    async def _send_status_to_master(self, message: str) -> None:
        """Send a status message to the master channel."""
        if self.master_channel:
            try:
                await self.master_channel.send(message)
            except Exception as e:
                logger.error(f"Failed to send to master channel: {e}")

    async def _load_match_configs(self):
        """Load match configurations from matches.json into the runner."""
        try:
            with open(config.MATCHES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            default_budget = data.get("default_budget_per_market", 500)

            for match_data in data.get("active_matches", []):
                channel_id = match_data.get("channel_id")
                if not channel_id:
                    continue

                # Determine team order (first team listed is "home")
                teams = match_data.get("teams", {})
                team_keys = list(teams.keys())
                if len(team_keys) < 2:
                    logger.warning(f"Match {match_data.get('name')} has less than 2 teams")
                    continue

                home_team, away_team = team_keys[0], team_keys[1]
                self.team_order[int(channel_id)] = (home_team, away_team)

                # Load match config for the runner
                match_config = load_match_config_from_json(match_data, (home_team, away_team))
                if match_config:
                    # Override budget if specified in match data
                    if "budget_per_market" in match_data:
                        match_config.budget_per_market = match_data["budget_per_market"]
                    else:
                        match_config.budget_per_market = default_budget

                    self.match_runner.add_match(match_config)
                    logger.info(f"Loaded match config: {match_config.name} "
                               f"(home: {home_team}, away: {away_team}, "
                               f"budget: ${match_config.budget_per_market}/market)")

        except Exception as e:
            logger.error(f"Failed to load match configs: {e}")

    async def on_message(self, message: discord.Message):
        """Process incoming messages."""
        # Ignore messages from the bot itself
        if message.author == self.user:
            return

        # Only process messages from configured match channels
        if not self.parser.is_active_channel(message.channel.id):
            return

        content_lower = message.content.lower().strip()

        # Check for undo command
        if content_lower in ("undo", "undo goal", "forget last goal", "cancel last goal"):
            await self._handle_undo(message)
            return

        # Check for score command
        if content_lower in ("score", "scores", "current score"):
            await self._handle_score_request(message)
            return

        # Check for buttons command
        if content_lower in ("buttons", "!buttons", "show buttons", "goals"):
            await self._send_goal_buttons(message)
            return

        # Goals can only be signaled through buttons, not text commands

    async def _handle_undo(self, message: discord.Message):
        """Handle undo command to revert the last goal and emergency close positions."""
        channel_id = message.channel.id

        # ESCAPE MECHANISM: Emergency close any open positions
        if self.position_manager.has_open_positions(str(channel_id)):
            await message.channel.send("**EMERGENCY CLOSE: Selling all open positions...**")
            closed_count = await self.position_manager.emergency_close(str(channel_id))
            await message.channel.send(f"Emergency closed {closed_count} position(s)")
            logger.warning(f"Emergency close triggered for channel {channel_id}: {closed_count} positions")

        # Check if there's a goal to undo
        if channel_id not in self.last_goal or self.last_goal[channel_id] is None:
            await message.channel.send("No goal to undo.")
            logger.info(f"Undo requested but no goal to undo in channel {channel_id}")
            return

        team_name = self.last_goal[channel_id]
        scores = self._get_current_score(channel_id)

        # Check if team has goals to remove
        if team_name not in scores or scores[team_name] <= 0:
            await message.channel.send(f"Cannot undo: {team_name} has no goals.")
            logger.warning(f"Undo failed: {team_name} has no goals")
            return

        # Decrement the goal
        scores[team_name] -= 1
        score_str = self._format_score_display(channel_id)

        # Clear last goal (can only undo once)
        self.last_goal[channel_id] = None

        # Reset cooldown so user can type new goal immediately
        self.last_goal_time[channel_id] = 0

        # Send confirmation
        await message.channel.send(f"Goal by {team_name} removed.\n**Score: {score_str}** (cooldown reset)")
        logger.info(f"Undo: removed goal by {team_name}, new score: {score_str}")

    async def _handle_score_request(self, message: discord.Message):
        """Handle request to display current score."""
        channel_id = message.channel.id
        score_str = self._format_score_display(channel_id)
        await message.channel.send(f"**Current Score: {score_str}**")

    async def _send_goal_buttons(self, message: discord.Message):
        """Send goal buttons for the match in this channel."""
        channel_id = message.channel.id

        # Get team names for this channel
        if channel_id not in self.team_order:
            await message.channel.send("No match configured for this channel.")
            return

        home_team, away_team = self.team_order[channel_id]
        score_str = self._format_score_display(channel_id)

        # Create button view
        view = GoalButtonView(home_team, away_team, self, channel_id)

        # Send buttons with current score
        await message.channel.send(
            f"**{home_team.upper()} vs {away_team.upper()}**\n"
            f"Score: {score_str}\n\n"
            f"Click a button to record a goal:",
            view=view
        )
        logger.info(f"Goal buttons sent to channel {channel_id}")

    async def _send_buttons_to_channel(self, channel_id: int):
        """Send goal buttons to a specific channel (used on startup)."""
        channel = self.get_channel(channel_id)
        if not channel:
            logger.warning(f"Channel {channel_id} not found - cannot send buttons")
            return

        if channel_id not in self.team_order:
            return

        home_team, away_team = self.team_order[channel_id]

        # Create and register persistent view
        view = GoalButtonView(home_team, away_team, self, channel_id)
        self.add_view(view)

        # Send buttons
        await channel.send(
            f"**🎮 GOAL BUTTONS READY**\n"
            f"**{home_team.upper()} vs {away_team.upper()}**\n\n"
            f"Click to record a goal:",
            view=view
        )
        logger.info(f"Auto-sent goal buttons to channel {channel_id}")

    async def _send_buttons_to_all_channels(self):
        """Send goal buttons to all configured match channels on startup."""
        logger.info("Sending goal buttons to all match channels...")

        for channel_id in self.team_order.keys():
            try:
                await self._send_buttons_to_channel(channel_id)
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to send buttons to channel {channel_id}: {e}")

        logger.info(f"Goal buttons sent to {len(self.team_order)} channels")

    def _get_current_score(self, channel_id: int) -> dict:
        """Get current score for a match channel."""
        if channel_id not in self.match_scores:
            self.match_scores[channel_id] = {}
        return self.match_scores[channel_id]

    def _update_score(self, channel_id: int, team_name: str) -> tuple:
        """
        Update score and return (scoring_team_goals, opponent_goals, goal_difference).

        Returns:
            Tuple of (scoring_team_goals, opponent_goals, goal_diff_after)
        """
        scores = self._get_current_score(channel_id)

        # Initialize team if first goal
        if team_name not in scores:
            scores[team_name] = 0

        # Increment goal
        scores[team_name] += 1
        scoring_team_goals = scores[team_name]

        # Calculate opponent goals (sum of all other teams' goals)
        opponent_goals = sum(g for t, g in scores.items() if t != team_name)

        # Goal difference from perspective of scoring team
        goal_diff = scoring_team_goals - opponent_goals

        return scoring_team_goals, opponent_goals, goal_diff

    def _calculate_price_ceiling(self, pre_goal_price: float, goal_diff: int) -> float:
        """
        Calculate maximum price to pay for WIN bet based on goal difference.

        Uses proprietary formula based on goal difference and market dynamics.
        """
        if goal_diff <= 0:
            return 0.0

        # Price ceiling calculation based on goal advantage
        # Implementation details omitted for public repository
        price_ceiling = self._compute_ceiling(pre_goal_price, goal_diff)
        return min(0.99, max(0.0, price_ceiling))

    def _calculate_draw_price_ceiling(self, pre_goal_price: float, abs_goal_diff: int) -> float:
        """
        Calculate maximum price to pay for DRAW bet based on goal difference.

        Uses proprietary formula accounting for draw probability dynamics.
        """
        # Draw price ceiling calculation
        # Implementation details omitted for public repository
        price_ceiling = self._compute_ceiling(pre_goal_price, abs_goal_diff + 1)
        return min(0.99, max(0.0, price_ceiling))

    def _compute_ceiling(self, base_price: float, factor: int) -> float:
        """Compute price ceiling with safety margin. Override in production."""
        # Placeholder implementation - configure for live trading
        return base_price * 1.05

    async def _handle_goal_event(self, event: GoalEvent, message: discord.Message):
        """
        Handle a detected goal event using Poisson model for buy decisions.

        Strategy (Poisson model):
        - Get pre-computed buy decisions from match runner
        - Use PRE-SIGNED orders when available (saves ~150-300ms)
        - Place MARKET orders on ALL markets where probability increases by >=5%
        - Track positions for automatic selling after 2-3 minutes
        - Respect per-market budget limits

        Args:
            event: The parsed goal event
            message: The original Discord message
        """
        channel_id = message.channel.id

        # START LATENCY TRACKING
        goal_id = f"goal_{channel_id}_{int(time.time() * 1000)}"
        metrics = self.latency_tracker.start(goal_id, str(channel_id))

        # DUPLICATE PROTECTION: Check cooldown
        now = time.time()
        last_time = self.last_goal_time.get(channel_id, 0)
        if now - last_time < self.GOAL_COOLDOWN_SECONDS:
            await message.channel.send(
                f"Goal ignored (duplicate protection: {self.GOAL_COOLDOWN_SECONDS}s cooldown). "
                f"Wait {self.GOAL_COOLDOWN_SECONDS - (now - last_time):.0f}s or type 'undo' first."
            )
            logger.warning(f"Goal ignored due to cooldown in channel {channel_id}")
            return
        self.last_goal_time[channel_id] = now

        metrics.mark_validation_done()

        # Determine if home or away team scored
        team_order = self.team_order.get(channel_id)
        if not team_order:
            logger.warning(f"No team order configured for channel {channel_id}")
            await message.channel.send("Match not configured for Poisson model - using fallback")
            await self._handle_goal_event_fallback(event, message)
            return

        home_team, away_team = team_order
        home_scores = event.team_name.lower() == home_team.lower()

        # Update score tracking
        scoring_goals, opponent_goals, goal_diff = self._update_score(channel_id, event.team_name)
        score_display = self._format_score_display(channel_id)

        # Track last goal for undo functionality
        self.last_goal[channel_id] = event.team_name

        # Update match runner with new score
        scores = self._get_current_score(channel_id)
        home_goals = scores.get(home_team, 0)
        away_goals = scores.get(away_team, 0)
        self.match_runner.update_score(str(channel_id), home_goals, away_goals)

        # Get buy decisions from Poisson model
        decisions = self.match_runner.get_buy_decisions(str(channel_id), home_scores)

        # Get pre-signed orders (speed optimization)
        presigned = self.match_runner.get_presigned_orders(str(channel_id), home_scores)

        metrics.mark_decisions_ready()

        if decisions is None:
            logger.warning(f"No Poisson decisions available for channel {channel_id}")
            await message.channel.send(
                f"**GOAL {event.team_name.upper()}!**\n"
                f"Score: **{score_display}**\n"
                f"Model not ready - match may not have started yet"
            )
            return

        # Track pre-signing info
        if presigned and presigned.orders:
            metrics.used_presigned = True
            metrics.presign_age_ms = presigned.age_ms()

        logger.info(
            f"Goal detected! Team: {event.team_name} ({'home' if home_scores else 'away'}), "
            f"Match: {event.match_name}, "
            f"Score: {home_goals}-{away_goals}, "
            f"User: {message.author}"
        )

        # Build orders to place (silent - no verbose output to channel)
        orders_to_place = []

        # Map decision keys to token IDs and names
        # Includes all markets: 1X2 + O/U + Handicaps
        match_state = self.match_runner.get_match_state(str(channel_id))
        token_map = {}
        if match_state:
            cfg = match_state.config
            # Base 1X2 markets
            token_map = {
                'home_win': (cfg.home_win_token, f"{home_team} WIN"),
                'away_win': (cfg.away_win_token, f"{away_team} WIN"),
                'draw': (cfg.draw_token, "DRAW"),
            }
            # Add O/U tokens if configured
            if cfg.over_2_5_token:
                token_map['over_2_5'] = (cfg.over_2_5_token, "OVER 2.5")
            if cfg.under_2_5_token:
                token_map['under_2_5'] = (cfg.under_2_5_token, "UNDER 2.5")
            if cfg.over_1_5_token:
                token_map['over_1_5'] = (cfg.over_1_5_token, "OVER 1.5")
            if cfg.under_1_5_token:
                token_map['under_1_5'] = (cfg.under_1_5_token, "UNDER 1.5")
            # Add handicap tokens if configured
            if cfg.home_minus_1_5_token:
                token_map['home_minus_1_5'] = (cfg.home_minus_1_5_token, f"{home_team} -1.5")
            if cfg.home_plus_1_5_token:
                token_map['home_plus_1_5'] = (cfg.home_plus_1_5_token, f"{home_team} +1.5")
            if cfg.away_minus_1_5_token:
                token_map['away_minus_1_5'] = (cfg.away_minus_1_5_token, f"{away_team} -1.5")
            if cfg.away_plus_1_5_token:
                token_map['away_plus_1_5'] = (cfg.away_plus_1_5_token, f"{away_team} +1.5")

        buy_targets = []
        for market_key, decision in decisions.items():
            if market_key not in token_map:
                continue

            token_id, market_name = token_map[market_key]
            buy = decision.get('buy', False)
            price_ceiling = decision.get('price_ceiling', 0)
            prob_increase = decision.get('probability_increase', 0)

            if buy and token_id:
                buy_targets.append(market_name)
                orders_to_place.append({
                    'market': market_key,
                    'name': market_name,
                    'token_id': token_id,
                    'price_ceiling': price_ceiling,
                    'prob_increase': prob_increase,
                })
                # Log decision details (not to channel)
                logger.info(f"  BUY {market_name} @ {price_ceiling:.2f} (+{prob_increase*100:.1f}%)")
            else:
                logger.info(f"  SKIP {market_name} (+{prob_increase*100:.1f}%)")

        # Add context to metrics
        metrics.team_scored = event.team_name
        metrics.score_after = score_display

        metrics.mark_orders_built()

        # Send SHORT confirmation to match channel
        buy_str = f" → Buying: {', '.join(buy_targets)}" if buy_targets else ""
        await message.channel.send(f"**⚽ GOAL {event.team_name.upper()}!** Score: {score_display}{buy_str}")

        # Log decisions
        self._log_order(
            event, message, success=True, test_mode=not config.EXECUTE_ORDERS,
            extra={
                "poisson_model": True,
                "decisions": decisions,
                "score": f"{home_goals}-{away_goals}",
                "home_scores": home_scores,
            }
        )

        # Place orders if execution is enabled
        if not config.EXECUTE_ORDERS:
            logger.info(f"Order execution disabled - would place {len(orders_to_place)} orders")
            metrics.mark_orders_submitted()
            metrics.mark_discord_confirmed()
            self.latency_tracker.finish(metrics)
            await message.channel.send(metrics.summary())
            return

        if not self.exchange:
            logger.error("Exchange not initialized - cannot place orders")
            return

        # Get budget per market
        budget = 500  # default
        if match_state:
            budget = match_state.config.budget_per_market

        # Place all orders IN PARALLEL for speed
        # Use pre-signed orders when available (saves ~50-100ms per order)
        async def place_single_order(order_info: dict) -> dict:
            """Place a single order and return result with fill info."""
            market_key = order_info['market']
            symbol = f"{order_info['token_id']}:YES"
            shares_to_buy = budget / order_info['price_ceiling']

            sign_ms = 0
            post_ms = 0

            try:
                # Check if we have a pre-signed order for this market
                presigned_order = None
                if presigned and presigned.orders:
                    presigned_order = presigned.orders.get(market_key)

                if presigned_order:
                    # USE PRE-SIGNED ORDER (skip signing step)
                    sign_ms = 0  # Already signed
                    t_post_start = now_ms()
                    order_result = self.exchange.post_presigned_order(presigned_order)
                    post_ms = now_ms() - t_post_start
                    logger.info(f"Used pre-signed order for {order_info['name']} (age: {presigned.age_ms():.0f}ms)")
                else:
                    # SIGN AND POST (fallback)
                    t_sign_start = now_ms()
                    # We need to sign fresh - use place_limit_order which does both
                    order_result = self.exchange.place_limit_order(
                        symbol=symbol,
                        side="buy",
                        amount=shares_to_buy,
                        price=order_info['price_ceiling'],
                    )
                    total_ms = now_ms() - t_sign_start
                    # Estimate sign vs post (rough split: 30% sign, 70% post)
                    sign_ms = total_ms * 0.3
                    post_ms = total_ms * 0.7
                    logger.info(f"Signed fresh order for {order_info['name']}")

                # Extract order ID
                order_id = (order_result.get("orderID") or
                           order_result.get("id") or
                           order_result.get("order_id"))

                # Check if order filled by looking at order status
                filled = False
                fill_price = order_info['price_ceiling']
                fill_amount = shares_to_buy

                status = order_result.get("status", "").upper()
                if status in ("FILLED", "MATCHED", "CLOSED"):
                    filled = True
                    if "price" in order_result:
                        fill_price = float(order_result["price"])
                    if "size" in order_result or "filledSize" in order_result:
                        fill_amount = float(order_result.get("filledSize", order_result.get("size", shares_to_buy)))

                # Skip the extra fetch_open_orders check for speed
                # Trust the order response status
                if not filled and status in ("LIVE", "OPEN"):
                    filled = False  # Order is in book, not filled
                elif not filled:
                    # Unknown status, assume filled for safety
                    filled = True

                return {
                    "success": True,
                    "order_info": order_info,
                    "symbol": symbol,
                    "order_id": order_id,
                    "shares_requested": shares_to_buy,
                    "filled": filled,
                    "fill_price": fill_price,
                    "fill_amount": fill_amount,
                    "order_result": order_result,
                    "sign_ms": sign_ms,
                    "post_ms": post_ms,
                }

            except Exception as e:
                return {
                    "success": False,
                    "order_info": order_info,
                    "symbol": symbol,
                    "error": str(e),
                    "sign_ms": sign_ms,
                    "post_ms": post_ms,
                }

        # Execute all orders in parallel
        if orders_to_place:
            order_tasks = [place_single_order(oi) for oi in orders_to_place]
            results = await asyncio.gather(*order_tasks)

            metrics.mark_orders_submitted()

            # Process results and track positions (minimal channel output)
            filled_orders = []
            failed_orders = []

            for result in results:
                order_info = result["order_info"]

                # Track per-order latency
                metrics.add_order_latency(
                    market=order_info['name'],
                    sign_ms=result.get("sign_ms", 0),
                    post_ms=result.get("post_ms", 0),
                    filled=result.get("filled", False),
                    error=result.get("error"),
                )

                if result["success"]:
                    fill_status = "FILLED" if result["filled"] else "PENDING"
                    logger.info(f"{order_info['name']} order {fill_status}: {result.get('order_result')}")

                    # Only track position if order filled
                    if result["filled"]:
                        filled_orders.append(f"{order_info['name']} @ {result['fill_price']:.2f}")
                        self.position_manager.add_position(
                            token_id=order_info['token_id'],
                            symbol=result["symbol"],
                            market_name=order_info['name'],
                            amount=result["fill_amount"],
                            buy_price=result["fill_price"],
                            predicted_sell_price=order_info['price_ceiling'],
                            channel_id=str(channel_id),
                        )
                        logger.info(f"Position tracked: {result['fill_amount']:.1f} shares")
                    else:
                        logger.info(f"Order PENDING in orderbook - position NOT tracked")
                else:
                    logger.error(f"Failed to place {order_info['name']} order: {result['error']}")
                    failed_orders.append(order_info['name'])

            # Send single summary message to channel
            if filled_orders:
                await message.channel.send(f"✅ Filled: {', '.join(filled_orders)}")
            if failed_orders:
                await message.channel.send(f"❌ Failed: {', '.join(failed_orders)}")

            # Send latency report
            metrics.mark_discord_confirmed()
            self.latency_tracker.finish(metrics)
            await message.channel.send(metrics.summary())
        else:
            # No orders to place
            metrics.mark_orders_submitted()
            metrics.mark_discord_confirmed()
            self.latency_tracker.finish(metrics)

    async def _handle_goal_event_fallback(self, event: GoalEvent, message: discord.Message):
        """Fallback handler using simple formula when Poisson model unavailable."""
        channel_id = message.channel.id
        win_symbol = f"{event.token_id}:YES"

        # Fetch pre-goal price
        pre_win_price = 0.5
        if self.exchange:
            try:
                ticker = self.exchange.fetch_ticker(win_symbol)
                pre_win_price = ticker.get("last", 0.5)
            except Exception as e:
                logger.warning(f"Failed to fetch win price: {e}")

        # Update score
        scoring_goals, opponent_goals, goal_diff = self._update_score(channel_id, event.team_name)
        score_display = self._format_score_display(channel_id)

        # Track last goal for undo
        self.last_goal[channel_id] = event.team_name

        # Simple fallback: only bet on WIN if in the lead
        if goal_diff <= 0:
            await message.channel.send(f"**⚽ GOAL {event.team_name.upper()}!** Score: {score_display}")
            logger.info(f"Fallback: No bet (goal diff: {goal_diff:+d})")
            return

        price_ceiling = self._calculate_price_ceiling(pre_win_price, goal_diff)
        await message.channel.send(f"**⚽ GOAL {event.team_name.upper()}!** Score: {score_display} → Buying: WIN")
        logger.info(f"Fallback bet: WIN @ max {price_ceiling:.2f}")

        if config.EXECUTE_ORDERS and self.exchange:
            try:
                order = self.exchange.place_limit_order(
                    symbol=win_symbol,
                    side="buy",
                    amount=event.order_size,
                    price=price_ceiling,
                )
                logger.info(f"Fallback WIN order placed: {order}")
                await message.channel.send(f"✅ Filled: WIN @ {price_ceiling:.2f}")
            except Exception as e:
                logger.error(f"Failed to place fallback order: {e}")
                await message.channel.send(f"❌ Order failed")

    def _format_score(self, channel_id: int, scoring_team: str) -> str:
        """Format current score as string like 'Chelsea 2 - 1 Bournemouth'."""
        scores = self._get_current_score(channel_id)
        if not scores:
            return "0-0"

        # Put scoring team first
        parts = []
        scoring_goals = scores.get(scoring_team, 0)
        parts.append(f"{scoring_team} {scoring_goals}")

        for team, goals in scores.items():
            if team != scoring_team:
                parts.append(f"{goals} {team}")

        return " - ".join(parts) if len(parts) > 1 else parts[0]

    def _format_score_display(self, channel_id: int) -> str:
        """Format score for Discord display like 'Chelsea 2 - 1 Bournemouth'."""
        scores = self._get_current_score(channel_id)
        if not scores:
            return "0 - 0"

        # Sort teams alphabetically for consistent display
        teams = sorted(scores.keys())
        if len(teams) == 0:
            return "0 - 0"
        elif len(teams) == 1:
            return f"{teams[0].title()} {scores[teams[0]]} - 0"
        else:
            return f"{teams[0].title()} {scores[teams[0]]} - {scores[teams[1]]} {teams[1].title()}"

    def _log_order(
        self,
        event: GoalEvent,
        message: discord.Message,
        success: bool,
        order_id: str = None,
        error: str = None,
        test_mode: bool = False,
        extra: dict = None,
    ):
        """Log order details to file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "team": event.team_name,
            "match": event.match_name,
            "token_id": event.token_id,
            "order_size": event.order_size,
            "discord_user": str(message.author),
            "discord_message": message.content,
            "success": success,
            "order_id": order_id,
            "error": error,
            "test_mode": test_mode,
        }

        # Add extra fields if provided
        if extra:
            log_entry.update(extra)

        # Append to log file
        try:
            with open(config.LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")


def run_bot():
    """Run the Discord bot."""
    # Validate configuration
    if config.DISCORD_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("ERROR: Please set DISCORD_BOT_TOKEN in config.py or environment variable")
        return

    bot = TradingBot()

    # Check if any channels are configured
    if not bot.parser.active_channels:
        print("WARNING: No channels configured in matches.json")
        print("The bot will run but won't respond to any messages until you configure matches.")

    try:
        bot.run(config.DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        print("ERROR: Invalid bot token")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    run_bot()
