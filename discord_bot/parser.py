"""
Message Parser for Discord Trading Bot

Parses messages like "goal milan" and returns the corresponding
Polymarket token_id for order execution.
"""

import json
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from . import config


@dataclass
class GoalEvent:
    """Represents a parsed goal event."""
    team_name: str
    token_id: str
    order_size: int
    match_name: str
    draw_token_id: Optional[str] = None
    draw_order_size: int = 10


class MessageParser:
    """
    Parses Discord messages and matches them to Polymarket tokens.
    Each channel is mapped to a specific match.

    Usage:
        parser = MessageParser()
        event = parser.parse("goal milan", channel_id=123456789)
        if event:
            print(f"Goal by {event.team_name}, token: {event.token_id}")
    """

    def __init__(self, matches_file: str = None):
        """
        Initialize parser with matches configuration.

        Args:
            matches_file: Path to matches.json. Defaults to config.MATCHES_FILE
        """
        self.matches_file = matches_file or config.MATCHES_FILE
        self.matches_data = {}
        # channel_id -> {alias -> team_info}
        self.channel_lookup = {}
        # Set of active channel IDs
        self.active_channels = set()

        self.reload_matches()

    def reload_matches(self) -> bool:
        """
        Reload matches configuration from JSON file.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.matches_file):
            print(f"Warning: Matches file not found: {self.matches_file}")
            return False

        try:
            with open(self.matches_file, "r", encoding="utf-8") as f:
                self.matches_data = json.load(f)

            self._build_channel_lookup()
            return True

        except json.JSONDecodeError as e:
            print(f"Error parsing matches.json: {e}")
            return False
        except Exception as e:
            print(f"Error loading matches: {e}")
            return False

    def _build_channel_lookup(self):
        """Build a lookup table from channel_id + team alias to token info."""
        self.channel_lookup = {}
        self.active_channels = set()

        default_order_size = self.matches_data.get("default_order_size", config.DEFAULT_ORDER_SIZE)

        for match in self.matches_data.get("active_matches", []):
            match_name = match.get("name", "Unknown Match")
            match_order_size = match.get("order_size", default_order_size)
            channel_id_str = match.get("channel_id", "")

            # Skip matches without channel_id
            if not channel_id_str or channel_id_str == "REPLACE_WITH_CHANNEL_ID":
                continue

            try:
                channel_id = int(channel_id_str)
            except ValueError:
                print(f"Invalid channel_id for match {match_name}: {channel_id_str}")
                continue

            self.active_channels.add(channel_id)

            # Initialize channel lookup if needed
            if channel_id not in self.channel_lookup:
                self.channel_lookup[channel_id] = {}

            # Get draw token info for this match
            draw_token_id = match.get("draw_token_id")
            if draw_token_id and draw_token_id.startswith("REPLACE_"):
                draw_token_id = None
            draw_order_size = match.get("draw_order_size", match_order_size)

            teams = match.get("teams", {})
            for team_key, team_info in teams.items():
                token_id = team_info.get("token_id")
                aliases = team_info.get("aliases", [team_key])
                team_order_size = team_info.get("order_size", match_order_size)

                if not token_id or token_id.startswith("REPLACE_"):
                    continue

                # Add all aliases (lowercase) to channel lookup
                for alias in aliases:
                    alias_lower = alias.lower().strip()
                    self.channel_lookup[channel_id][alias_lower] = {
                        "token_id": token_id,
                        "order_size": team_order_size,
                        "match_name": match_name,
                        "team_name": team_key,
                        "draw_token_id": draw_token_id,
                        "draw_order_size": draw_order_size,
                    }

        if config.VERBOSE:
            print(f"Loaded {len(self.active_channels)} active channels")
            for ch_id, teams in self.channel_lookup.items():
                print(f"  Channel {ch_id}: {len(teams)} team aliases")

    def is_active_channel(self, channel_id: int) -> bool:
        """Check if a channel is configured for a match."""
        return channel_id in self.active_channels

    def parse(self, message: str, channel_id: int = None) -> Optional[GoalEvent]:
        """
        Parse a message and return GoalEvent if it's a valid goal message.

        Args:
            message: The Discord message content
            channel_id: The Discord channel ID where message was sent

        Returns:
            GoalEvent if message is "goal [team]" and team is found, None otherwise
        """
        if not message:
            return None

        # Normalize message
        message = message.lower().strip()

        # Check if message starts with "goal "
        if not message.startswith("goal "):
            return None

        # Extract team name (everything after "goal ")
        team_query = message[5:].strip()

        if not team_query:
            return None

        # Look up team in the specific channel
        if channel_id is None or channel_id not in self.channel_lookup:
            if config.VERBOSE:
                print(f"Channel {channel_id} not configured for any match")
            return None

        team_info = self.channel_lookup[channel_id].get(team_query)

        if not team_info:
            if config.VERBOSE:
                print(f"Unknown team '{team_query}' in channel {channel_id}")
            return None

        return GoalEvent(
            team_name=team_info["team_name"],
            token_id=team_info["token_id"],
            order_size=team_info["order_size"],
            match_name=team_info["match_name"],
            draw_token_id=team_info.get("draw_token_id"),
            draw_order_size=team_info.get("draw_order_size", 10),
        )

    def get_active_teams(self, channel_id: int = None) -> list:
        """Get list of all active team aliases for a channel."""
        if channel_id and channel_id in self.channel_lookup:
            return list(self.channel_lookup[channel_id].keys())
        # Return all teams across all channels
        all_teams = set()
        for teams in self.channel_lookup.values():
            all_teams.update(teams.keys())
        return list(all_teams)

    def get_matches_summary(self) -> str:
        """Get a summary of active matches for logging."""
        matches = self.matches_data.get("active_matches", [])
        if not matches:
            return "No active matches configured"

        lines = ["Active matches:"]
        for match in matches:
            channel_id = match.get("channel_id", "NOT SET")
            kick_off = match.get("kick_off", "")

            # Format kick-off time if available
            kick_off_str = ""
            if kick_off and kick_off != "Unknown":
                try:
                    # Parse and format nicely
                    from datetime import datetime
                    time_str = kick_off.replace("+00", "+00:00")
                    if " " in time_str and "T" not in time_str:
                        time_str = time_str.replace(" ", "T")
                    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    kick_off_str = f" | Kick-off: {dt.strftime('%H:%M')} UTC"
                except:
                    kick_off_str = f" | Kick-off: {kick_off}"

            lines.append(f"  - {match.get('name', 'Unknown')} (channel: {channel_id}){kick_off_str}")
            for team_key, team_info in match.get("teams", {}).items():
                aliases = ", ".join(team_info.get("aliases", [team_key]))
                lines.append(f"      {team_key}: [{aliases}]")

        return "\n".join(lines)


# Convenience function for quick testing
def test_parser():
    """Test the parser with sample messages."""
    parser = MessageParser()

    print(parser.get_matches_summary())
    print()

    test_messages = [
        "goal milan",
        "GOAL Milan",
        "Goal JUVE",
        "goal juventus",
        "goal ac milan",
        "hello world",
        "goal unknown_team",
        "",
    ]

    for msg in test_messages:
        result = parser.parse(msg)
        if result:
            print(f"'{msg}' -> {result.team_name} (token: {result.token_id[:20]}...)")
        else:
            print(f"'{msg}' -> No match")


if __name__ == "__main__":
    test_parser()
