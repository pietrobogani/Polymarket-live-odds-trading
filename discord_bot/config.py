"""
Discord Bot Configuration

Configure your bot settings here or via environment variables.
"""

import os

# =============================================================================
# DISCORD CONFIGURATION
# =============================================================================

# Your Discord bot token (from https://discord.com/developers/applications)
# Set via environment variable for security - never commit tokens to git
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")

# Channel ID where the bot listens for goal messages
# Right-click the channel in Discord -> Copy ID (need Developer Mode enabled)
DISCORD_CHANNEL_ID = int(os.environ.get("DISCORD_CHANNEL_ID", "0"))

# Master channel for bot status updates (optional)
# Set to None to disable status updates
MASTER_CHANNEL_ID = os.environ.get("MASTER_CHANNEL_ID", None)
if MASTER_CHANNEL_ID:
    MASTER_CHANNEL_ID = int(MASTER_CHANNEL_ID)

# =============================================================================
# FILE PATHS
# =============================================================================

# Base directory (discord_bot folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to matches.json (weekly match configuration)
MATCHES_FILE = os.path.join(BASE_DIR, "matches.json")

# Path to secret.json (Polymarket credentials)
SECRET_FILE = os.path.join(BASE_DIR, "..", "secret.json")

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

# Default order size if not specified in matches.json
DEFAULT_ORDER_SIZE = 10

# Enable/disable actual order execution (set False for testing)
EXECUTE_ORDERS = False

# =============================================================================
# LOGGING
# =============================================================================

# Log file for order history
LOG_FILE = os.path.join(BASE_DIR, "orders.log")

# Enable verbose logging
VERBOSE = True
