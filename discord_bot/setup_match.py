"""
Match Setup Script

Automatically configures matches.json from a Polymarket URL.

Usage:
    python -m discord_bot.setup_match

Or directly:
    python discord_bot/setup_match.py
"""

import json
import os
import re
import sys
import requests
from typing import Optional, Dict, List, Tuple


# Gamma API endpoint (Polymarket's public API)
GAMMA_API = "https://gamma-api.polymarket.com"

# Path to matches.json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MATCHES_FILE = os.path.join(SCRIPT_DIR, "matches.json")


def extract_slug_from_url(url: str) -> Optional[str]:
    """Extract event slug from Polymarket URL."""
    # Handle various URL formats:
    # https://polymarket.com/event/epl-che-bou-2025-12-30
    # https://polymarket.com/event/epl-che-bou-2025-12-30?tid=...

    patterns = [
        r'polymarket\.com/event/([a-zA-Z0-9\-]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def fetch_event_data(slug: str) -> Optional[Dict]:
    """Fetch event data from Gamma API."""
    try:
        # Try fetching by slug
        response = requests.get(f"{GAMMA_API}/events?slug={slug}")
        response.raise_for_status()
        events = response.json()

        if events and len(events) > 0:
            return events[0]

        # Try searching by slug as title
        response = requests.get(f"{GAMMA_API}/events?title={slug}")
        response.raise_for_status()
        events = response.json()

        if events and len(events) > 0:
            return events[0]

        return None

    except Exception as e:
        print(f"Error fetching event data: {e}")
        return None


def parse_markets(event: Dict) -> Tuple[Dict, Dict, Optional[Dict]]:
    """
    Parse markets from event data to extract team win tokens and draw token.

    Note: This function is deprecated - use find_tokens_from_markets instead.

    Returns:
        Tuple of (team1_info, team2_info, draw_info)
        Each info dict contains: name, token_id, aliases
    """
    markets = event.get("markets", [])

    team1_info = None
    team2_info = None
    draw_info = None

    for market in markets:
        question = market.get("question", "").lower()
        outcomes = market.get("outcomes", [])
        token_ids_raw = market.get("clobTokenIds", [])

        # Parse token IDs (may be JSON string or list)
        if isinstance(token_ids_raw, str):
            try:
                token_ids = json.loads(token_ids_raw)
            except:
                token_ids = []
        else:
            token_ids = token_ids_raw if token_ids_raw else []

        # Skip if no token IDs
        if not token_ids:
            continue

        # Look for moneyline/winner market
        # Common patterns: "Will X win?", "X to win", "Winner: X"
        if any(keyword in question for keyword in ["will", "win", "winner", "moneyline"]):
            # Check for team markets (single outcome per market)
            if len(outcomes) == 2 and len(token_ids) == 2:
                # This is likely a Yes/No market for a specific team
                # The first token is usually YES
                outcome_yes = outcomes[0] if outcomes else "Yes"

                # Try to extract team name from question
                # E.g., "Will Chelsea win?" -> "Chelsea"
                team_match = re.search(r'will\s+(.+?)\s+win', question)
                if team_match:
                    team_name = team_match.group(1).strip()
                    team_name_clean = team_name.lower().replace(" fc", "").replace(" afc", "").strip()

                    info = {
                        "name": team_name_clean,
                        "token_id": token_ids[0],  # YES token
                        "display_name": team_name.title(),
                    }

                    if team1_info is None:
                        team1_info = info
                    elif team2_info is None:
                        team2_info = info

        # Look for draw market
        if "draw" in question:
            if len(token_ids) >= 1:
                draw_info = {
                    "name": "draw",
                    "token_id": token_ids[0],  # YES token for draw
                }

    return team1_info, team2_info, draw_info


def fetch_all_markets_for_event(slug: str) -> List[Dict]:
    """Fetch all markets for an event using different API endpoints."""
    all_markets = []

    try:
        # Try the events endpoint
        response = requests.get(f"{GAMMA_API}/events?slug={slug}")
        if response.ok:
            events = response.json()
            if events:
                all_markets.extend(events[0].get("markets", []))

        # Also try markets endpoint directly
        response = requests.get(f"{GAMMA_API}/markets?slug={slug}")
        if response.ok:
            markets = response.json()
            if isinstance(markets, list):
                all_markets.extend(markets)

    except Exception as e:
        print(f"Warning: Error fetching additional markets: {e}")

    return all_markets


def parse_token_ids(token_ids_raw) -> List[str]:
    """Parse token IDs which may be a list or a JSON string."""
    if token_ids_raw is None:
        return []

    # If it's already a list, return it
    if isinstance(token_ids_raw, list):
        return token_ids_raw

    # If it's a string, try to parse as JSON
    if isinstance(token_ids_raw, str):
        try:
            parsed = json.loads(token_ids_raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return []


def extract_game_start_time(markets: List[Dict]) -> Optional[str]:
    """Extract game start time from markets data."""
    for market in markets:
        game_start = market.get("gameStartTime")
        if game_start:
            return game_start
    return None


def fetch_market_by_slug(slug: str) -> Optional[Dict]:
    """
    Fetch a single market by its slug.

    Returns:
        Market dict or None if not found
    """
    try:
        response = requests.get(f"{GAMMA_API}/markets?slug={slug}")
        if response.ok:
            markets = response.json()
            if markets and len(markets) > 0:
                return markets[0]
    except Exception as e:
        print(f"  Warning: Could not fetch market {slug}: {e}")
    return None


def find_ou_tokens_by_slug(base_slug: str) -> Dict[str, str]:
    """
    Find Over/Under tokens by constructing slugs from the base event slug.

    Polymarket uses predictable slug patterns:
    - O/U 1.5: {base_slug}-total-1pt5
    - O/U 2.5: {base_slug}-total-2pt5

    Returns:
        Dict with keys: over_2_5, under_2_5, over_1_5, under_1_5
    """
    ou_tokens = {
        "over_2_5": "",
        "under_2_5": "",
        "over_1_5": "",
        "under_1_5": "",
    }

    # Try O/U 2.5
    market = fetch_market_by_slug(f"{base_slug}-total-2pt5")
    if market:
        token_ids = parse_token_ids(market.get("clobTokenIds", []))
        if len(token_ids) >= 2:
            ou_tokens["over_2_5"] = token_ids[0]
            ou_tokens["under_2_5"] = token_ids[1]

    # Try O/U 1.5
    market = fetch_market_by_slug(f"{base_slug}-total-1pt5")
    if market:
        token_ids = parse_token_ids(market.get("clobTokenIds", []))
        if len(token_ids) >= 2:
            ou_tokens["over_1_5"] = token_ids[0]
            ou_tokens["under_1_5"] = token_ids[1]

    return ou_tokens


def find_handicap_tokens_by_slug(base_slug: str) -> Dict[str, str]:
    """
    Find handicap tokens by constructing slugs from the base event slug.

    Polymarket uses predictable slug patterns:
    - Home -1.5: {base_slug}-spread-home-1pt5

    The spread market resolves to home team if they win by 2+, otherwise away.
    So the first token is home -1.5, second is away +1.5.

    Returns:
        Dict with keys: home_minus_1_5, home_plus_1_5, away_minus_1_5, away_plus_1_5
    """
    handicap_tokens = {
        "home_minus_1_5": "",
        "home_plus_1_5": "",
        "away_minus_1_5": "",
        "away_plus_1_5": "",
    }

    # The spread market contains both sides
    # First token = home team covers -1.5 (wins by 2+)
    # Second token = away team covers +1.5 (doesn't lose by 2+)
    market = fetch_market_by_slug(f"{base_slug}-spread-home-1pt5")
    if market:
        token_ids = parse_token_ids(market.get("clobTokenIds", []))
        if len(token_ids) >= 2:
            handicap_tokens["home_minus_1_5"] = token_ids[0]
            handicap_tokens["away_plus_1_5"] = token_ids[1]

    return handicap_tokens


def find_ou_tokens(markets: List[Dict]) -> Dict[str, str]:
    """
    Find Over/Under tokens from markets (fallback method).

    Looks for patterns like:
    - "Will there be over 2.5 goals?"
    - "Over 2.5 goals?"
    - "Will the match have over 1.5 goals?"

    Returns:
        Dict with keys: over_2_5, under_2_5, over_1_5, under_1_5
    """
    ou_tokens = {
        "over_2_5": "",
        "under_2_5": "",
        "over_1_5": "",
        "under_1_5": "",
    }

    for market in markets:
        question = market.get("question", "")
        question_lower = question.lower()
        token_ids = parse_token_ids(market.get("clobTokenIds", []))

        if not token_ids:
            continue

        # Look for O/U 2.5 patterns
        if re.search(r'over\s*2\.?5', question_lower) or re.search(r'2\.?5\s*goals', question_lower) or "o/u 2.5" in question_lower:
            # First token is usually "Yes" (Over), second is "No" (Under)
            if len(token_ids) >= 1 and not ou_tokens["over_2_5"]:
                ou_tokens["over_2_5"] = token_ids[0]
            if len(token_ids) >= 2 and not ou_tokens["under_2_5"]:
                ou_tokens["under_2_5"] = token_ids[1]

        # Look for O/U 1.5 patterns
        if re.search(r'over\s*1\.?5', question_lower) or re.search(r'1\.?5\s*goals', question_lower) or "o/u 1.5" in question_lower:
            if len(token_ids) >= 1 and not ou_tokens["over_1_5"]:
                ou_tokens["over_1_5"] = token_ids[0]
            if len(token_ids) >= 2 and not ou_tokens["under_1_5"]:
                ou_tokens["under_1_5"] = token_ids[1]

    return ou_tokens


def find_handicap_tokens(markets: List[Dict], team1_name: str, team2_name: str) -> Dict[str, str]:
    """
    Find handicap tokens from markets (fallback method).

    Looks for patterns like:
    - "Will Chelsea win with handicap -1.5?"
    - "Chelsea -1.5"
    - "Will Bournemouth cover +1.5?"

    Args:
        markets: List of market data
        team1_name: Home team name (cleaned)
        team2_name: Away team name (cleaned)

    Returns:
        Dict with keys: home_minus_1_5, home_plus_1_5, away_minus_1_5, away_plus_1_5
    """
    handicap_tokens = {
        "home_minus_1_5": "",
        "home_plus_1_5": "",
        "away_minus_1_5": "",
        "away_plus_1_5": "",
    }

    team1_lower = team1_name.lower()
    team2_lower = team2_name.lower()

    for market in markets:
        question = market.get("question", "")
        question_lower = question.lower()
        token_ids = parse_token_ids(market.get("clobTokenIds", []))

        if not token_ids:
            continue

        # Check if this is a handicap/spread market
        is_handicap = any(kw in question_lower for kw in ["handicap", "-1.5", "+1.5", "cover", "spread"])
        if not is_handicap:
            continue

        # Check which team and which handicap
        has_team1 = team1_lower in question_lower
        has_team2 = team2_lower in question_lower

        has_minus = "-1.5" in question or "minus 1.5" in question_lower
        has_plus = "+1.5" in question or "plus 1.5" in question_lower

        # First token is YES (team covers the spread)
        if has_team1 and has_minus and not handicap_tokens["home_minus_1_5"]:
            handicap_tokens["home_minus_1_5"] = token_ids[0]
        elif has_team1 and has_plus and not handicap_tokens["home_plus_1_5"]:
            handicap_tokens["home_plus_1_5"] = token_ids[0]
        elif has_team2 and has_minus and not handicap_tokens["away_minus_1_5"]:
            handicap_tokens["away_minus_1_5"] = token_ids[0]
        elif has_team2 and has_plus and not handicap_tokens["away_plus_1_5"]:
            handicap_tokens["away_plus_1_5"] = token_ids[0]

    return handicap_tokens


def find_tokens_from_markets(markets: List[Dict], event_title: str = "") -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Find team win tokens and draw token from a list of markets.

    Returns:
        Tuple of (team1_info, team2_info, draw_info)
    """
    team1_info = None
    team2_info = None
    draw_info = None

    # Try to extract team names from event title
    # E.g., "Chelsea FC vs. AFC Bournemouth" -> ["Chelsea", "Bournemouth"]
    title_lower = event_title.lower()

    for market in markets:
        question = market.get("question", "")
        question_lower = question.lower()
        outcomes = market.get("outcomes", [])
        token_ids_raw = market.get("clobTokenIds", [])

        # Parse token IDs (may be JSON string or list)
        token_ids = parse_token_ids(token_ids_raw)

        if not token_ids:
            continue

        # Look for "Will X win?" pattern
        win_match = re.search(r'will\s+(.+?)\s+win', question_lower)
        if win_match:
            team_name = win_match.group(1).strip()
            # Clean up team name
            team_name_clean = re.sub(r'\s*(fc|afc|cf|sc)\s*', '', team_name, flags=re.IGNORECASE).strip()

            info = {
                "name": team_name_clean,
                "token_id": token_ids[0],
                "display_name": team_name.title().replace(" Fc", " FC").replace(" Afc", " AFC"),
            }

            if team1_info is None:
                team1_info = info
            elif team2_info is None and info["name"] != team1_info["name"]:
                team2_info = info

        # Look for draw market
        if "draw" in question_lower and "will" in question_lower:
            draw_info = {
                "name": "draw",
                "token_id": token_ids[0],
            }

    return team1_info, team2_info, draw_info


def generate_aliases(team_name: str) -> List[str]:
    """Generate common aliases for a team name."""
    aliases = [team_name.lower()]

    # Add common variations
    name_lower = team_name.lower()

    # Remove common suffixes and add as alias
    for suffix in [" fc", " afc", " cf", " sc"]:
        if name_lower.endswith(suffix):
            aliases.append(name_lower.replace(suffix, "").strip())

    # Add first 3 letters as abbreviation
    if len(name_lower) >= 3:
        abbrev = name_lower[:3]
        if abbrev not in aliases:
            aliases.append(abbrev)

    # Common team abbreviations
    abbreviations = {
        "chelsea": ["che", "blues"],
        "bournemouth": ["bou", "cherries"],
        "arsenal": ["ars", "gunners"],
        "liverpool": ["liv", "reds"],
        "manchester united": ["manu", "man utd", "united"],
        "manchester city": ["mancity", "man city", "city"],
        "tottenham": ["tot", "spurs"],
        "juventus": ["juve", "jfc"],
        "milan": ["acm", "ac milan"],
        "inter": ["inter milan", "internazionale"],
        "napoli": ["nap", "ssc napoli"],
        "roma": ["rom", "as roma"],
        "barcelona": ["barca", "fcb"],
        "real madrid": ["real", "madrid", "rma"],
    }

    for key, abbrevs in abbreviations.items():
        if key in name_lower:
            for a in abbrevs:
                if a not in aliases:
                    aliases.append(a)

    return aliases


def load_matches() -> Dict:
    """Load existing matches.json or create default structure."""
    if os.path.exists(MATCHES_FILE):
        try:
            with open(MATCHES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass

    return {
        "description": "Match configuration. Update before each matchday.",
        "default_order_size": 10,
        "active_matches": []
    }


def save_matches(data: Dict):
    """Save matches.json."""
    with open(MATCHES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nSaved to: {MATCHES_FILE}")


def setup_match():
    """Interactive match setup."""
    print("=" * 60)
    print("MATCH SETUP SCRIPT")
    print("=" * 60)
    print()

    # Get Polymarket URL
    url = input("Enter Polymarket event URL: ").strip()
    if not url:
        print("Error: URL is required")
        return

    # Extract slug
    slug = extract_slug_from_url(url)
    if not slug:
        print(f"Error: Could not extract event slug from URL: {url}")
        return

    print(f"\nEvent slug: {slug}")
    print("Fetching event data from Polymarket...")

    # Fetch event data
    event = fetch_event_data(slug)
    if not event:
        print("Error: Could not fetch event data")
        return

    event_title = event.get("title", "Unknown Match")
    print(f"Found event: {event_title}")

    # Get all markets
    markets = event.get("markets", [])
    print(f"Found {len(markets)} markets")

    # Find tokens
    team1, team2, draw = find_tokens_from_markets(markets, event_title)

    # Find O/U and handicap tokens using slug-based fetching (primary method)
    print("Fetching O/U and spread markets...")
    ou_tokens = find_ou_tokens_by_slug(slug)
    handicap_tokens = find_handicap_tokens_by_slug(slug)

    # If slug-based didn't find tokens, try pattern matching in event markets (fallback)
    if not any(ou_tokens.values()):
        ou_tokens = find_ou_tokens(markets)
    if team1 and team2 and not any(handicap_tokens.values()):
        handicap_tokens = find_handicap_tokens(markets, team1["name"], team2["name"])

    # Extract game start time
    game_start_time = extract_game_start_time(markets)

    # Display found tokens
    print("\n" + "-" * 40)
    print("DETECTED TOKENS:")
    print("-" * 40)

    if game_start_time:
        # Parse and format the time nicely
        try:
            from datetime import datetime
            # Handle format like "2025-12-30 19:30:00+00"
            time_str = game_start_time.replace("+00", "+00:00")
            if " " in time_str and "T" not in time_str:
                time_str = time_str.replace(" ", "T")
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            print(f"KICK-OFF: {dt.strftime('%Y-%m-%d %H:%M')} UTC")
            # Also show local time (assuming UTC+1 for Italy)
            local_hour = (dt.hour + 1) % 24
            print(f"          {dt.strftime('%Y-%m-%d')} {local_hour:02d}:{dt.strftime('%M')} (Italy)")
        except:
            print(f"KICK-OFF: {game_start_time}")
    else:
        print("KICK-OFF: Not available")

    print()

    if team1:
        print(f"Team 1: {team1['display_name']}")
        print(f"  Token ID: {team1['token_id'][:30]}...")
    else:
        print("Team 1: NOT FOUND")

    if team2:
        print(f"Team 2: {team2['display_name']}")
        print(f"  Token ID: {team2['token_id'][:30]}...")
    else:
        print("Team 2: NOT FOUND")

    if draw:
        print(f"Draw: YES")
        print(f"  Token ID: {draw['token_id'][:30]}...")
    else:
        print("Draw: NOT FOUND")

    # Display O/U tokens
    print()
    print("OVER/UNDER 2.5:")
    if ou_tokens.get("over_2_5"):
        print(f"  Over 2.5:  {ou_tokens['over_2_5'][:30]}...")
    else:
        print("  Over 2.5:  NOT FOUND")
    if ou_tokens.get("under_2_5"):
        print(f"  Under 2.5: {ou_tokens['under_2_5'][:30]}...")
    else:
        print("  Under 2.5: NOT FOUND")

    print()
    print("OVER/UNDER 1.5:")
    if ou_tokens.get("over_1_5"):
        print(f"  Over 1.5:  {ou_tokens['over_1_5'][:30]}...")
    else:
        print("  Over 1.5:  NOT FOUND")
    if ou_tokens.get("under_1_5"):
        print(f"  Under 1.5: {ou_tokens['under_1_5'][:30]}...")
    else:
        print("  Under 1.5: NOT FOUND")

    # Display handicap tokens
    print()
    team1_label = team1['name'] if team1 else "Home"
    team2_label = team2['name'] if team2 else "Away"
    print(f"HANDICAP ({team1_label.upper()} / {team2_label.upper()}):")
    if handicap_tokens.get("home_minus_1_5"):
        print(f"  {team1_label.title()} -1.5: {handicap_tokens['home_minus_1_5'][:30]}...")
    else:
        print(f"  {team1_label.title()} -1.5: NOT FOUND")
    if handicap_tokens.get("home_plus_1_5"):
        print(f"  {team1_label.title()} +1.5: {handicap_tokens['home_plus_1_5'][:30]}...")
    else:
        print(f"  {team1_label.title()} +1.5: NOT FOUND")
    if handicap_tokens.get("away_minus_1_5"):
        print(f"  {team2_label.title()} -1.5: {handicap_tokens['away_minus_1_5'][:30]}...")
    else:
        print(f"  {team2_label.title()} -1.5: NOT FOUND")
    if handicap_tokens.get("away_plus_1_5"):
        print(f"  {team2_label.title()} +1.5: {handicap_tokens['away_plus_1_5'][:30]}...")
    else:
        print(f"  {team2_label.title()} +1.5: NOT FOUND")

    # Check if we have enough data
    if not team1 or not team2:
        print("\nError: Could not find both team tokens.")
        print("You may need to manually add token IDs to matches.json")

        # Show available markets for debugging
        print("\nAvailable markets:")
        for m in markets[:10]:
            print(f"  - {m.get('question', 'Unknown')}")
            if m.get('clobTokenIds'):
                print(f"    Token IDs: {m.get('clobTokenIds')}")
        return

    # Get Discord channel ID
    print("\n" + "-" * 40)
    channel_id = input("Enter Discord Channel ID: ").strip()
    if not channel_id:
        print("Error: Channel ID is required")
        return

    # Validate it's a number
    try:
        int(channel_id)
    except ValueError:
        print("Error: Channel ID must be a number")
        return

    # Get order size (optional)
    order_size_input = input("Enter order size per team (default: 10): ").strip()
    order_size = int(order_size_input) if order_size_input else 10

    # Get budget per market for Poisson model
    budget_input = input("Enter budget per market in USD (default: 500): ").strip()
    budget_per_market = float(budget_input) if budget_input else 500.0

    # Extract date from slug if possible
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', slug)
    match_date = date_match.group(1) if date_match else "YYYY-MM-DD"

    # Convert UTC kick-off to CET (UTC+1)
    kick_off_cet = None
    if game_start_time:
        try:
            from datetime import datetime, timedelta
            # Parse the UTC time
            time_str = game_start_time.replace("+00", "+00:00")
            if " " in time_str and "T" not in time_str:
                time_str = time_str.replace(" ", "T")
            dt_utc = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            # Convert to CET (add 1 hour)
            dt_cet = dt_utc + timedelta(hours=1)
            kick_off_cet = dt_cet.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Warning: Could not convert to CET: {e}")

    # Build match config
    match_config = {
        "name": event_title,
        "date": match_date,
        "kick_off": game_start_time or "Unknown",
        "kick_off_cet": kick_off_cet or "Unknown",
        "channel_id": channel_id,
        "polymarket_url": url,
        "draw_token_id": draw["token_id"] if draw else "REPLACE_WITH_DRAW_TOKEN_ID",
        "draw_order_size": order_size,
        "budget_per_market": budget_per_market,
        "teams": {
            team1["name"]: {
                "token_id": team1["token_id"],
                "aliases": generate_aliases(team1["name"]),
                "order_size": order_size
            },
            team2["name"]: {
                "token_id": team2["token_id"],
                "aliases": generate_aliases(team2["name"]),
                "order_size": order_size
            }
        },
        # Over/Under tokens
        "over_2_5_token": ou_tokens.get("over_2_5", ""),
        "under_2_5_token": ou_tokens.get("under_2_5", ""),
        "over_1_5_token": ou_tokens.get("over_1_5", ""),
        "under_1_5_token": ou_tokens.get("under_1_5", ""),
        # Handicap tokens (home team perspective)
        "home_minus_1_5_token": handicap_tokens.get("home_minus_1_5", ""),
        "home_plus_1_5_token": handicap_tokens.get("home_plus_1_5", ""),
        "away_minus_1_5_token": handicap_tokens.get("away_minus_1_5", ""),
        "away_plus_1_5_token": handicap_tokens.get("away_plus_1_5", ""),
    }

    # Load existing matches
    matches_data = load_matches()

    # Check if match already exists (by channel_id or name)
    existing_idx = None
    for i, m in enumerate(matches_data.get("active_matches", [])):
        if m.get("channel_id") == channel_id or m.get("name") == event_title:
            existing_idx = i
            break

    if existing_idx is not None:
        print(f"\nMatch already exists at index {existing_idx}. Replacing...")
        matches_data["active_matches"][existing_idx] = match_config
    else:
        matches_data["active_matches"].append(match_config)

    # Preview
    print("\n" + "=" * 60)
    print("MATCH CONFIGURATION PREVIEW:")
    print("=" * 60)
    print(json.dumps(match_config, indent=2))

    # Confirm save
    confirm = input("\nSave this configuration? (y/n): ").strip().lower()
    if confirm == "y":
        save_matches(matches_data)
        print("\nSetup complete! Restart the bot to load the new match.")
        print("\nNext steps:")
        print("1. Restart the bot: python -m discord_bot.bot")
        print("2. Send 'goal <team>' in the Discord channel to trigger bets")
    else:
        print("Cancelled. No changes saved.")


if __name__ == "__main__":
    setup_match()
