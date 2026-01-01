# Polymarket Football Bot

Discord bot for trading football matches on Polymarket. Uses a Poisson model to predict how odds shift when goals are scored.

## What it does

When a goal happens, the bot figures out which bets become more valuable and places orders automatically. The math is based on the fact that goals in football roughly follow a Poisson distribution - so you can predict what the new odds *should* be after a goal, and buy before the market catches up.

## Structure

```
discord_bot/
    bot.py              - main bot, handles discord events
    poisson_model.py    - the probability math
    match_runner.py     - fetches prices in background
    position_manager.py - tracks open positions
    parser.py           - parses messages
    config.py           - settings

utilities/
    polymarket_exchange.py - polymarket API wrapper
```

## The model

Football scoring is modeled as two independent Poisson processes. The difference between two Poissons follows a Skellam distribution, which gives us P(home win), P(draw), P(away win).

Given current market prices, we calibrate λ_home and λ_away (expected remaining goals), then compute what happens to all probabilities if either team scores. If the predicted probability jumps enough, we buy.

## Running it

```bash
pip install -r requirements.txt

# set your discord token
export DISCORD_BOT_TOKEN="..."

# configure matches in discord_bot/matches.json
# (see matches.example.json)

python -m discord_bot.bot
```

You'll also need a `secret.json` with your Polymarket credentials.

## Notes

- Prices are fetched every minute during live matches
- Buy decisions are pre-computed so execution is fast when goals happen
- The bot uses discord buttons for goal input (more reliable than parsing messages)

---

*For educational purposes. Prediction markets carry risk.*
