#!/bin/bash
# Wrapper for cron: cron jobs run with a bare environment, so the
# TEAMS_* variables you `export` in an interactive terminal are invisible
# to them. This sources a plain env file instead -- keep your real values
# there once you have them, not in the crontab line itself (crontab -l
# would leak the client secret in plain sight otherwise).
#
# Scheduled to fire once a day, shortly after the recurring channel meeting
# ends (Mon-Fri, meeting ends ~11:00 AM IST, checked at 11:20 AM IST -- this
# Mac's system clock is already IST, confirmed via `date +%Z`, so no
# timezone conversion needed in the cron line itself):
#   crontab -l
#   20 11 * * 1-5 /Users/sudalaiyandi/Documents/agent/run_auto_mom.sh >> ...
# This replaced an earlier every-15-minutes poll -- checking once, right
# after the one meeting this is scoped to should have ended, gets the same
# result with 96x fewer no-op runs per day. If the meeting's day/time ever
# changes, update the "20 11 * * 1-5" fields to match (minute hour * * dow,
# dow 0=Sunday..6=Saturday).
#
# One-time setup:
#   1. Create ~/.agent_teams_env with the shared app credentials (same
#      Entra app can serve multiple meetings):
#        export TEAMS_TENANT_ID="..."
#        export TEAMS_CLIENT_ID="..."
#        export TEAMS_CLIENT_SECRET="..."
#      Then: chmod 600 ~/.agent_teams_env (keep it readable only by you).
#   2. Copy meetings_config.example.json to meetings_config.json in this
#      repo and fill in each meeting's organizer/join URL/webhook (see
#      auto_mom.py's docstring). Add one meeting per recurring call.

ENV_FILE="$HOME/.agent_teams_env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

cd /Users/sudalaiyandi/Documents/agent || exit 1
/opt/homebrew/bin/python3 auto_mom.py
