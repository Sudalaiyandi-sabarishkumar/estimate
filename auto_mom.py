#!/usr/bin/env python3
"""Unattended MoM pipeline: Teams channel meeting transcript -> MoM -> posted
back to the channel, with no human running a command.

This is the "full end-to-end auto-pull" half of the mom feature -- the
Phase 1 manual path (`/mom <file>` in task_breakdown_chat.py) keeps working
unchanged regardless of this. THIS SCRIPT CANNOT DO ANYTHING until your
tenant admin completes the three steps in teams_graph.py's docstring --
run it anyway and it'll tell you exactly which one is still missing rather
than failing silently.

Meant to be run on a schedule (cron/launchd), not as a long-lived daemon --
each run polls once for transcripts it hasn't processed yet and exits.
Nothing here reacts in real time to a call ending; the interval you schedule
this at is the real-world latency between "call ends" and "MoM posted"
(Teams itself also takes some minutes to finish processing a transcript
after the call ends, so there's no point polling faster than every ~10-15
min).

Setup (all one-time, in addition to the tenant-admin steps in
teams_graph.py):
  export TEAMS_TENANT_ID="..."          # Entra ID tenant (directory) ID
  export TEAMS_CLIENT_ID="..."          # the registered app's client ID
  export TEAMS_CLIENT_SECRET="..."      # a client secret for that app
  export TEAMS_ORGANIZER_USER_ID="..."  # Entra object ID of the user who
                                         # organizes the recurring channel
                                         # meeting (the access policy in
                                         # teams_graph.py is granted to this
                                         # same user)
  export TEAMS_MEETING_JOIN_URL="..."   # that meeting's Join URL, copied
                                         # once from its calendar invite
  export TEAMS_WEBHOOK_URL="..."        # same webhook /mom already uses to
                                         # post -- see teams_notify.py

Then schedule it, e.g. via cron:
  */15 * * * * cd /path/to/agent && /usr/bin/python3 auto_mom.py >> auto_mom.log 2>&1

Run it once by hand first (no cron) to confirm it actually posts before
scheduling it unattended.
"""

import json
import os
import sys
from datetime import datetime, timezone

from teams_graph import (get_app_token, resolve_meeting_id,
                          list_new_transcripts, get_transcript_vtt)
from teams_notify import send_to_teams
from task_breakdown_chat import run_mom_pipeline

STATE_FILE = "auto_mom_state.json"
REQUIRED_ENV = [
    "TEAMS_TENANT_ID", "TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET",
    "TEAMS_ORGANIZER_USER_ID", "TEAMS_MEETING_JOIN_URL", "TEAMS_WEBHOOK_URL",
]


def _log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}")


def _load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"meeting_id": None, "processed_transcript_ids": []}
    with open(STATE_FILE) as f:
        return json.load(f)


def _save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        _log("Not configured -- missing: " + ", ".join(missing))
        _log("See this file's docstring for what each one is and how to get it.")
        sys.exit(1)

    tenant_id = os.environ["TEAMS_TENANT_ID"]
    client_id = os.environ["TEAMS_CLIENT_ID"]
    client_secret = os.environ["TEAMS_CLIENT_SECRET"]
    organizer_id = os.environ["TEAMS_ORGANIZER_USER_ID"]
    join_url = os.environ["TEAMS_MEETING_JOIN_URL"]
    webhook_url = os.environ["TEAMS_WEBHOOK_URL"]

    token, error = get_app_token(tenant_id, client_id, client_secret)
    if error:
        _log(f"Auth failed: {error}")
        sys.exit(1)

    state = _load_state()
    if not state.get("meeting_id"):
        meeting_id, error = resolve_meeting_id(token, organizer_id, join_url)
        if error:
            _log(f"Couldn't resolve meeting ID: {error}")
            sys.exit(1)
        state["meeting_id"] = meeting_id
        _save_state(state)
        _log(f"Resolved and cached meeting ID: {meeting_id}")

    already_seen = set(state.get("processed_transcript_ids", []))
    new_transcripts, error = list_new_transcripts(
        token, organizer_id, state["meeting_id"], already_seen)
    if error:
        _log(f"Couldn't list transcripts: {error}")
        sys.exit(1)

    if not new_transcripts:
        _log("No new transcripts since last run.")
        return

    for transcript in new_transcripts:
        transcript_id = transcript["id"]
        created = transcript.get("createdDateTime", "")
        _log(f"New transcript {transcript_id} (created {created}) -- fetching...")

        vtt_text, error = get_transcript_vtt(token, organizer_id, state["meeting_id"], transcript_id)
        if error:
            _log(f"  Couldn't fetch content, skipping this one for now: {error}")
            continue  # don't mark as processed -- retry it next run

        date_str = created[:10] if created else "unknown-date"
        base_name = f"{date_str}_teams_meeting_{transcript_id[-8:]}"
        out_path, reply = run_mom_pipeline(vtt_text, base_name)
        if reply is None:
            _log("  MoM generation failed, skipping this one for now.")
            continue  # don't mark as processed -- retry it next run

        _log(f"  Saved to {out_path}")

        ok, error = send_to_teams(webhook_url, f"Minutes of Meeting — {date_str}", reply)
        if ok:
            _log("  Posted to Teams.")
        else:
            _log(f"  Generated fine, but couldn't post to Teams: {error}")
            # Still mark as processed -- the MoM exists locally and
            # re-running won't fix a webhook problem; don't reprocess forever.

        already_seen.add(transcript_id)
        state["processed_transcript_ids"] = sorted(already_seen)
        _save_state(state)  # persist after each one, not just at the end


if __name__ == "__main__":
    main()
