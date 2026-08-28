#!/usr/bin/env python3
"""Unattended MoM pipeline: Teams channel meeting transcript -> MoM -> posted
back to the channel, with no human running a command. Handles any number of
recurring meetings, each independently configured.

This is the "full end-to-end auto-pull" half of the mom feature -- the
Phase 1 manual path (`/mom <file>` in task_breakdown_chat.py) keeps working
unchanged regardless of this. THIS SCRIPT CANNOT DO ANYTHING until your
tenant admin completes the three steps in teams_graph.py's docstring --
run it anyway and it'll tell you exactly which one is still missing rather
than failing silently.

Meant to be run on a schedule (cron/launchd), not as a long-lived daemon --
each run checks every configured meeting once for transcripts it hasn't
processed yet, then exits. It does not react in real time to a call ending;
whatever you schedule this at is the real-world latency between "call ends"
and "MoM posted" (Teams itself also takes some minutes to finish processing
a transcript, so there's no benefit to checking faster than that). Since
different meetings can end at different times, it's fine -- and expected --
to have several cron entries at different times all calling this same
script; every invocation checks *all* configured meetings regardless of
which one's schedule triggered it, so nothing needs to know which meeting
"caused" a given run.

Setup:
  1. Shared app credentials (same Entra app can serve multiple meetings --
     the admin just needs to grant the Application Access Policy to each
     additional organizer, not register a new app per meeting):
       export TEAMS_TENANT_ID="..."
       export TEAMS_CLIENT_ID="..."
       export TEAMS_CLIENT_SECRET="..."

  2. One entry per meeting in meetings_config.json (see
     meetings_config.example.json for the shape) -- each with its own
     organizer, join URL, and target channel webhook, since those
     genuinely differ per meeting.

  3. Schedule it, e.g. via cron -- one line per meeting's expected end
     time (all calling the same script; see run_auto_mom.sh):
       20 11 * * 1-5  .../run_auto_mom.sh >> auto_mom.log 2>&1   # daily standup
       0 16 * * 3     .../run_auto_mom.sh >> auto_mom.log 2>&1   # weekly planning

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
MEETINGS_CONFIG_FILE = "meetings_config.json"
REQUIRED_ENV = ["TEAMS_TENANT_ID", "TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET"]
REQUIRED_MEETING_KEYS = ["name", "organizer_user_id", "join_url", "webhook_url"]


def _log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}")


def _load_meetings_config():
    """Returns (list_of_meeting_dicts, error). Each dict has 'name',
    'organizer_user_id', 'join_url', 'webhook_url'."""
    if not os.path.exists(MEETINGS_CONFIG_FILE):
        return None, (f"{MEETINGS_CONFIG_FILE} not found -- copy "
                       f"meetings_config.example.json to {MEETINGS_CONFIG_FILE} "
                       f"and fill in your real meeting(s).")
    with open(MEETINGS_CONFIG_FILE) as f:
        meetings = json.load(f)
    if not isinstance(meetings, list) or not meetings:
        return None, f"{MEETINGS_CONFIG_FILE} must be a non-empty JSON list of meetings."
    for m in meetings:
        missing_keys = [k for k in REQUIRED_MEETING_KEYS if not m.get(k)]
        if missing_keys:
            return None, f"Meeting entry {m!r} is missing: {', '.join(missing_keys)}"
    names = [m["name"] for m in meetings]
    if len(names) != len(set(names)):
        return None, f"Meeting names in {MEETINGS_CONFIG_FILE} must be unique."
    return meetings, None


def _load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"meetings": {}}
    with open(STATE_FILE) as f:
        state = json.load(f)
    state.setdefault("meetings", {})
    return state


def _save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def process_meeting(token: str, meeting: dict, state: dict):
    """One meeting's worth of work: resolve/cache its meeting ID, find and
    process any transcripts not yet seen for it. State for each meeting is
    tracked separately under state['meetings'][name] so meetings don't
    interfere with each other's processed/seen history."""
    name = meeting["name"]
    organizer_id = meeting["organizer_user_id"]
    join_url = meeting["join_url"]
    webhook_url = meeting["webhook_url"]

    meeting_state = state["meetings"].setdefault(
        name, {"meeting_id": None, "processed_transcript_ids": []})

    if not meeting_state.get("meeting_id"):
        meeting_id, error = resolve_meeting_id(token, organizer_id, join_url)
        if error:
            _log(f"[{name}] Couldn't resolve meeting ID: {error}")
            return
        meeting_state["meeting_id"] = meeting_id
        _save_state(state)
        _log(f"[{name}] Resolved and cached meeting ID: {meeting_id}")

    already_seen = set(meeting_state.get("processed_transcript_ids", []))
    new_transcripts, error = list_new_transcripts(
        token, organizer_id, meeting_state["meeting_id"], already_seen)
    if error:
        _log(f"[{name}] Couldn't list transcripts: {error}")
        return

    if not new_transcripts:
        _log(f"[{name}] No new transcripts since last run.")
        return

    for transcript in new_transcripts:
        transcript_id = transcript["id"]
        created = transcript.get("createdDateTime", "")
        _log(f"[{name}] New transcript {transcript_id} (created {created}) -- fetching...")

        vtt_text, error = get_transcript_vtt(token, organizer_id, meeting_state["meeting_id"], transcript_id)
        if error:
            _log(f"[{name}]   Couldn't fetch content, skipping this one for now: {error}")
            continue  # don't mark as processed -- retry it next run

        date_str = created[:10] if created else "unknown-date"
        base_name = f"{date_str}_{name}_{transcript_id[-8:]}"
        out_path, reply = run_mom_pipeline(vtt_text, base_name)
        if reply is None:
            _log(f"[{name}]   MoM generation failed, skipping this one for now.")
            continue  # don't mark as processed -- retry it next run

        _log(f"[{name}]   Saved to {out_path}")

        ok, error = send_to_teams(webhook_url, f"Minutes of Meeting — {name} — {date_str}", reply)
        if ok:
            _log(f"[{name}]   Posted to Teams.")
        else:
            _log(f"[{name}]   Generated fine, but couldn't post to Teams: {error}")
            # Still mark as processed -- the MoM exists locally and
            # re-running won't fix a webhook problem; don't reprocess forever.

        already_seen.add(transcript_id)
        meeting_state["processed_transcript_ids"] = sorted(already_seen)
        _save_state(state)  # persist after each one, not just at the end


def main():
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        _log("Not configured -- missing: " + ", ".join(missing))
        _log("See this file's docstring for what each one is and how to get it.")
        sys.exit(1)

    meetings, error = _load_meetings_config()
    if error:
        _log(f"Not configured -- {error}")
        sys.exit(1)

    token, error = get_app_token(
        os.environ["TEAMS_TENANT_ID"], os.environ["TEAMS_CLIENT_ID"],
        os.environ["TEAMS_CLIENT_SECRET"])
    if error:
        _log(f"Auth failed: {error}")
        sys.exit(1)

    state = _load_state()
    for meeting in meetings:
        process_meeting(token, meeting, state)


if __name__ == "__main__":
    main()
