"""Microsoft Graph client for pulling Teams channel-meeting transcripts
automatically -- the Phase 2 half of the mom command (see transcript_utils.py
and task_breakdown_chat.py's mom docstring for the Phase 1 manual-file path,
which keeps working unchanged regardless of this).

THIS CANNOT WORK until a tenant admin does three things -- there is no
request-side or code-side way around any of them (verified against current
Microsoft Learn docs, Aug 2026):

  1. Turn on Graph access to transcripts for the tenant (off by default):
     Teams admin center -> Meetings -> Meeting settings -> Transcript API
     access -> "Microsoft Graph access" toggle On.
     Equivalent PowerShell (Teams module):
       Set-CsTeamsMeetingConfiguration -EnableGraphTranscriptAccess $true `
           -Identity Global

  2. Register an Azure AD app (Entra ID admin center -> App registrations)
     and grant admin consent for the Application permission
     OnlineMeetingTranscript.Read.All. Note this is the "least privileged"
     permission Microsoft documents for this API -- there's no narrower
     application-only option; OnlineMeetingTranscript.Read.Chat is chat-only
     and explicitly does not apply to channel meetings.

  3. Create + grant an Application Access Policy scoping that app to the
     specific user who organizes the recurring channel meeting (via the
     Teams/Skype for Business PowerShell module, connected as an admin):
       New-CsApplicationAccessPolicy -Identity <policy-name> `
           -AppIds "<this app's client ID>" -Description "MoM auto-pull"
       Grant-CsApplicationAccessPolicy -PolicyName <policy-name> `
           -Identity "<organizer's Entra object ID>"
     Changes can take up to 30 minutes to take effect. Without this step,
     Graph calls fail with 403 "No application access policy found for this
     app" even though the permission above was granted.

Once all three are done, everything below just works against api.rst
endpoints Microsoft's own docs use for exactly this scenario:
  - https://learn.microsoft.com/en-us/microsoftteams/meeting-transcript-api-access
  - https://learn.microsoft.com/en-us/graph/api/onlinemeeting-list-transcripts
  - https://learn.microsoft.com/en-us/graph/cloud-communication-online-meeting-application-access-policy
"""

import json
import urllib.error
import urllib.parse
import urllib.request

GRAPH_ROOT = "https://graph.microsoft.com/v1.0"


def get_app_token(tenant_id: str, client_id: str, client_secret: str):
    """App-only (client credentials) OAuth2 token -- no signed-in user, this
    is meant to run unattended. Returns (token, None) or (None, error)."""
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))["access_token"], None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        return None, f"Token request failed (HTTP {exc.code}): {detail}"
    except urllib.error.URLError as exc:
        return None, f"Couldn't reach login.microsoftonline.com: {exc.reason}"


def _graph_get(token: str, path_and_query: str):
    """Returns (json_or_None, error_or_None). Surfaces the two documented
    error codes callers actually need to distinguish -- the tenant-wide
    toggle being off vs. the per-user access policy being missing -- rather
    than one generic 'request failed' message, since the fix is different
    for each."""
    req = urllib.request.Request(
        f"{GRAPH_ROOT}{path_and_query}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if "GraphAccessToTranscriptsDisabled" in detail:
            return None, ("Tenant admin has not enabled Graph access to transcripts "
                           "-- see teams_graph.py's docstring, step 1.")
        if "No application access policy found" in detail:
            return None, ("No application access policy is granted for this app on "
                           "this organizer -- see teams_graph.py's docstring, step 3.")
        return None, f"Graph request failed (HTTP {exc.code}): {detail[:400]}"
    except urllib.error.URLError as exc:
        return None, f"Couldn't reach Microsoft Graph: {exc.reason}"


def resolve_meeting_id(token: str, organizer_user_id: str, join_web_url: str):
    """One-time lookup: turns the recurring channel meeting's join URL (copy
    it from the meeting invite/calendar event once) into the onlineMeetingId
    that every other call here needs. Returns (meeting_id, None) or
    (None, error). Cache the result -- it's stable for the life of that
    meeting series, no need to re-resolve it on every poll."""
    encoded = urllib.parse.quote(f"JoinWebUrl eq '{join_web_url}'")
    result, error = _graph_get(
        token, f"/users/{organizer_user_id}/onlineMeetings?$filter={encoded}")
    if error:
        return None, error
    values = result.get("value", [])
    if not values:
        return None, "No online meeting found for that join URL -- double check it was copied in full."
    return values[0]["id"], None


def list_new_transcripts(token: str, organizer_user_id: str, meeting_id: str,
                          already_seen: set):
    """Returns (list_of_new_transcript_dicts, error). Each dict has at least
    'id' and 'createdDateTime'. `already_seen` is the caller's set of
    transcript IDs processed on a previous poll (see auto_mom.py's state
    file) -- a recurring meeting accumulates one transcript per occurrence,
    so this is what makes polling idempotent."""
    result, error = _graph_get(
        token, f"/users/{organizer_user_id}/onlineMeetings/{meeting_id}/transcripts")
    if error:
        return None, error
    all_transcripts = result.get("value", [])
    new = [t for t in all_transcripts if t["id"] not in already_seen]
    new.sort(key=lambda t: t.get("createdDateTime", ""))
    return new, None


def get_transcript_vtt(token: str, organizer_user_id: str, meeting_id: str,
                        transcript_id: str):
    """Returns (vtt_text, None) or (None, error). The .vtt text this returns
    is byte-for-byte the same shape as what Teams' UI "Download transcript"
    button gives you -- transcript_utils.load_transcript()'s _parse_vtt()
    already handles it with zero changes needed."""
    url = (f"{GRAPH_ROOT}/users/{organizer_user_id}/onlineMeetings/{meeting_id}"
           f"/transcripts/{transcript_id}/content?$format=text/vtt")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8"), None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        return None, f"Couldn't fetch transcript content (HTTP {exc.code}): {detail}"
    except urllib.error.URLError as exc:
        return None, f"Couldn't reach Microsoft Graph: {exc.reason}"
