# Setting up the Teams auto-pull (`auto_mom.py`) on your real org

This targets your actual work tenant, not a throwaway dev sandbox. Nothing
in `auto_mom.py`/`teams_graph.py` changes either way — only who does which
setup step changes.

---

## Part A — things you can do yourself, no admin needed

### 1. Register the app in Entra ID
Most tenants allow any user to register an app (only *consenting* it needs
admin rights — see Part B). If this step gets blocked by tenant policy,
that becomes an extra admin ask.

1. Go to https://entra.microsoft.com → **Applications → App registrations →
   New registration**.
2. Name it anything (e.g. "mom-auto-pull"). Leave the default "Accounts in
   this organizational directory only" option. No redirect URI needed —
   this app never signs a user in interactively. Click **Register**.
3. On the app's Overview page, copy:
   - **Application (client) ID** → `TEAMS_CLIENT_ID`
   - **Directory (tenant) ID** → `TEAMS_TENANT_ID`
4. Go to **Certificates & secrets → Client secrets → New client secret**.
   Give it a description and expiry, click **Add**, then immediately copy
   the secret's **Value** (not the Secret ID) → `TEAMS_CLIENT_SECRET`. It's
   only shown once.
5. Go to **API permissions → Add a permission → Microsoft Graph →
   Application permissions**, search for `OnlineMeetingTranscript.Read.All`,
   check it, **Add permissions**. It'll show as "Not granted" until your
   admin consents it (Part B, step 2) — that's expected, move on.

### 2. Find the meeting organizer's object ID
Whoever organizes the recurring channel meeting (probably you, or whoever
schedules it) — Entra admin center → **Users** → that person → the
**Object ID** field on their profile page → `TEAMS_ORGANIZER_USER_ID`.

### 3. Get the meeting's Join URL
Open the recurring channel meeting's calendar invite / meeting details in
Teams, copy the **Join URL** → `TEAMS_MEETING_JOIN_URL`.

### 4. Set up the Teams webhook for posting
In the target channel: **"..." → Workflows → "Post to a channel when a
webhook request is received"** → finish the wizard → copy the URL →
`TEAMS_WEBHOOK_URL`. Channel-level action, no admin needed.

---

## Part B — send this to your admin

Just these three actions. Give them the app's **client ID** (from Part A,
step 1) and the organizer's **object ID** (Part A, step 2) along with the
ask.

### 1. Turn on Graph access to meeting transcripts
Off tenant-wide by default. Teams admin center → **Meetings → Meeting
settings → Transcript API access → Microsoft Graph access → On**.

Equivalent PowerShell:
```powershell
Set-CsTeamsMeetingConfiguration -EnableGraphTranscriptAccess $true -Identity Global
```

### 2. Grant admin consent on the app's permission
Entra ID → **App registrations** → the app from Part A → **API
permissions** → **Grant admin consent for [org name]**. The
`OnlineMeetingTranscript.Read.All` row should turn to a green check.

### 3. Create and grant the Application Access Policy
This is the step that's easy to miss — without it, everything above still
fails with `"No application access policy found for this app"`. Needs the
Teams PowerShell module, connected as admin:

```powershell
Connect-MicrosoftTeams

New-CsApplicationAccessPolicy -Identity mom-auto-pull-policy `
    -AppIds "<client ID from Part A>" `
    -Description "MoM auto-pull"

Grant-CsApplicationAccessPolicy -PolicyName mom-auto-pull-policy `
    -Identity "<organizer's object ID from Part A>"
```

Can take up to 30 minutes to propagate — Microsoft's own docs call this out
explicitly, so don't assume it's still broken if it fails right after
granting.

---

## Part C — test it

Once Part B is confirmed done:

```bash
export TEAMS_TENANT_ID="..."
export TEAMS_CLIENT_ID="..."
export TEAMS_CLIENT_SECRET="..."
export TEAMS_ORGANIZER_USER_ID="..."
export TEAMS_MEETING_JOIN_URL="..."
export TEAMS_WEBHOOK_URL="..."

python3 auto_mom.py
```

Run it by hand first (not via cron) so you can see what happens. First run
resolves and caches the meeting ID (`auto_mom_state.json`), then finds the
most recent transcript for that meeting, generates the MoM, and posts it to
the channel. Run it again right after — it should print "No new transcripts
since last run" instead of reposting, confirming the idempotency tracking
works.

If it fails, the printed error tells you which part is still missing —
`teams_graph.py` distinguishes "Graph access to transcripts is off" (Part
B, step 1) from "no access policy for this app" (Part B, step 3) from a
plain auth failure (Part A, step 1), rather than one generic message.

Once it's working reliably, schedule it via cron instead of running by
hand, e.g. every 15 minutes:
```
*/15 * * * * cd /path/to/agent && /usr/bin/python3 auto_mom.py >> auto_mom.log 2>&1
```
