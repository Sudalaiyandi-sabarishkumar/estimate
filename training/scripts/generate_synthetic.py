#!/usr/bin/env python3
"""Generates the three synthetic training JSONL files.

Content (feature domains, task lists, dependency scenarios, wording) is authored
directly below — this script's job is only to render that authored content into
the matrix of variants (JSON vs chat format, correction-present vs absent,
dependency direction/ID-order/presentation-order) needed to reach a workable
training volume, since real ADO data alone (12 examples, 0 for dependency) is
far too thin to fine-tune on.

Output:
  training/data/synthetic/breakdown_json_synthetic.jsonl
  training/data/synthetic/breakdown_chat_synthetic.jsonl
  training/data/synthetic/dependency_synthetic.jsonl
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from azure_task_breakdown import BASE_SYSTEM_PROMPT as JSON_SYSTEM_PROMPT
from task_breakdown_chat import BASE_SYSTEM_PROMPT as CHAT_SYSTEM_PROMPT

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

# ---------------------------------------------------------------------------
# Feature domains for breakdown examples (title, acceptance-criteria-style
# description, and a real task list with category/complexity/hours).
# ---------------------------------------------------------------------------

DOMAINS = [
    {
        "title": "As a user, I want to reset my password via email link so that I can regain access if I forget it.",
        "criteria": "Given the user is on the login page, when they click 'Forgot password' and enter their email, "
                     "then the system sends a reset link valid for 30 minutes. When they click the link and set a "
                     "new password meeting complexity rules, then their password is updated and they are logged in.",
        "tasks": [
            ("Forgot-password form + validation", "Frontend", 2, 4),
            ("Reset-password form with complexity rules", "Frontend", 3, 5),
            ("Password reset token generation + email send endpoint", "Backend", 4, 8),
            ("Token verification + password update endpoint", "Backend", 3, 6),
            ("Rate limiting on reset requests", "Backend", 2, 4),
            ("Unit tests for token expiry and complexity rules", "Testing", 2, 4),
            ("End-to-end test of the full reset flow", "Testing", 3, 5),
        ],
    },
    {
        "title": "As a user, I want to upload a profile picture so that other users can recognize me.",
        "criteria": "Given the user is on their profile page, when they select an image under 5MB, then it is "
                     "cropped to a square, uploaded, and shown immediately. Invalid file types are rejected with a "
                     "clear error message.",
        "tasks": [
            ("Image picker + client-side crop UI", "Frontend", 3, 6),
            ("Upload progress indicator and error states", "Frontend", 2, 3),
            ("Image upload endpoint with size/type validation", "Backend", 3, 6),
            ("Image resize/thumbnail generation pipeline", "Backend", 4, 8),
            ("Unit tests for file validation", "Testing", 1, 3),
            ("Manual QA across file types and sizes", "Testing", 2, 3),
        ],
    },
    {
        "title": "As a user, I want to receive a push notification when someone comments on my post, so I don't miss engagement.",
        "criteria": "Given another user comments on my post, when the comment is saved, then I receive a push "
                     "notification within a few seconds, and tapping it opens the post at that comment.",
        "tasks": [
            ("Push notification permission prompt UI", "Frontend", 1, 3),
            ("Deep link handling to open post at comment", "Frontend", 3, 5),
            ("Notification dispatch service on comment creation", "Backend", 4, 8),
            ("Device token registration + storage", "Backend", 2, 4),
            ("Integration test for notification delivery", "Testing", 3, 5),
        ],
    },
    {
        "title": "As a shopper, I want to apply a promo code at checkout so that I get a discount.",
        "criteria": "Given the user is on the checkout page, when they enter a valid promo code, then the order "
                     "total updates to reflect the discount. Invalid or expired codes show a clear error and the "
                     "total does not change.",
        "tasks": [
            ("Promo code input field on checkout", "Frontend", 2, 3),
            ("Discount summary line in order total UI", "Frontend", 1, 2),
            ("Promo code validation endpoint (expiry, usage limits)", "Backend", 4, 8),
            ("Discount calculation logic in pricing service", "Backend", 3, 6),
            ("Unit tests for expired/invalid/over-limit codes", "Testing", 2, 4),
            ("End-to-end checkout test with and without promo", "Testing", 2, 4),
        ],
    },
    {
        "title": "As a user, I want to search for products by name and filter by category and price range, so I can find what I need quickly.",
        "criteria": "Given the user is on the search page, when they type a query, then matching products appear "
                     "with debounced search-as-you-type. Category and price filters narrow results without a full "
                     "page reload.",
        "tasks": [
            ("Search bar with debounced input", "Frontend", 2, 4),
            ("Category and price range filter UI", "Frontend", 3, 6),
            ("Search results list with pagination", "Frontend", 2, 4),
            ("Search endpoint with filtering and pagination", "Backend", 4, 8),
            ("Search index/query optimization", "Backend", 4, 8),
            ("Unit tests for filter combinations", "Testing", 2, 4),
            ("Performance test with large product catalog", "Testing", 3, 6),
        ],
    },
    {
        "title": "As a user, I want to enable two-factor authentication using an authenticator app, so my account is more secure.",
        "criteria": "Given the user is in security settings, when they enable 2FA, then a QR code is shown to scan "
                     "into an authenticator app, and a 6-digit code must be confirmed before 2FA is active. Login "
                     "afterward requires the code.",
        "tasks": [
            ("2FA setup screen with QR code display", "Frontend", 3, 5),
            ("Code entry screen at login when 2FA is enabled", "Frontend", 2, 4),
            ("TOTP secret generation + QR code endpoint", "Backend", 4, 8),
            ("Code verification endpoint + backup codes", "Backend", 5, 10),
            ("Unit tests for TOTP validation window", "Testing", 2, 4),
            ("End-to-end test of enable/login/disable flow", "Testing", 3, 6),
        ],
    },
    {
        "title": "As an admin, I want to export a report of monthly active users as a CSV, so I can share it with stakeholders.",
        "criteria": "Given the admin is on the reports page, when they select a month and click Export, then a CSV "
                     "downloads with daily active user counts and a monthly total. Large date ranges show a loading "
                     "state instead of blocking the UI.",
        "tasks": [
            ("Month picker + export button UI", "Frontend", 1, 3),
            ("Loading state for large exports", "Frontend", 1, 2),
            ("CSV generation endpoint with date-range aggregation", "Backend", 4, 8),
            ("Background job for large exports to avoid timeouts", "Backend", 4, 8),
            ("Unit tests for aggregation correctness", "Testing", 2, 4),
        ],
    },
    {
        "title": "As a user, I want to see a live typing indicator in chat, so I know when the other person is responding.",
        "criteria": "Given two users are in a chat, when one starts typing, then the other sees 'X is typing...' "
                     "within a second, and it disappears if typing stops for 3 seconds or a message is sent.",
        "tasks": [
            ("Typing indicator UI component", "Frontend", 2, 3),
            ("Debounced typing-state emit on keystroke", "Frontend", 2, 3),
            ("WebSocket event for typing state broadcast", "Backend", 4, 8),
            ("Typing-state timeout/cleanup logic", "Backend", 2, 4),
            ("Manual test across two clients for timing", "Testing", 2, 3),
        ],
    },
    {
        "title": "As a user, I want to filter my order history by status (delivered, cancelled, returned), so I can find a specific order.",
        "criteria": "Given the user is on order history, when they select a status filter, then only matching "
                     "orders show, with a count badge per status and an empty state when none match.",
        "tasks": [
            ("Status filter tabs with count badges", "Frontend", 2, 4),
            ("Empty state UI for no matching orders", "Frontend", 1, 2),
            ("Order history endpoint with status filtering", "Backend", 3, 6),
            ("Unit tests for filter + empty cases", "Testing", 1, 3),
        ],
    },
    {
        "title": "As a user, I want to link my bank account for direct payouts, so I can withdraw earnings.",
        "criteria": "Given the user is in payout settings, when they enter valid bank details, then the account is "
                     "verified via a micro-deposit or instant verification provider before payouts are enabled. "
                     "Invalid details are rejected with specific field errors.",
        "tasks": [
            ("Bank account entry form with field-level validation", "Frontend", 3, 5),
            ("Verification status UI (pending/verified/failed)", "Frontend", 2, 3),
            ("Bank verification provider integration", "Backend", 6, 12),
            ("Payout eligibility check based on verification status", "Backend", 3, 6),
            ("Unit tests for verification state transitions", "Testing", 3, 5),
            ("Security review of bank data handling", "Other", 3, 6),
        ],
    },
    {
        "title": "As a user, I want to schedule a recurring reminder (daily/weekly), so I don't have to set it manually each time.",
        "criteria": "Given the user creates a reminder, when they choose a recurrence (daily or weekly) and a time, "
                     "then the reminder fires at that time on each matching day until cancelled or the recurrence "
                     "ends.",
        "tasks": [
            ("Recurrence picker UI (daily/weekly + time)", "Frontend", 3, 5),
            ("Upcoming reminders list with recurrence indicator", "Frontend", 2, 3),
            ("Recurring schedule engine + reminder dispatch", "Backend", 5, 10),
            ("Timezone-aware scheduling logic", "Backend", 4, 8),
            ("Unit tests for daily/weekly edge cases (DST, month-end)", "Testing", 3, 6),
        ],
    },
    {
        "title": "As a user, I want to see a dark mode toggle in settings, so I can use the app comfortably at night.",
        "criteria": "Given the user opens settings, when they toggle dark mode, then all screens immediately switch "
                     "theme, and the preference persists across app restarts.",
        "tasks": [
            ("Theme toggle UI in settings", "Frontend", 1, 2),
            ("Dark theme token/color definitions across screens", "Frontend", 4, 8),
            ("Persist theme preference locally", "Frontend", 1, 2),
            ("Visual QA across all screens in both themes", "Testing", 3, 6),
        ],
    },
    {
        "title": "As a user, I want to invite a friend via a referral link, so we both get a reward when they sign up.",
        "criteria": "Given the user shares their referral link, when a new user signs up through it, then both "
                     "the referrer and referee receive the reward once the referee completes onboarding.",
        "tasks": [
            ("Referral link generation + share sheet UI", "Frontend", 2, 4),
            ("Referral status screen (pending/completed rewards)", "Frontend", 2, 4),
            ("Referral link tracking + attribution endpoint", "Backend", 4, 8),
            ("Reward issuance triggered on referee onboarding completion", "Backend", 4, 8),
            ("Fraud check for self-referrals", "Backend", 3, 6),
            ("Unit tests for attribution + reward issuance", "Testing", 3, 6),
        ],
    },
    {
        "title": "As a user, I want to see my daily step count synced from my phone's health app, so I can track activity.",
        "criteria": "Given the user grants health data permission, when they open the activity screen, then today's "
                     "step count syncs and displays, refreshing when the app is foregrounded.",
        "tasks": [
            ("Health data permission request UI", "Frontend", 1, 3),
            ("Step count display widget", "Frontend", 2, 3),
            ("Health platform integration (read steps)", "Backend", 4, 8),
            ("Background sync on app foreground", "Backend", 2, 4),
            ("Manual test across permission grant/deny paths", "Testing", 2, 3),
        ],
    },
    {
        "title": "As a user, I want to leave a star rating and written review after a purchase, so others can benefit from my experience.",
        "criteria": "Given the user has a delivered order, when they open the review prompt, then they can select "
                     "1-5 stars and write an optional comment, submit once per order, and edit within 48 hours.",
        "tasks": [
            ("Star rating + comment input UI", "Frontend", 3, 5),
            ("Edit-review flow within the 48h window", "Frontend", 2, 4),
            ("Review submission endpoint with one-per-order rule", "Backend", 3, 6),
            ("Edit window enforcement logic", "Backend", 2, 4),
            ("Unit tests for edit-window boundary", "Testing", 2, 3),
        ],
    },
]

# Some domains are frontend-only or backend-only, to make sure the model
# doesn't learn "always produce all four categories."
DOMAINS.append({
    "title": "As a user, I want the onboarding carousel to support swipe gestures between slides, so navigation feels natural.",
    "criteria": "Given the user is on the onboarding carousel, when they swipe left or right, then the view "
                 "transitions to the adjacent slide with animation; swiping past the last slide navigates to sign-up.",
    "tasks": [
        ("Swipe gesture handling on carousel", "Frontend", 3, 5),
        ("Slide transition animation", "Frontend", 2, 4),
        ("Navigation to sign-up on final swipe", "Frontend", 1, 2),
        ("Manual QA of gesture edge cases (fast swipe, partial swipe)", "Testing", 2, 3),
    ],
})
DOMAINS.append({
    "title": "As a system, I want stale idempotency keys purged nightly, so the idempotency table doesn't grow unbounded.",
    "criteria": "Given idempotency keys older than 7 days exist, when the nightly cleanup job runs, then those "
                 "records are deleted and the job logs how many were removed.",
    "tasks": [
        ("Nightly cleanup job for stale idempotency keys", "Backend", 3, 6),
        ("Job run logging + row-count metric", "Backend", 1, 3),
        ("Unit test for the 7-day cutoff boundary", "Testing", 1, 3),
    ],
})


def make_json_task(title, category, hmin, hmax, dependencies=None, risk=""):
    return {
        "title": title,
        "category": category,
        "description": f"{title} — implement and wire up as described in the acceptance criteria.",
        "complexity": "S" if hmax <= 4 else ("L" if hmin > 16 else "M"),
        "estimate_hours_min": hmin,
        "estimate_hours_max": hmax,
        "dependencies": dependencies or [],
        "risks": risk,
    }


def summarize_title(title: str) -> str:
    """'As a X, I want to Y, so that Z.' -> 'Y' (capitalized), a real one-sentence goal."""
    m = __import__("re").search(r"I want to (.+?)(?:, so that|\.$|$)", title)
    goal = m.group(1).strip() if m else title.strip()
    return goal[0].upper() + goal[1:] if goal else title


def render_json_example(domain, item_id, jitter=0):
    tasks = [make_json_task(t, cat, hmin + jitter, hmax + jitter) for (t, cat, hmin, hmax) in domain["tasks"]]
    user_content = (
        f"WORK ITEM CONTEXT (#{item_id})\n\nType: User Story\n\nTitle: {domain['title']}\n\n"
        f"Acceptance criteria:\n{domain['criteria']}"
    )
    assistant_obj = {
        "summary": summarize_title(domain["title"]),
        "assumptions": ["Requirements are as stated in the acceptance criteria; no additional hidden requirements."],
        "tasks": tasks,
    }
    return {
        "messages": [
            {"role": "system", "content": JSON_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_obj)},
        ],
        "_task_type": "breakdown_json",
    }


def render_chat_example(domain, item_id, correction=None):
    by_cat = {}
    for (t, cat, hmin, hmax) in domain["tasks"]:
        by_cat.setdefault(cat, []).append((t, hmin, hmax))

    # Correction says backend work is needed -- add it even if the domain has none.
    if correction and correction.get("added_task"):
        by_cat.setdefault("Backend", [])
        by_cat["Backend"].append((correction["added_task"], correction["hmin"], correction["hmax"]))

    lines = []
    grand_min = grand_max = 0
    for cat in ["Frontend", "Backend", "Testing", "Other"]:
        items = by_cat.get(cat)
        if not items:
            continue
        lines.append(f"{cat}:")
        cat_min = cat_max = 0
        for (t, hmin, hmax) in items:
            lines.append(f"- {t} ({hmin}-{hmax}h)")
            cat_min += hmin
            cat_max += hmax
        lines.append(f"Total: {cat_min}-{cat_max}h\n")
        grand_min += cat_min
        grand_max += cat_max
    lines.append(f"Total estimate: {grand_min}-{grand_max} hours (~{grand_min/8:.1f}-{grand_max/8:.1f} days)")
    assistant_text = "\n".join(lines)

    context = f"WORK ITEM CONTEXT (#{item_id})\n\nType: User Story\n\nTitle: {domain['title']}\n\nAcceptance criteria:\n{domain['criteria']}"
    if correction:
        context += (
            "\n\nCORRECTIONS FOR THIS SPECIFIC ITEM — mandatory, not optional:\n"
            f"  - {correction['note']}\n"
            "For each correction above, actually change the breakdown to match it: "
            "add, remove, or resize real tasks with real hour estimates. Do not just "
            "mention or cite the correction in a task description. Never write \"N/A\" "
            "or \"not needed\" for something a correction says is required. If a "
            "correction says backend work is needed, give Backend its own tasks and "
            "hours in the Backend section — do not fold it into a Frontend sub-bullet."
        )
    user_content = context + "\n\nGive me a full breakdown."

    return {
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ],
        "_task_type": "breakdown_chat",
    }


def build_breakdown_examples():
    json_examples = []
    chat_examples = []
    next_id = 90001

    # JSON-format family: each domain rendered ~3x with small hour jitter.
    for domain in DOMAINS:
        for jitter in (0, 1, -1):
            json_examples.append(render_json_example(domain, next_id, jitter=max(jitter, 0)))
            next_id += 1

    # Chat-format family, no correction: subset of domains.
    for domain in DOMAINS[:12]:
        chat_examples.append(render_chat_example(domain, next_id))
        next_id += 1

    # Chat-format family WITH a correction block that must be honored.
    correction_specs = [
        {"note": "this needs backend changes too — the reset flow needs server-side rate "
                  "limiting we haven't scoped", "added_task": "Server-side rate limiting for reset attempts",
         "hmin": 3, "hmax": 6},
        {"note": "this needs backend changes too — profile picture moderation needs a "
                  "backend content-safety check before publishing", "added_task": "Automated content-safety check on upload",
         "hmin": 4, "hmax": 8},
        {"note": "this needs backend changes too — comments need spam filtering before "
                  "triggering a notification", "added_task": "Spam filter check before notification dispatch",
         "hmin": 3, "hmax": 6},
        {"note": "this needs backend changes too — promo codes need fraud detection for "
                  "abuse patterns", "added_task": "Fraud detection on promo code redemption",
         "hmin": 4, "hmax": 8},
        {"note": "this needs backend changes too — search needs query logging for analytics",
         "added_task": "Search query logging pipeline", "hmin": 2, "hmax": 4},
        {"note": "this needs backend changes too — 2FA needs an audit log of enable/disable "
                  "events", "added_task": "Audit logging for 2FA state changes", "hmin": 2, "hmax": 4},
        {"note": "this needs backend changes too — exports need access-control checks per "
                  "admin role", "added_task": "Role-based access control on export endpoint",
         "hmin": 3, "hmax": 6},
        {"note": "this needs backend changes too — typing indicators need to be rate-limited "
                  "server-side to avoid abuse", "added_task": "Server-side rate limiting on typing events",
         "hmin": 2, "hmax": 4},
        {"note": "this needs backend changes too — order-history filtering needs caching for "
                  "large accounts", "added_task": "Caching layer for order history queries",
         "hmin": 3, "hmax": 6},
        {"note": "this needs backend changes too — bank linking needs a webhook handler for "
                  "async verification updates", "added_task": "Webhook handler for verification provider callbacks",
         "hmin": 4, "hmax": 8},
        {"note": "this it needs only half the frontend hours estimated — the design is "
                  "already done and just needs wiring", "added_task": None, "hmin": None, "hmax": None},
        {"note": "this needs backend changes too — reminders need a dead-letter queue for "
                  "failed dispatch attempts", "added_task": "Dead-letter queue for failed reminder dispatch",
         "hmin": 3, "hmax": 6},
    ]
    for domain, spec in zip(DOMAINS, correction_specs):
        if spec["added_task"] is None:
            # "reduce hours" correction variant -- halve frontend hours in the response,
            # rather than adding a Backend task.
            reduced_domain = {
                "title": domain["title"],
                "criteria": domain["criteria"],
                "tasks": [(t, cat, max(hmin // 2, 1), max(hmax // 2, max(hmin // 2, 1) + 1))
                          if cat == "Frontend" else (t, cat, hmin, hmax)
                          for (t, cat, hmin, hmax) in domain["tasks"]],
            }
            chat_examples.append(render_chat_example(reduced_domain, next_id, correction={"note": spec["note"]}))
        else:
            chat_examples.append(render_chat_example(domain, next_id, correction=spec))
        next_id += 1

    return json_examples, chat_examples


# ---------------------------------------------------------------------------
# Dependency scenarios.
# ---------------------------------------------------------------------------

# Each scenario: two short flow descriptions that share a point (or don't),
# plus which one is the "depended on" one (flow ENDS at shared point) when
# a real dependency exists. category is one of:
#   clear_dependency, shared_parent_only, explicit_ado_link, no_relationship
DEPENDENCY_SCENARIOS = [
    {
        "category": "clear_dependency",
        "shared_point": "the order confirmation screen",
        "depended_on": {"title": "Checkout flow", "flow": "collects payment details and, on success, navigates to the order confirmation screen."},
        "dependent": {"title": "Order tracking", "flow": "starts from the order confirmation screen and lets the user track shipment status."},
        "parent_feature": "Checkout & Fulfillment",
    },
    {
        "category": "clear_dependency",
        "shared_point": "the user's verified email address",
        "depended_on": {"title": "Email verification", "flow": "sends a verification link and marks the email as verified once clicked."},
        "dependent": {"title": "Newsletter subscription", "flow": "requires a verified email address before the user can subscribe to the newsletter."},
        "parent_feature": "Account Management",
    },
    {
        "category": "clear_dependency",
        "shared_point": "the created workspace",
        "depended_on": {"title": "Workspace creation", "flow": "lets an admin create a new workspace and configure its name and settings."},
        "dependent": {"title": "Team member invites", "flow": "starts from an existing workspace and lets the admin invite members to it."},
        "parent_feature": "Workspace Management",
    },
    {
        "category": "clear_dependency",
        "shared_point": "the uploaded document",
        "depended_on": {"title": "Document upload", "flow": "lets the user upload a PDF and stores it for later access."},
        "dependent": {"title": "Document annotation", "flow": "opens an already-uploaded document and lets the user highlight and comment on it."},
        "parent_feature": "Document Workspace",
    },
    {
        "category": "clear_dependency",
        "shared_point": "the connected payment method",
        "depended_on": {"title": "Payment method setup", "flow": "lets the user add and verify a credit card or bank account."},
        "dependent": {"title": "Subscription checkout", "flow": "starts from a verified payment method and completes a subscription purchase."},
        "parent_feature": "Billing",
    },
    {
        "category": "clear_dependency",
        "shared_point": "the generated API key",
        "depended_on": {"title": "API key generation", "flow": "lets a developer generate a scoped API key from the developer console."},
        "dependent": {"title": "Webhook configuration", "flow": "requires an existing API key to authenticate before a webhook endpoint can be registered."},
        "parent_feature": "Developer Platform",
    },
    {
        "category": "clear_dependency",
        "shared_point": "the completed KYC check",
        "depended_on": {"title": "KYC identity verification", "flow": "collects ID documents and marks the user as verified once approved."},
        "dependent": {"title": "Withdrawal request", "flow": "requires a completed KYC check before allowing the user to request a withdrawal."},
        "parent_feature": "Compliance",
    },
    {
        "category": "clear_dependency",
        "shared_point": "the published product listing",
        "depended_on": {"title": "Product listing creation", "flow": "lets a seller fill in product details and publish a listing."},
        "dependent": {"title": "Listing promotion", "flow": "starts from an already-published listing and lets the seller boost its visibility."},
        "parent_feature": "Seller Tools",
    },
    {
        "category": "clear_dependency",
        "shared_point": "the imported contact list",
        "depended_on": {"title": "Contact import", "flow": "lets the user import contacts from a CSV or their address book."},
        "dependent": {"title": "Bulk invite", "flow": "starts from an imported contact list and sends invites to selected contacts."},
        "parent_feature": "Growth",
    },
    {
        "category": "clear_dependency",
        "shared_point": "the linked calendar account",
        "depended_on": {"title": "Calendar account linking", "flow": "lets the user connect their Google or Outlook calendar."},
        "dependent": {"title": "Meeting scheduling", "flow": "requires a linked calendar before it can check availability and book a meeting."},
        "parent_feature": "Scheduling",
    },
    {
        "category": "shared_parent_only",
        "shared_point": None,
        "depended_on": {"title": "Push notification preferences", "flow": "lets the user toggle which notification categories they receive."},
        "dependent": {"title": "In-app notification badge", "flow": "shows an unread-count badge on the notifications icon."},
        "parent_feature": "Notifications",
    },
    {
        "category": "shared_parent_only",
        "shared_point": None,
        "depended_on": {"title": "Font size accessibility setting", "flow": "lets the user increase text size across the app."},
        "dependent": {"title": "Color contrast accessibility setting", "flow": "lets the user enable a high-contrast color theme."},
        "parent_feature": "Accessibility",
    },
    {
        "category": "shared_parent_only",
        "shared_point": None,
        "depended_on": {"title": "Export report as CSV", "flow": "lets an admin download a report as a CSV file."},
        "dependent": {"title": "Export report as PDF", "flow": "lets an admin download the same report as a formatted PDF."},
        "parent_feature": "Reporting",
    },
    {
        "category": "shared_parent_only",
        "shared_point": None,
        "depended_on": {"title": "Light theme customization", "flow": "lets the user pick an accent color for light mode."},
        "dependent": {"title": "Dark theme customization", "flow": "lets the user pick an accent color for dark mode."},
        "parent_feature": "Theming",
    },
    {
        "category": "explicit_ado_link",
        "shared_point": "the migrated user database",
        "depended_on": {"title": "Database migration to new schema", "flow": "migrates existing user records to the new normalized schema."},
        "dependent": {"title": "New profile fields UI", "flow": "displays profile fields that only exist in the new schema."},
        "parent_feature": "Platform Migration",
        "has_explicit_link": True,
    },
    {
        "category": "explicit_ado_link",
        "shared_point": "the new pricing engine",
        "depended_on": {"title": "Pricing engine rewrite", "flow": "replaces the legacy pricing calculation service."},
        "dependent": {"title": "Checkout price display", "flow": "calls the pricing engine to show line-item and total prices at checkout."},
        "parent_feature": "Pricing",
        "has_explicit_link": True,
    },
    {
        "category": "explicit_ado_link",
        "shared_point": "the new authentication service",
        "depended_on": {"title": "Auth service replatform", "flow": "replaces session-cookie auth with token-based auth."},
        "dependent": {"title": "Mobile app login screen update", "flow": "updates the login screen to use the new token-based auth service."},
        "parent_feature": "Platform Migration",
        "has_explicit_link": True,
    },
    {
        "category": "no_relationship",
        "shared_point": None,
        "depended_on": {"title": "Weather widget on home screen", "flow": "shows current weather based on device location."},
        "dependent": {"title": "Invoice PDF generation", "flow": "generates a downloadable PDF invoice for a completed order."},
        "parent_feature": None,
    },
    {
        "category": "no_relationship",
        "shared_point": None,
        "depended_on": {"title": "App icon badge count", "flow": "shows the number of unread messages on the app icon."},
        "dependent": {"title": "Warehouse inventory sync", "flow": "syncs stock counts from the warehouse system every hour."},
        "parent_feature": None,
    },
    {
        "category": "no_relationship",
        "shared_point": None,
        "depended_on": {"title": "Splash screen animation", "flow": "plays a short logo animation on app launch."},
        "dependent": {"title": "Admin audit log viewer", "flow": "lets an admin search and filter historical audit log entries."},
        "parent_feature": None,
    },
]


def render_dependency_example(scenario, id_a, id_b, a_is_first, ex_id):
    """id_a is always the 'depended_on' item's synthetic id, id_b the
    'dependent' item's, before applying presentation order."""
    depended = scenario["depended_on"]
    dependent = scenario["dependent"]
    category = scenario["category"]

    def describe(item_id, role):
        d = role
        return (f"WORK ITEM CONTEXT (#{item_id})\n\nType: User Story\n\n"
                f"Title: {d['title']}\n\nAcceptance criteria:\n{d['title']} {d['flow']}")

    ctx_depended = describe(id_a, depended)
    ctx_dependent = describe(id_b, dependent)

    if scenario.get("has_explicit_link"):
        ctx_dependent += (
            f"\n\nLinked work items:\n  - Depends on: #{id_a} [User Story, New] {depended['title']}"
        )

    contexts = [ctx_depended, ctx_dependent] if a_is_first else [ctx_dependent, ctx_depended]

    instruction = (
        f"Analyse the dependency between #{id_a} and #{id_b} using the method described "
        f"above. Refer to the two items ONLY as #{id_a} and #{id_b} throughout your answer "
        f"— never substitute a parent Feature's ID or a linked Task's ID for either of "
        f"these two items, even when citing which one a Feature or Task belongs to.\n"
        f"Show your work — do not skip straight to a conclusion:\n"
        f"Step 1: What point (screen, entity, or data) do the two items share, if any?\n"
        f"Step 2: Which item's flow ENDS at that point, and which item's flow STARTS "
        f"there? Apply the rule exactly: ENDS-there = depended on, STARTS-there = "
        f"dependent. Do not skip this step or guess the direction any other way.\n"
        f"Step 3: Do they share a parent Feature? (Note it, but it doesn't decide direction.)\n"
        f"Step 4: Check the 'Linked work items' data above — is there an explicit "
        f"Depends-on/Blocks relation, or only Parent/Child?\n"
        f"Then give your final conclusion in one line: which item depends on which, and "
        f"whether that's formally recorded in ADO or only implied by the requirements."
    )
    user_content = "\n\n---\n\n".join(contexts) + "\n\n" + instruction

    if category in ("clear_dependency", "explicit_ado_link"):
        shared = scenario["shared_point"]
        step1 = f"Step 1: The two items share {shared}."
        step2 = (f"Step 2: #{id_a} ({depended['title']}) ENDS at {shared} -> it is depended on. "
                 f"#{id_b} ({dependent['title']}) STARTS at {shared} -> it is the dependent one.")
        step3 = (f"Step 3: They share a parent Feature ({scenario['parent_feature']})."
                 if scenario.get("parent_feature") else "Step 3: They do not share a parent Feature.")
        if scenario.get("has_explicit_link"):
            step4 = f"Step 4: The 'Linked work items' data shows an explicit Depends-on relation: #{id_b} depends on #{id_a}."
            recorded = "formally recorded in ADO as an explicit Depends-on relation"
        else:
            step4 = "Step 4: The 'Linked work items' data shows only Parent/Child links to their own Features, not an explicit Depends-on relation."
            recorded = "only implied by the requirements, not formally recorded in ADO"
        conclusion = f"Conclusion: #{id_b} depends on #{id_a} — its flow begins at the exact point #{id_a}'s flow ends ({shared}); this is {recorded}."
        assistant = "\n".join([step1, step2, step3, step4, conclusion])
    elif category == "shared_parent_only":
        step1 = "Step 1: The two items do not share any screen, entity, or data — they only share a parent Feature."
        step2 = "Step 2: Neither item's flow starts or ends at a point the other uses, so the ENDS/STARTS rule does not apply."
        step3 = f"Step 3: They share a parent Feature ({scenario['parent_feature']}), but that alone does not create a dependency."
        step4 = "Step 4: The 'Linked work items' data shows only Parent/Child links to their shared Feature, no explicit Depends-on relation."
        conclusion = f"Conclusion: Neither #{id_a} nor #{id_b} depends on the other — they are independent items under the same Feature."
        assistant = "\n".join([step1, step2, step3, step4, conclusion])
    else:  # no_relationship
        step1 = "Step 1: The two items share no screen, entity, or data."
        step2 = "Step 2: The ENDS/STARTS rule does not apply since there is no shared point."
        step3 = "Step 3: They do not share a parent Feature either."
        step4 = "Step 4: The 'Linked work items' data shows no relation of any kind between them."
        conclusion = f"Conclusion: #{id_a} and #{id_b} are unrelated — neither depends on the other."
        assistant = "\n".join([step1, step2, step3, step4, conclusion])

    return {
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ] + [{"role": "assistant", "content": assistant}],
        "_task_type": "dependency_reasoning",
        "_category": category,
        "_id_order": "lower_depended_on" if int(id_a) < int(id_b) else "higher_depended_on",
        "_presentation_order": "depended_first" if a_is_first else "dependent_first",
    }


def build_dependency_examples():
    examples = []
    next_id = 50001
    for scenario in DEPENDENCY_SCENARIOS:
        # 4 variants: id-order (lower/higher depended-on) x presentation-order (which context comes first)
        for id_order_lower_depended in (True, False):
            for a_is_first in (True, False):
                if id_order_lower_depended:
                    id_a, id_b = next_id, next_id + 1
                else:
                    id_a, id_b = next_id + 1, next_id
                examples.append(render_dependency_example(scenario, str(id_a), str(id_b), a_is_first, next_id))
        next_id += 2
    return examples


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    json_examples, chat_examples = build_breakdown_examples()
    dep_examples = build_dependency_examples()

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    write_jsonl(os.path.join(OUT_DIR, "breakdown_json_synthetic.jsonl"), json_examples)
    write_jsonl(os.path.join(OUT_DIR, "breakdown_chat_synthetic.jsonl"), chat_examples)
    write_jsonl(os.path.join(OUT_DIR, "dependency_synthetic.jsonl"), dep_examples)

    print(f"breakdown_json_synthetic.jsonl: {len(json_examples)} examples")
    print(f"breakdown_chat_synthetic.jsonl: {len(chat_examples)} examples")
    print(f"dependency_synthetic.jsonl: {len(dep_examples)} examples")
    print(f"Total synthetic: {len(json_examples) + len(chat_examples) + len(dep_examples)}")


if __name__ == "__main__":
    main()
