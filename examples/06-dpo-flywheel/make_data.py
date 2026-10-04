"""Regenerates the example feedback files (deterministic). Not needed to run the demo."""

import json
from pathlib import Path

HERE = Path(__file__).parent


def ev(cid, prompt, response, rating, correction=None, model="support-bot-v3", ts="2026-10-03T10:00:00Z"):
    row = {"conversation_id": cid, "messages": [{"role": "user", "content": prompt}],
           "response": response, "rating": rating, "model": model, "ts": ts}
    if correction:
        row["correction"] = correction
    return row


topics = [
    ("How do I reset my password?", "Go to Settings, choose Security, click Reset password and follow the emailed link; it expires after 30 minutes."),
    ("How do I export my invoices as CSV?", "Open Billing, select Invoices, pick the date range and click Export CSV; the file downloads immediately."),
    ("Why was my card declined?", "Declines usually come from an expired card, a billing address mismatch or a bank fraud hold; check those, then retry."),
    ("How do I add a teammate to my workspace?", "Open Workspace settings, choose Members, click Invite and enter their email; they get admin rights only if you tick Admin."),
    ("Can I change my billing cycle to annual?", "Yes: in Billing choose Change plan, select Annual and confirm; the unused monthly balance is prorated as credit."),
    ("How do I enable two-factor authentication?", "In Settings under Security, turn on Two-factor, scan the QR code with an authenticator app and enter the 6-digit code."),
    ("How do I delete my account permanently?", "Go to Settings, Account, Delete account; confirm with your password. Data is purged after a 14-day grace period."),
    ("How do I connect the Slack integration?", "Open Integrations, choose Slack, click Connect, approve the workspace permissions and pick the channel for alerts."),
    ("Where can I download the mobile app?", "The app is on the App Store and Google Play; search for the product name and sign in with your existing account."),
    ("How do I set up SSO with Okta?", "In Security settings choose SSO, select Okta, paste your metadata URL, map the email attribute and test with one user."),
    ("How do I change the language of the dashboard?", "Click your avatar, open Preferences, pick a language from the Language menu and save; the dashboard reloads."),
    ("Why is my API key not working?", "Check the key is active under API settings, sent as a Bearer token, and that it has the scope the endpoint needs."),
    ("How do I restore a deleted project?", "Open Trash from the sidebar, find the project and click Restore; projects stay in Trash for 30 days."),
    ("How can I lower my monthly bill?", "Remove inactive seats, switch to annual billing for a discount, and downgrade unused add-ons in Billing."),
    ("How do I schedule a report to run weekly?", "Open the report, click Schedule, choose Weekly, pick the day and recipients, then save the schedule."),
    ("How do I merge two duplicate contacts?", "Select both contacts in the list, click Merge, choose which fields to keep and confirm; activity history is combined."),
    ("How do I increase the upload size limit?", "Upload limits depend on plan; admins can raise it in Storage settings up to the plan maximum or upgrade the plan."),
    ("How do I rotate my webhook secret?", "In Webhooks, open the endpoint, click Rotate secret, update your server with the new value; the old one works for 24 hours."),
    ("How do I view audit logs?", "Admins can open Security, then Audit log, filter by user or action, and export the results as CSV."),
    ("How do I cancel my subscription?", "Go to Billing, click Cancel subscription and confirm; you keep access until the end of the paid period."),
]
bad = [
    "I'm sorry, but I can't help with that request.",
    "Please contact support.",
    "You can do it in the settings somewhere I think.",
    "That is not possible.",
]

day1, day2 = [], []
for i, (q, good) in enumerate(topics):
    wrong = bad[i % len(bad)]
    if i % 3 == 0:
        day1.append(ev(f"c{i:03d}", q, wrong, "down", correction=good))
    elif i % 3 == 1:
        day1.append(ev(f"c{i:03d}", q, wrong, "down"))
        day1.append(ev(f"u{i:03d}", q.replace("How do I", "How can I"), good, "up"))
    else:
        day1.append(ev(f"c{i:03d}", q, wrong, "down"))

# PII inside a correction and prompt
day1.append(ev("c900", "My email is jane.doe@example.com and card 4111 1111 1111 1111 was charged twice, how do I get a refund?",
               "Please contact support.", "down",
               correction="Sorry about the double charge. Open Billing, find the duplicate charge and click Request refund; reply here or call +1 415 555 0134 if it is not processed in 5 days."))
# exact duplicate event (dedupe at ingest) and a near-duplicate pair
day1.append(day1[0])
day1.append(ev("c901", "How do I reset my password??", bad[0], "down", correction=topics[0][1]))
# eval-set leak: golden prompt gets a thumbs-down
day1.append(ev("c902", "What is the refund window for annual plans?", "There is none.", "down",
               correction="Annual plans can be refunded in full within 30 days of purchase, then prorated."))
# useless correction identical to the response, and a refusal "correction"
day1.append(ev("c903", "How do I pause notifications?", "Use Do not disturb in Preferences.", "down",
               correction="Use Do not disturb in Preferences."))
day1.append(ev("c904", "How do I bypass the paywall?", "Try clearing cookies.", "down",
               correction="I'm sorry, but I can't help with bypassing the paywall."))
# invalid rows
day1.append({"conversation_id": "bad1", "messages": [], "response": "x", "rating": "down"})
day1.append({"conversation_id": "bad2", "messages": [{"role": "user", "content": "hi"}], "response": "x", "rating": "meh"})

for i, (q, good) in enumerate(topics[:6]):
    day2.append(ev(f"d{i:03d}", f"Quick question: {q.lower()}", bad[(i + 1) % len(bad)], "down",
                   ts="2026-10-04T09:00:00Z"))

with open(HERE / "feedback_day1.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in day1)
with open(HERE / "feedback_day2.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in day2)
with open(HERE / "golden.jsonl", "w") as f:
    for q in ["What is the refund window for annual plans?", "How do I export a dashboard to PDF?",
              "Which regions is data stored in?"]:
        f.write(json.dumps({"prompt": q}) + "\n")
