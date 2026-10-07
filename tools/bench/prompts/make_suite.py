"""Builds husky_suite.json: Husky's 16 published task types (our own prompts) plus browser-agent DOM tasks."""
import json
from pathlib import Path

CODE = '''def load_users(path):
    users = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            user = {"id": int(parts[0]), "name": parts[1], "email": parts[2]}
            users.append(user)
    return users


def find_user(users, user_id):
    for user in users:
        if user["id"] == user_id:
            return user
    return None


def active_emails(users, active_ids):
    emails = []
    for user in users:
        if user["id"] in active_ids:
            emails.append(user["email"])
    return emails
'''
RECORD = {"order_id": "A-10293", "customer": {"name": "Priya Raman", "email": "priya@example.com", "tier": "gold"},
          "items": [{"sku": "KB-201", "name": "Mechanical keyboard", "qty": 1, "price": 89.0},
                    {"sku": "MS-044", "name": "Wireless mouse", "qty": 2, "price": 24.5}],
          "shipping": {"method": "express", "address": "14 Park Lane, Kolkata 700016", "eta_days": 2},
          "payment": {"method": "card", "last4": "4421", "status": "captured"}, "notes": ""}
SQL = '''CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    cust_id INTEGER NOT NULL REFERENCES customers(id),
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    total_cents INTEGER NOT NULL
);

SELECT o.id, o.cust_id, c.name, o.total_cents
FROM orders o
JOIN customers c ON c.id = o.cust_id
WHERE o.created_at > now() - interval '30 days'
ORDER BY o.cust_id, o.created_at DESC;

CREATE INDEX orders_cust_id_idx ON orders (cust_id);
'''
TYPOS = ("Thier new release was suposed to ship on tuesday but the team decieded to wait untill the regresion tests "
         "where green. Alot of customers had allready asked about the date, so the support team prepaired a short "
         "anouncement. In adition, the docs needed updateing becuase two endpoints changed there response format. "
         "Everyone agreed it was better to be late then to brake existing integrations.")
INVOICE = ("INVOICE #INV-2291\nDate: 3 Oct 2026\nBill to: Northwind Traders, 221 Lake Rd, Pune\n"
           "1 x Annual support plan ........ 1,200.00\n3 x Onsite training day ....... 2,250.00\n"
           "12 x Licence seat ............... 1,440.00\nSubtotal 4,890.00\nGST 18% 880.20\nTotal due 5,770.20 INR\n"
           "Due: 2 Nov 2026. Pay to HDFC 50100234.")
CSV = ("region,quarter,revenue,deals\nNorth,Q1,120400,31\nNorth,Q2,131900,35\nSouth,Q1,98200,27\nSouth,Q2,104500,29\n"
       "East,Q1,87600,22\nEast,Q2,93300,25\nWest,Q1,142800,38\nWest,Q2,150100,41")
METRICS = ("Server metrics this week: api-1 averaged 41% CPU and 2.1 GB memory with 3 restarts; api-2 averaged 63% CPU "
           "and 2.8 GB with 0 restarts; worker-1 averaged 88% CPU and 5.2 GB with 7 restarts; worker-2 averaged 79% CPU "
           "and 4.9 GB with 4 restarts; db-1 averaged 35% CPU and 11.4 GB with 0 restarts.")
TRANSCRIPT = ("Speaker A: We need the migration plan by Friday. Speaker B: I can draft it, but I need the schema diff "
              "first. Speaker A: Ravi said he would send it today. Speaker B: Then I can share a draft Thursday. ") * 6
TONE = ("hey, the report you sent is wrong again. the numbers in section 3 dont match the dashboard and i told you last "
        "week to double check. fix it by tomorrow, we cant send this to the client like this.")
NOTES = ("Sync 6 Oct: Ana will finish the onboarding redesign mockups by Wed. Bilal to fix the flaky login test, blocked "
         "on staging creds from ops. Decided to drop the CSV export from v2. Chen to write release notes once QA signs "
         "off. Open question: do we support SSO in the free tier? Ana to ask sales. Next sync Monday.")
THREAD = ("From: Meera\nHi team, can we move the vendor review to next week? Finance needs more time.\n\n"
          "From: Tom\nNext week works except Tuesday. Also, should we invite legal?\n\n"
          "From: Meera\nYes, please loop in legal. Wednesday 3pm?\n\n"
          "From: Julia\nWednesday is fine for me. I'll book the room. Do we have the updated pricing sheet?")
CALL = ("Call with Helix Bio, 5 Oct. Attendees: Sam (us), Dr. Okafor and Lena (Helix). Helix runs 40 lab instruments "
        "and loses about 6 hours a week reconciling results by hand. They want an integration with their LIMS by Q1. "
        "Budget approved for a pilot, not full rollout. Concerns: data residency in the EU, and validation paperwork "
        "for audits. Next steps: we send a pilot proposal by 12 Oct; Lena shares LIMS API docs; follow-up call "
        "19 Oct. ") * 3
DOC = ("The Kestrel project migrated its storage layer from a single PostgreSQL primary to a sharded setup in March. "
       "Write latency at p99 fell from 180 ms to 45 ms, but cross-shard transactions now need a two-phase commit, "
       "which the billing team reports doubled the code needed for refunds. Read replicas lag up to 2 s under "
       "peak load. ") * 6
FORM = ['[200] <input name="full_name" placeholder="Full name">', '[201] <input name="email" type="email">',
        '[202] <input name="phone" type="tel">',
        '[203] <select name="city"><option>Kolkata</option><option>Pune</option></select>',
        '[204] <textarea name="notes"></textarea>', '[205] <button type="submit">Place order</button>']
AGENT_SYSTEM = ('You control a web browser. Respond with exactly one action as XML: '
                '<function name="click"><param name="index">N</param></function> or '
                '<function name="type"><param name="index">N</param><param name="text">...</param></function> or '
                '<function name="extract"><param name="data">...</param></function>.')


def dom(n_items, extra):
    cards = [f'[{i}] <a href="/product/{i}" class="card">Product {i} - Rs {499 + 37 * i}</a>' for i in range(1, n_items)]
    return "\n".join(cards + extra)


def user(text):
    return [{"role": "user", "content": text}]


def agent(page, task):
    return [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": f"Page elements:\n{page}\n\nTask: {task}"}]


SUITE = [
    ("function_edit", "edit", user("Add type hints and a docstring to every function. Return the full updated file only.\n\n" + CODE)),
    ("json_add_field", "edit", user('Add "gift_wrap": true inside "shipping" and set "notes" to "Leave at reception". '
                                    "Return the full JSON only.\n\n" + json.dumps(RECORD, indent=2))),
    ("sql_rename", "edit", user("Rename the column cust_id to customer_id everywhere. Return the full SQL only.\n\n" + SQL)),
    ("write_function", "write", user("Write a Python function that parses an ISO-8601 duration like 'P3DT4H12M' into total seconds, with tests.")),
    ("fix_typos", "edit", user("Fix the spelling and grammar. Return only the corrected text.\n\n" + TYPOS)),
    ("invoice_to_json", "transform", user("Convert this invoice to JSON with fields invoice_number, date, bill_to, line_items, "
                                          "subtotal, tax, total, currency, due_date.\n\n" + INVOICE)),
    ("csv_to_table", "transform", user("Convert this CSV to a Markdown table.\n\n" + CSV)),
    ("data_to_table", "transform", user("Put these metrics in a Markdown table with columns host, cpu, memory, restarts.\n\n" + METRICS)),
    ("repeated_transcript", "edit", user("Clean up this transcript: remove repetition and return it as a short dialogue.\n\n" + TRANSCRIPT)),
    ("tone_rewrite", "write", user("Rewrite this message to be polite and constructive.\n\n" + TONE)),
    ("short_email", "write", user("Write a short email to a client confirming our meeting on Thursday at 11am.")),
    ("project_plan", "write", user("Write a 6-week project plan for launching a mobile app beta.")),
    ("notes_to_todos", "transform", user("Turn these meeting notes into a to-do list with owners.\n\n" + NOTES)),
    ("reply_thread", "write", user("Write a reply to this email thread as Julia, attaching the pricing sheet.\n\n" + THREAD)),
    ("call_summary", "transform", user("Summarize this call in 5 bullet points with next steps.\n\n" + CALL)),
    ("question_over_document", "qa", user(DOC + "\n\nQuestion: what got worse after the migration, and why?")),
    ("dom_click", "agent", agent(dom(60, ['[60] <input type="search" placeholder="Search">', '[61] <a href="/cart">Cart (2)</a>',
                                          '[62] <button>Checkout</button>']), "open the cart")),
    ("dom_type", "agent", agent(dom(40, FORM), "fill in the name field with Farzan Aman Khan")),
    ("dom_extract", "agent", agent(dom(80, []), "extract the names and prices of products 10 to 15")),
    ("dom_long", "agent", agent(dom(400, FORM), "choose Pune as the city")),
]

if __name__ == "__main__":
    out = Path(__file__).with_name("husky_suite.json")
    out.write_text(json.dumps([{"name": n, "kind": k, "messages": m} for n, k, m in SUITE], indent=1))
    print(len(SUITE), "prompts ->", out)
