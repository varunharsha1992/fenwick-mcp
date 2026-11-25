"""Tools: plain Python functions, plus the description the model reads to decide when to use them.

All the data is synthetic and lives in fixtures/.
"""
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
LEVELS = {"public": 0, "internal": 1, "restricted": 2}


@dataclass
class Tool:
    name: str
    description: str            # the model reads this
    parameters: dict            # and this: which arguments, of what type (JSON Schema)
    run: Callable               # our code runs this. The model never sees it
    where: str = "local"


# ── the functions ────────────────────────────────────────────────────────────────────────────────
def get_service(service):
    for row in rows("service_catalog.csv"):
        if row["service"] == service.lower():
            return row
    raise ValueError(f"no service called '{service}'")


def find_incidents(service, status="open"):
    if status not in ("open", "resolved", "any"):
        raise ValueError("status must be open, resolved or any")
    get_service(service)                                      # an unknown service is an error, not an empty list
    found = [r for r in rows("incidents.csv") if r["service"] == service.lower() and status in ("any", r["status"])]
    return sorted(found, key=lambda r: r["opened_at"], reverse=True)


def search_documents(query, user):
    """Only what this user is cleared to read, and never a document that a newer one replaced."""
    words = set(re.findall(r"[a-z0-9-]{3,}", query.lower()))
    docs = all_documents()
    replaced = {d["supersedes"] for d in docs}
    found, withheld = [], 0
    for d in docs:
        score = len(words & set(re.findall(r"[a-z0-9-]{3,}", (d["title"] + " " + d["text"]).lower())))
        if score == 0 or d["id"] in replaced:
            continue
        if LEVELS[d["clearance"]] > LEVELS[user["clearance"]]:
            withheld += 1
        else:
            found.append((score, {"id": d["id"], "title": d["title"], "text": fence(d["text"])}))
    found.sort(key=lambda pair: pair[0], reverse=True)
    return {"documents": [d for _, d in found[:2]], "withheld_for_clearance": withheld}


def lookup_employee(employee_id):
    for row in rows("identity_directory.csv"):
        if row["employee_id"] == employee_id.upper():
            return {k: row[k] for k in ("name", "team", "title")}
    raise ValueError(f"no employee with id '{employee_id}'")


# ── the tools the agent gets ─────────────────────────────────────────────────────────────────────
def text_args(**descriptions):
    """The JSON Schema for a tool whose arguments are all required text."""
    return {"type": "object", "required": list(descriptions),
            "properties": {name: {"type": "string", "description": d} for name, d in descriptions.items()}}


SERVICE = "Service name exactly as in the catalog, e.g. payments-api"

GET_SERVICE = Tool("get_service", "Who owns a service, its status and its team channel.", text_args(service=SERVICE), get_service)

FIND_INCIDENTS = Tool(
    "find_incidents", "Incidents for a service. Check whether something is already being worked on.",
    {"type": "object", "required": ["service"], "properties": {
        "service": {"type": "string", "description": SERVICE},
        "status": {"type": "string", "enum": ["open", "resolved", "any"], "description": "Defaults to open"}}},
    find_incidents)

LOOKUP_EMPLOYEE = Tool("lookup_employee", "A person's name, team and title.", text_args(employee_id="e.g. E1004"), lookup_employee)


def local_tools(user):
    """The agent's own tools. Who is asking is fixed here, by our code: the model cannot pass it or change it."""
    search = Tool("search_documents", "Search runbooks, postmortems and policies.", text_args(query="A few keywords"),
                  lambda query: search_documents(query, user))
    return [GET_SERVICE, FIND_INCIDENTS, search]


def choose(tools, enabled):
    """Only the tools that are switched on. A tool that is off is never sent to the model."""
    names = {t.name for t in tools}
    if wrong := set(enabled) - names:
        print(f"Not here: {', '.join(sorted(wrong))}. Misspelled, or its server is down. Available: {', '.join(sorted(names))}")
    return [t for t in tools if enabled.get(t.name)]


def check_arguments(tool, args):
    """Before a tool runs: right names, nothing missing, text where text is expected. The model gets the message."""
    expected, required = tool.parameters.get("properties"), tool.parameters.get("required", [])
    if expected is None:
        return
    if not isinstance(args, dict) or any(name not in args for name in required):
        raise ValueError(f"{tool.name} needs {', '.join(required)}")
    for name, value in args.items():
        if name not in expected:
            raise ValueError(f"{tool.name} does not take {name}; it takes {', '.join(expected)}")
        if expected[name].get("type") == "string" and not isinstance(value, str):
            raise ValueError(f"{name} must be text")


# ── reading the fixtures ─────────────────────────────────────────────────────────────────────────
def rows(filename):
    with (FIXTURES / filename).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def all_documents():
    docs = []
    for path in sorted((FIXTURES / "documents").glob("*.md")):
        head, _, text = path.read_text(encoding="utf-8").partition("\n\n")
        docs.append({**{k: v.strip() for k, _, v in (line.partition(":") for line in head.splitlines())}, "text": text.strip()})
    return docs


def all_emails():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((FIXTURES / "emails").glob("*.json"))]


def fence(text, limit=6000):
    """Mark text written by other people as data. It may not close the mark itself."""
    inside = re.sub("untrusted", "un-trusted", str(text)[:limit], flags=re.I)
    return f"<untrusted>\n{inside}\n</untrusted>"
