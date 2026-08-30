"""
Observe one complete SnapQuery exchange, and bring home only what is safe.

    # SERVER
    python3 experiments/snapquery_probe.py --discover
    python3 experiments/snapquery_probe.py --ask "which drugs were given to kidney transplant patients?"
    python3 experiments/snapquery_probe.py --reply "yes" --session <id>

The thesis compares PCST retrieval against SnapQuery, and not one real SnapQuery
response has ever been observed. Everything the baseline harness needs -- how a
session is held, whether slot-filling can be answered by a script, what the
result rows look like, whether they name graph entities -- is currently a guess
copied out of an architecture document. This script replaces the guess with a
recording, so that the harness itself can then be written offline.

Two files per run, and the split is the point:

  raw/<stamp>.json    every byte of the exchange, mode 600. STAYS ON THE SERVER.
  safe/<stamp>.json   the same exchange with every value replaced by its type
                      and length. This is what writing a harness actually needs
                      -- which keys, what nesting, is there a session id -- and
                      it carries no patient data.

Read the safe file before copying it anywhere regardless. Redaction is a
mechanism, not a guarantee, and an error message can quote anything.

Standard library only: it must run before anything is installed, under the
server's Python 3.10.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE = "http://localhost:8002"

# Paths worth trying blind. /openapi.json is the jackpot: FastAPI publishes the
# full request and response schema there, which answers most of what this script
# exists to ask without sending a single question through the model.
PROBE_PATHS = (
    "/",
    "/docs",
    "/openapi.json",
    "/health",
    "/healthz",
    "/chat/",
    "/chat/continue",
    "/api/chat/",
    "/v1/chat/",
)

# Keys whose values describe the protocol rather than the patient. Everything
# else is masked. Kept deliberately short: when in doubt, a value is data.
#
# "loc" and "msg" are here because of a real failure: a 422 names the fields it
# wanted in "loc", and masking those turned the single most useful reply the
# service can give -- here is the field you got wrong -- into "<str len=4>".
# A validation error is about the request, which we wrote, not about a patient.
STRUCTURAL_KEYS = frozenset({
    "status", "state", "step", "role", "type", "action", "next_action",
    "finished", "done", "complete", "requires_confirmation", "confirmed",
    "model", "tool", "tool_name", "node", "phase", "kind", "event",
    "loc", "msg",
})

# Keys holding a generated database query. Worth keeping -- it is the clearest
# statement of what the comparator actually did -- but its string literals can
# name a patient, so the literals go and the structure stays.
QUERY_KEYS = re.compile(r"cypher|query|sql|statement", re.IGNORECASE)

QUERY_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")


def mask_query(text: str) -> str:
    """Keep a generated query's shape, drop its literals."""
    return QUERY_LITERAL.sub("'<literal>'", text)


def redact(value, key: str = ""):
    """Replace every value with a description of it, keeping all key names.

    Key names are schema and are safe to carry off the server; values are not.
    Lists collapse to a length plus one redacted sample, because a thousand
    result rows teach nothing the first row does not.
    """
    if isinstance(value, dict):
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        # A structural list is short and is itself the message -- ["body",
        # "center_id"] means nothing if only its first element survives.
        if key in STRUCTURAL_KEYS and len(value) <= 8:
            return [redact(item, key) for item in value]
        return {
            "__list_len__": len(value),
            "__sample__": redact(value[0], key) if value else None,
        }
    if isinstance(value, str):
        if QUERY_KEYS.search(key):
            return {"__query_masked__": mask_query(value)}
        if key in STRUCTURAL_KEYS and len(value) <= 40:
            return value
        return f"<str len={len(value)}>"
    if isinstance(value, bool) or value is None:
        return value
    return f"<{type(value).__name__}>"


def request(url: str, method: str = "GET", payload=None, token: str = "",
            timeout: float = 60.0) -> dict:
    """One HTTP call, recorded whether it succeeds or fails."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:            # 4xx and 5xx still inform
        body = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return {"url": url, "method": method, "status": None,
                "error": str(exc), "seconds": round(time.time() - started, 2)}

    record = {"url": url, "method": method, "status": status,
              "seconds": round(time.time() - started, 2)}
    try:
        record["json"] = json.loads(body)
    except json.JSONDecodeError:
        record["text"] = body[:4000]
    return record


def discover(base: str, token: str) -> dict:
    """What is listening, and what does it say it accepts?"""
    found = {"base": base, "probes": []}
    for path in PROBE_PATHS:
        result = request(base + path, timeout=10.0, token=token)
        found["probes"].append({
            "path": path,
            "status": result.get("status"),
            "error": result.get("error"),
            "seconds": result.get("seconds"),
        })
        outcome = result.get("status") or result.get("error")
        print(f"  {path:20s} {str(outcome)[:60]}")

    spec = request(base + "/openapi.json", timeout=15.0, token=token)
    if spec.get("status") == 200 and isinstance(spec.get("json"), dict):
        found["openapi"] = spec["json"]
        print("\n  endpoints the service declares:")
        for path, ops in sorted(spec["json"].get("paths", {}).items()):
            for method, op in ops.items():
                if method.lower() in ("get", "post", "put", "patch", "delete"):
                    print(f"    {method.upper():6s} {path:34s} {op.get('summary', '')}")
    else:
        print("\n  no /openapi.json -- the request shape has to be found by hand")
    return found


def find_session_ids(value, key: str = "", found=None) -> dict:
    """Pull out anything that looks like a session handle.

    The redactor masks these along with everything else, but continuing an
    exchange needs one, so they are surfaced to the terminal. A session handle
    is a protocol token, not clinical data.
    """
    found = {} if found is None else found
    if isinstance(value, dict):
        for k, v in value.items():
            find_session_ids(v, k, found)
    elif isinstance(value, list):
        for item in value[:5]:
            find_session_ids(item, key, found)
    elif isinstance(value, (str, int)) and re.search(r"session|thread|conversation|chat_id", key, re.I):
        found[key] = value
    return found


def show_schema(schema, defs: dict, indent: str = "   ", depth: int = 0) -> None:
    """Print one OpenAPI schema's fields, resolving $ref against components."""
    if depth > 3 or not isinstance(schema, dict):
        return
    if "$ref" in schema:
        show_schema(defs.get(schema["$ref"].split("/")[-1], {}), defs, indent, depth)
        return
    required = set(schema.get("required", []))
    for name, prop in (schema.get("properties") or {}).items():
        kind = prop.get("type") or prop.get("$ref", "").split("/")[-1] or "any"
        if "anyOf" in prop:
            kind = "|".join(
                a.get("type", a.get("$ref", "").split("/")[-1]) for a in prop["anyOf"]
            )
        print(f"{indent}{'*' if name in required else ' '} {name}: {kind}")
        if "$ref" in prop or prop.get("items"):
            show_schema(prop.get("items") or prop, defs, indent + "    ", depth + 1)


def report_schema(spec: dict) -> None:
    """What the service says it accepts and returns, per endpoint.

    Worth reading before sending anything: it names the request fields, which
    otherwise have to be guessed one 422 at a time.
    """
    defs = spec.get("components", {}).get("schemas", {})
    for route, ops in sorted(spec.get("paths", {}).items()):
        for method, op in ops.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            print(f"\n=== {method.upper()} {route} ===")
            body = (op.get("requestBody", {}).get("content", {})
                    .get("application/json", {}).get("schema"))
            if body:
                print(" request:")
                show_schema(body, defs)
            for code, resp in (op.get("responses") or {}).items():
                sch = resp.get("content", {}).get("application/json", {}).get("schema")
                if sch:
                    print(f" response {code}:")
                    show_schema(sch, defs)


def latest_discovery() -> dict:
    """The most recent saved discovery, so the spec need not be fetched twice."""
    files = sorted((Path.home() / "snapquery" / "raw").glob("*-discover.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8")).get("openapi", {})


def outdir() -> Path:
    root = Path.home() / "snapquery"
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "safe").mkdir(parents=True, exist_ok=True)
    return root


def save(record: dict, label: str) -> tuple:
    root = outdir()
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + label
    raw_path = root / "raw" / f"{stamp}.json"
    safe_path = root / "safe" / f"{stamp}.json"

    raw_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    os.chmod(raw_path, 0o600)
    safe_path.write_text(json.dumps(redact(record), indent=2, ensure_ascii=False), encoding="utf-8")
    os.chmod(safe_path, 0o600)
    return raw_path, safe_path


def selftest() -> int:
    """Check the redactor against a fabricated response, offline, before it matters."""
    pretend = {
        "session_id": "abc-123-def",
        "status": "awaiting_confirmation",
        "message": "Shall I run this over the last 12 months?",
        "cypher": "MATCH (p:SubjectPseudoIdentifier {hasIdentifier: 'PATIENT-0042'}) RETURN p",
        "rows": [{"drug": "Tacrolimus", "dose": 5.0}, {"drug": "Mycophenolate", "dose": 720.0}],
        "confirmed": False,
    }
    safe = redact(pretend)
    flat = json.dumps(safe)
    for secret in ("abc-123-def", "PATIENT-0042", "Tacrolimus", "12 months"):
        if secret in flat:
            print(f"FAIL: {secret!r} survived redaction")
            return 1
    assert safe["status"] == "awaiting_confirmation", "structural key was masked"
    assert safe["confirmed"] is False, "boolean was masked"
    assert safe["rows"]["__list_len__"] == 2, "row count was lost"
    assert "SubjectPseudoIdentifier" in flat, "query structure was lost"

    # A 422 is the service telling us which field we got wrong. Masking that
    # wastes the round trip, and it describes our own request, not a patient.
    rejection = {"detail": [
        {"type": "missing", "loc": ["body", "center_id"], "msg": "Field required",
         "input": {"message": "which drugs..."}},
    ]}
    safe_rejection = redact(rejection)
    named = safe_rejection["detail"]["__sample__"]["loc"]
    assert named == ["body", "center_id"], f"validation error lost the field name: {named}"
    assert "which drugs" not in json.dumps(safe_rejection), "echoed input survived"

    print(json.dumps(safe, indent=2))
    print("\nselftest OK: values masked, keys, query structure and 422 fields kept")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record one SnapQuery exchange safely.")
    parser.add_argument("--base", default=os.environ.get("SNAPQUERY_BASE", DEFAULT_BASE))
    parser.add_argument("--token", default=os.environ.get("SNAPQUERY_TOKEN", ""))
    parser.add_argument("--discover", action="store_true",
                        help="what is listening, and what does it accept")
    parser.add_argument("--ask", metavar="QUESTION", help="start one exchange")
    parser.add_argument("--reply", metavar="TEXT",
                        help="answer a slot-filling or confirmation turn")
    parser.add_argument("--session", default="", help="session id from a previous turn")
    parser.add_argument("--endpoint", default="/chat/",
                        help="path to POST to (check --discover first)")
    parser.add_argument("--continue-endpoint", default="/chat/continue")
    parser.add_argument("--field", default="message",
                        help="request field holding the user text")
    parser.add_argument("--session-field", default="session_id")
    parser.add_argument("--schema", action="store_true",
                        help="request and response fields, from the saved spec")
    parser.add_argument("--selftest", action="store_true", help="check the redactor offline")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.discover:
        print(f"probing {args.base}")
        record = discover(args.base, args.token)
        raw, safe = save(record, "discover")
        if record.get("openapi"):
            report_schema(record["openapi"])
        print(f"\nraw  (server only): {raw}")
        print(f"safe (copyable)   : {safe}")
        return 0

    if args.schema:
        spec = latest_discovery()
        if not spec:
            print("no saved spec -- run --discover first")
            return 1
        report_schema(spec)
        return 0

    text = args.ask or args.reply
    if not text:
        parser.error("give --discover, --ask, --reply or --selftest")

    payload = {args.field: text}
    if args.session:
        payload[args.session_field] = args.session
    path = args.continue_endpoint if args.reply else args.endpoint

    print(f"POST {args.base}{path}")
    record = request(args.base + path, method="POST", payload=payload, token=args.token)
    record["sent"] = payload
    raw, safe = save(record, "ask" if args.ask else "reply")

    print(f"status {record.get('status')} in {record.get('seconds')}s")
    print("\nshape of the response (safe to read aloud):")
    body = record.get("json", record.get("text", {}))
    print(json.dumps(redact(body), indent=2)[:3000])

    handles = find_session_ids(body)
    if handles:
        print("\nsession handles, for --session on the next turn:")
        for key, value in handles.items():
            print(f"  {key} = {value}")

    print(f"\nraw  (server only): {raw}")
    print(f"safe (copyable)   : {safe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
