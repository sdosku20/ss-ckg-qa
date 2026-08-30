"""
ikgqa.baselines.snapquery
=========================

A client for the SnapQuery service, and the adaptation from what it returns to
what the thesis metric needs.

The protocol was measured on the CHIL server on 30 August 2026 rather than read
off a document, and it is not one request per question:

  POST /chat/          {"query": ..., "session_id": ...}
  POST /chat/continue  {"session_id": ...}          repeatedly

``/chat/`` runs the agent only until it wants a tool and then returns; each
``/chat/continue`` advances it without carrying any user input. A turn is
finished when ``cypher`` and ``data`` come back non-null. Both fields are
required by the service; ``session_id`` is issued by the *client*, which is why
an evaluation can start a guaranteed-clean session per question instead of
scraping a handle out of the previous response.

Three design points that are easy to get wrong:

**A failure is an outcome, not an exception.** The comparator is a live service
that can return 500, stall, or spend its turn budget without producing a query.
An evaluation that crashes on those has no numbers; one that silently drops them
reports a flattering average over the questions that happened to work. Every
exchange therefore ends with an ``Outcome``, and the failures are countable.

**An unexecuted tool call is detectable.** On 30 August the deployed planner
returned its tool call as raw text that the backend never executed. That is a
property of the deployment rather than of the question, so it is recognised and
labelled instead of being counted as "the model declined to answer".

**Rows are not a subgraph.** SnapQuery returns result rows with no explicit
nodes or edges. If a row carries a node identifier the induced subgraph is a
lookup; if it carries only values, recovering nodes means matching text, which
on this graph is ambiguous by construction -- one patient's 15,810 nodes share
614 distinct texts. ``entities_from_rows`` reports which of the two happened so
the ambiguity is visible in the results rather than assumed away.
"""

from __future__ import annotations

import dataclasses
import json
import re
import socket
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_BASE = "http://localhost:8002"

#: Keys a result row might use to name a graph node, most specific first.
ID_KEYS: Tuple[str, ...] = ("uid", "elementId", "element_id", "nodeId", "node_id", "id")

#: A tool call the backend emitted as text instead of executing. Both the
#: JSON-in-``<tool_call>`` convention and the ``<function=...>`` one are matched,
#: since which appears depends on the planner model that happens to be deployed.
UNPARSED_TOOL_CALL = re.compile(r"<tool_call>|<function=|</think>")


class Outcome:
    """How an exchange ended. Strings, so they land in a results table as-is."""

    ANSWERED = "answered"          # cypher and rows came back
    NO_QUERY = "no_query"          # ran to the turn cap without a query
    STALLED = "stalled"            # repeated itself; advancing further is futile
    TOOL_CALL_LEAKED = "tool_call_leaked"   # the deployment never ran its own tool
    HTTP_ERROR = "http_error"      # the service answered with a failure status
    TRANSPORT_ERROR = "transport_error"     # no answer at all


@dataclasses.dataclass(frozen=True)
class Turn:
    """One HTTP call, kept whether it succeeded or not."""

    endpoint: str
    status: Optional[int]
    seconds: float
    answer: str = ""
    cypher: Optional[str] = None
    rows: Optional[List[dict]] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200

    @property
    def leaked_tool_call(self) -> bool:
        return bool(UNPARSED_TOOL_CALL.search(self.answer))


@dataclasses.dataclass(frozen=True)
class Exchange:
    """Everything one question produced, including how it ended."""

    question: str
    session_id: str
    turns: Tuple[Turn, ...]
    outcome: str

    @property
    def answered(self) -> bool:
        return self.outcome == Outcome.ANSWERED

    @property
    def cypher(self) -> Optional[str]:
        for turn in reversed(self.turns):
            if turn.cypher:
                return turn.cypher
        return None

    @property
    def rows(self) -> List[dict]:
        for turn in reversed(self.turns):
            if turn.rows:
                return turn.rows
        return []

    @property
    def seconds(self) -> float:
        return round(sum(turn.seconds for turn in self.turns), 2)

    def summary(self) -> str:
        return (f"{self.outcome:16s} {len(self.turns)} turns  {self.seconds:6.1f}s  "
                f"{len(self.rows)} rows  {self.question[:48]!r}")


def http_transport(url: str, payload: dict, timeout: float) -> Tuple[Optional[int], Any, str]:
    """POST JSON, returning (status, parsed body or text, error).

    Separated from the client so tests can drive the whole protocol without a
    server, which is the only way the harness could be written at all while the
    deployment was returning 500.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body, status = response.read().decode("utf-8", "replace"), response.status
    except urllib.error.HTTPError as exc:
        body, status = exc.read().decode("utf-8", "replace"), exc.code
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return None, None, str(exc)

    try:
        return status, json.loads(body), ""
    except json.JSONDecodeError:
        return status, body, ""       # a 500 arrives as plain "Internal Server Error"


@dataclasses.dataclass
class SnapQueryClient:
    """Drives one question through the service to a query and rows, or to a
    labelled failure.

    ``max_turns`` is a cap, not a target: an agent that keeps reasoning without
    converging must not hold up an evaluation, and hitting the cap is itself a
    reportable outcome.
    """

    base: str = DEFAULT_BASE
    timeout: float = 300.0
    max_turns: int = 8
    #: Consecutive identical answers before the exchange is abandoned. Two is
    #: too eager -- an agent may legitimately restate itself once between tool
    #: calls -- and waiting for the full turn cap wastes real seconds against a
    #: live service.
    stall_repeats: int = 3
    transport: Callable[[str, dict, float], Tuple[Optional[int], Any, str]] = http_transport

    def _call(self, path: str, payload: dict) -> Turn:
        started = time.time()
        status, body, error = self.transport(self.base + path, payload, self.timeout)
        seconds = round(time.time() - started, 2)

        if error:
            return Turn(path, None, seconds, error=error)
        if not isinstance(body, dict):
            return Turn(path, status, seconds, error=str(body)[:200])

        rows = body.get("data")
        return Turn(
            endpoint=path,
            status=status,
            seconds=seconds,
            answer=body.get("answer") or "",
            cypher=body.get("cypher"),
            rows=list(rows) if isinstance(rows, list) else None,
        )

    def ask(self, question: str, session_id: str = "") -> Exchange:
        session = session_id or str(uuid.uuid4())
        turns: List[Turn] = []

        turn = self._call("/chat/", {"query": question, "session_id": session})
        turns.append(turn)
        repeats = 1

        def ended(outcome: str) -> Exchange:
            return Exchange(question, session, tuple(turns), outcome)

        while True:
            if turn.status is None:
                return ended(Outcome.TRANSPORT_ERROR)
            if not turn.ok:
                return ended(Outcome.HTTP_ERROR)
            if turn.rows is not None and turn.cypher:
                return ended(Outcome.ANSWERED)

            # A deployment that never executes its own tool call will repeat
            # itself forever, so the two exits are checked before spending
            # another turn -- and are reported apart, because one is a property
            # of the service and the other of the question.
            if repeats >= self.stall_repeats:
                return ended(Outcome.TOOL_CALL_LEAKED if turn.leaked_tool_call
                             else Outcome.STALLED)
            if len(turns) >= self.max_turns:
                return ended(Outcome.TOOL_CALL_LEAKED if turn.leaked_tool_call
                             else Outcome.NO_QUERY)

            previous = turn.answer
            turn = self._call("/chat/continue", {"session_id": session})
            turns.append(turn)
            repeats = repeats + 1 if (turn.ok and turn.answer and turn.answer == previous) else 1


def entities_from_rows(rows: Sequence[dict],
                       id_keys: Sequence[str] = ID_KEYS) -> Tuple[List[str], str]:
    """Node identifiers named by the rows, and how they were obtained.

    Returns ``(ids, "identity")`` when the rows carry a node identifier, and
    ``([], "by_value")`` when they do not. The second case is not a failure of
    this function: it means SnapQuery's answer names values rather than nodes,
    and recovering nodes requires matching text against the graph -- ambiguous
    here, because many nodes share one description. Callers must handle that
    explicitly, which is why the strategy is returned rather than logged.
    """
    if not rows:
        return [], "empty"

    first = rows[0]
    if not isinstance(first, dict):
        return [], "by_value"

    for key in id_keys:
        if key in first:
            found = [str(row[key]) for row in rows
                     if isinstance(row, dict) and row.get(key) is not None]
            if found:
                return found, "identity"
    return [], "by_value"


def outcome_counts(exchanges: Sequence[Exchange]) -> Dict[str, int]:
    """How the exchanges ended, for a results table.

    Reported alongside recall so that a high average over a handful of
    successful questions cannot be mistaken for a high average over all of them.
    """
    counts: Dict[str, int] = {}
    for exchange in exchanges:
        counts[exchange.outcome] = counts.get(exchange.outcome, 0) + 1
    return dict(sorted(counts.items()))
