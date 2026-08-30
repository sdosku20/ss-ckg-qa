"""
The SnapQuery client, driven entirely through a fake transport.

Every scenario here was observed on the CHIL server on 30 August 2026 or is a
direct consequence of the measured contract. Writing them as fixtures is what
allowed the harness to be finished while the deployment itself was returning
500: when the service is fixed, only the transport changes.
"""

from __future__ import annotations

import pytest

from ikgqa.baselines.snapquery import (
    Exchange,
    Outcome,
    SnapQueryClient,
    Turn,
    entities_from_rows,
    outcome_counts,
)


def scripted(*responses):
    """A transport that returns the given (status, body) pairs in order."""
    remaining = list(responses)
    sent = []

    def transport(url, payload, timeout):
        sent.append((url, payload))
        status, body = remaining.pop(0) if remaining else (500, "Internal Server Error")
        return status, body, ""

    transport.sent = sent
    return transport


def answering(cypher="MATCH (d:Drug) RETURN d.name", rows=None):
    return 200, {"answer": "Here are the results.", "cypher": cypher,
                 "data": rows if rows is not None else [{"uid": "n1", "drug": "X"}],
                 "status": "ok"}


THINKING = (200, {"answer": "Need schema lookup.", "cypher": None,
                  "data": None, "status": "ok"})

LEAKED = (200, {"answer": "</think>\n<tool_call>\n<function=snapquery_schema_lookup>\n",
                "cypher": None, "data": None, "status": "ok"})


# --- the happy path ---------------------------------------------------------

def test_a_question_reaches_a_query_through_continue_calls():
    client = SnapQueryClient(transport=scripted(THINKING, THINKING, answering()))
    exchange = client.ask("which drugs?")

    assert exchange.outcome == Outcome.ANSWERED
    assert exchange.answered
    assert exchange.cypher == "MATCH (d:Drug) RETURN d.name"
    assert len(exchange.rows) == 1
    assert len(exchange.turns) == 3


def test_the_first_call_carries_the_question_and_the_rest_carry_only_the_session():
    transport = scripted(THINKING, answering())
    client = SnapQueryClient(transport=transport)
    exchange = client.ask("which drugs?")

    first_url, first_payload = transport.sent[0]
    assert first_url.endswith("/chat/")
    assert first_payload == {"query": "which drugs?", "session_id": exchange.session_id}

    second_url, second_payload = transport.sent[1]
    assert second_url.endswith("/chat/continue")
    # Measured: /chat/continue takes session_id alone. Sending a query too is
    # what produced the first 500.
    assert second_payload == {"session_id": exchange.session_id}


def test_each_question_gets_a_fresh_session_unless_one_is_given():
    client = SnapQueryClient(transport=scripted(answering(), answering()))
    first = client.ask("a")
    second = client.ask("b")
    assert first.session_id != second.session_id

    client = SnapQueryClient(transport=scripted(answering()))
    assert client.ask("a", session_id="fixed").session_id == "fixed"


# --- failures are outcomes, not exceptions ----------------------------------

def test_a_500_is_recorded_rather_than_raised():
    client = SnapQueryClient(transport=scripted(THINKING, (500, "Internal Server Error")))
    exchange = client.ask("which drugs?")

    assert exchange.outcome == Outcome.HTTP_ERROR
    assert not exchange.answered
    assert exchange.turns[-1].error == "Internal Server Error"
    assert exchange.rows == []


def test_no_response_at_all_is_a_transport_error():
    def dead(url, payload, timeout):
        return None, None, "connection refused"

    exchange = SnapQueryClient(transport=dead).ask("which drugs?")
    assert exchange.outcome == Outcome.TRANSPORT_ERROR
    assert exchange.turns[-1].status is None


def test_an_agent_that_never_converges_stops_at_the_turn_cap():
    # Different text every turn, so this is genuine non-convergence rather than
    # the stall the next test covers.
    wandering = [(200, {"answer": f"thinking step {i}", "cypher": None,
                        "data": None, "status": "ok"}) for i in range(10)]
    client = SnapQueryClient(max_turns=4, transport=scripted(*wandering))
    exchange = client.ask("which drugs?")

    assert exchange.outcome == Outcome.NO_QUERY
    assert len(exchange.turns) == 4, "the cap must bound the work, not merely label it"


def test_repeating_itself_ends_the_exchange_early():
    # The same answer three times: continuing is not advancing anything, and
    # burning the remaining turns on a live service costs real seconds.
    client = SnapQueryClient(max_turns=8, transport=scripted(*[THINKING] * 8))
    exchange = client.ask("which drugs?")

    assert exchange.outcome == Outcome.STALLED
    assert len(exchange.turns) == 3


def test_one_repeated_answer_does_not_end_a_working_exchange():
    # An agent may restate itself once between tool calls. Ending there would
    # discard exchanges that were about to succeed.
    client = SnapQueryClient(transport=scripted(THINKING, THINKING, answering()))
    assert client.ask("which drugs?").outcome == Outcome.ANSWERED


def test_an_unexecuted_tool_call_is_labelled_as_such():
    # Measured on CHIL: the planner's tool call came back as text and the
    # backend never ran it. That is a property of the deployment, and counting
    # it as "the model produced no query" would misattribute the failure.
    client = SnapQueryClient(max_turns=3, transport=scripted(*[LEAKED] * 3))
    exchange = client.ask("which drugs?")

    assert exchange.outcome == Outcome.TOOL_CALL_LEAKED
    assert exchange.turns[0].leaked_tool_call


def test_ordinary_reasoning_is_not_mistaken_for_a_leaked_tool_call():
    assert not Turn("/chat/", 200, 1.0, answer="Which time window?").leaked_tool_call


# --- rows to entities -------------------------------------------------------

def test_rows_naming_nodes_give_identity():
    ids, how = entities_from_rows([{"uid": "n1", "drug": "X"}, {"uid": "n2", "drug": "Y"}])
    assert how == "identity"
    assert ids == ["n1", "n2"]


def test_rows_of_bare_values_report_that_nodes_must_be_matched_by_text():
    ids, how = entities_from_rows([{"drug_name": "X", "administrations": 12}])
    assert how == "by_value"
    assert ids == []


def test_no_rows_is_distinguished_from_rows_without_identity():
    # These need opposite responses: one means the query returned nothing, the
    # other means the comparison itself needs a mapping step.
    assert entities_from_rows([])[1] == "empty"
    assert entities_from_rows([{"drug": "X"}])[1] == "by_value"


def test_the_first_matching_id_key_wins_and_nulls_are_skipped():
    rows = [{"uid": "n1", "id": "other"}, {"uid": None, "id": "other"}]
    ids, how = entities_from_rows(rows)
    assert (ids, how) == (["n1"], "identity")


# --- reporting --------------------------------------------------------------

def test_outcomes_are_counted_so_a_flattering_average_is_impossible():
    exchanges = [
        Exchange("a", "s1", (), Outcome.ANSWERED),
        Exchange("b", "s2", (), Outcome.ANSWERED),
        Exchange("c", "s3", (), Outcome.HTTP_ERROR),
    ]
    assert outcome_counts(exchanges) == {"answered": 2, "http_error": 1}


def test_the_last_query_and_rows_win_when_several_turns_carry_them():
    turns = (
        Turn("/chat/", 200, 1.0, cypher="MATCH (a) RETURN a", rows=[{"uid": "n1"}]),
        Turn("/chat/continue", 200, 1.0, cypher="MATCH (b) RETURN b", rows=[{"uid": "n2"}]),
    )
    exchange = Exchange("q", "s", turns, Outcome.ANSWERED)
    assert exchange.cypher == "MATCH (b) RETURN b"
    assert exchange.rows == [{"uid": "n2"}]
    assert exchange.seconds == 2.0


@pytest.mark.parametrize("outcome", [
    Outcome.ANSWERED, Outcome.NO_QUERY, Outcome.STALLED,
    Outcome.TOOL_CALL_LEAKED, Outcome.HTTP_ERROR, Outcome.TRANSPORT_ERROR,
])
def test_every_outcome_renders_in_a_summary_line(outcome):
    line = Exchange("q" * 80, "s", (), outcome).summary()
    assert outcome in line
