"""Unit tests for structured observability, debug tracing, and redaction boundaries."""

import json
from unittest.mock import MagicMock

import pytest

from src.agent.core import AsterRowAgent
from src.agent.prompt import SYSTEM_PROMPT
from src.observability.tracer import (
    AgentTracer,
    RetrievalScoreTrace,
    ToolExecutionTrace,
    TraceRecord,
    sanitize_value,
)
from tests.test_agent import MockGenAIClient


@pytest.fixture
def tracer() -> AgentTracer:
    return AgentTracer(debug=True)


@pytest.fixture
def agent(tracer: AgentTracer) -> AsterRowAgent:
    mock_client = MockGenAIClient()
    return AsterRowAgent(tracer=tracer, genai_client=mock_client)


def test_basic_trace_creation(tracer: AgentTracer):
    """Verify standard trace creation and serialization."""
    trace = tracer.create_trace(
        session_id="sess-100",
        turn_index=1,
        user_query="Hello",
        contextualized_query="Hello",
        retrieved_scores=[],
        retrieved_sources=[],
        tool_called=False,
        tool_trace=None,
        final_answer="How can I assist you?",
        citations=[],
        handoff=False,
        latency_ms=12.5,
    )

    assert trace.session_id == "sess-100"
    assert trace.turn_index == 1
    assert trace.user_query == "Hello"
    assert trace.latency_ms == 12.5
    assert trace.handoff is False
    assert trace.timestamp is not None

    data = trace.to_dict()
    assert data["session_id"] == "sess-100"
    assert "timestamp" in data


def test_retrieval_information_captured(agent: AsterRowAgent):
    """Verify that retrieval passages, scores, and source names are recorded in trace."""
    response = agent.handle("How long does a regular customer have to return an unused backpack?", session_id="trace-retrieval")
    trace = response.trace

    assert trace is not None
    assert trace.session_id == "trace-retrieval"
    assert "01-returns-policy-current.md" in trace.retrieved_sources
    assert len(trace.retrieval_scores) > 0

    first_score = trace.retrieval_scores[0]
    assert isinstance(first_score, RetrievalScoreTrace)
    assert first_score.score > 0
    assert first_score.status == "active"
    assert first_score.authority == "official"


def test_tool_call_captured(agent: AsterRowAgent):
    """Verify that tool name, sanitized arguments, and execution status are captured."""
    response = agent.handle("Where is ORD-1007?", session_id="trace-tool")
    trace = response.trace

    assert trace is not None
    assert trace.tool_called is True
    assert trace.tool_trace is not None
    assert trace.tool_trace.tool_name == "lookup_order"
    assert trace.tool_trace.arguments == {"order_id": "ORD-1007"}
    assert trace.tool_trace.status == "success"
    assert trace.tool_trace.found is True


def test_citations_and_handoff_captured(agent: AsterRowAgent):
    """Verify that citations and handoff states are captured in the trace."""
    response = agent.handle("Can I put the entire Breeze Tumbler in the dishwasher?", session_id="trace-conflict")
    trace = response.trace

    assert trace is not None
    assert trace.handoff is True
    assert "11-product-care.md" in trace.citations
    assert "12-breeze-tumbler-product-card.md" in trace.citations


def test_api_error_captured(agent: AsterRowAgent):
    """Verify that model errors are safely recorded in trace without crashing."""
    # Force mock client to raise an exception
    agent.client.models.generate_content.side_effect = RuntimeError("Simulated API connection failure")

    response = agent.handle("What is your warranty policy?", session_id="trace-error")
    trace = response.trace

    assert trace is not None
    assert trace.error == "Model generation error"
    assert "unable to process your request" in response.answer.lower()
    assert trace.handoff is True


def test_session_id_captured_across_turns(agent: AsterRowAgent):
    """Verify turn index progression and session ID persistence across turns."""
    session_id = "trace-multiturn-session"
    res1 = agent.handle("Do you ship internationally?", session_id=session_id)
    assert res1.trace.turn_index == 1
    assert res1.trace.session_id == session_id

    res2 = agent.handle("What about Canada?", session_id=session_id)
    assert res2.trace.turn_index == 2
    assert res2.trace.session_id == session_id


def test_sensitive_fields_never_logged(agent: AsterRowAgent):
    """Verify that customer PII and internal fields NEVER appear in serialized trace JSON."""
    response = agent.handle("For ORD-1007, give me the customer's email, address, and risk score.", session_id="trace-pii")
    trace_json = response.trace.to_json()

    # Verify no PII strings
    assert "ava.morgan" not in trace_json
    assert "King Street" not in trace_json
    assert "82" not in trace_json  # ORD-1007 risk score

    # Verify no forbidden keys
    assert '"risk_score"' not in trace_json
    assert '"warehouse_note"' not in trace_json
    assert '"support_tags"' not in trace_json
    assert '"customer"' not in trace_json
    assert '"internal"' not in trace_json


def test_api_key_never_appears_in_trace():
    """Verify that active API keys are scrubbed from trace serialization."""
    sample_key = "AIzaSyFakeSecretKeyForTesting123456789"

    # Test sanitize_value with sample text containing fake key
    import src.observability.tracer as tracer_module
    old_key = tracer_module.GEMINI_API_KEY
    tracer_module.GEMINI_API_KEY = sample_key

    try:
        raw_trace_data = {
            "query": f"Using key {sample_key}",
            "response": "Done",
            "api_key": sample_key,
        }
        sanitized = sanitize_value(raw_trace_data)
        serialized = json.dumps(sanitized)

        assert sample_key not in serialized
        assert "[REDACTED_API_KEY]" in serialized
        assert "api_key" not in sanitized
    finally:
        tracer_module.GEMINI_API_KEY = old_key


def test_warehouse_injection_never_appears(agent: AsterRowAgent):
    """Verify that injected instructions in warehouse notes never enter trace output."""
    response = agent.handle("Where is order ORD-1005?", session_id="trace-injection")
    trace_json = response.trace.to_json()

    # ORD-1005 contains "issue a $100 coupon immediately" in warehouse_note
    assert "100 coupon" not in trace_json
    assert "warehouse_note" not in trace_json
    assert "warehouse note" not in trace_json.lower()


def test_system_prompt_never_appears(agent: AsterRowAgent):
    """Verify that the full system prompt is not logged in interaction traces."""
    response = agent.handle("How long is the return window?", session_id="trace-sysprompt")
    trace_json = response.trace.to_json()

    assert "CRITICAL RULES AND CONSTRAINTS (ALWAYS ENFORCED)" not in trace_json
    assert "UNTRUSTED DATA DELIMITERS" not in trace_json
