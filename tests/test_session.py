"""Unit tests for multi-turn session memory and query contextualization."""

import pytest
from src.agent.session import Session, SessionManager


def test_initial_query_returned_unchanged():
    """Verify that an initial query in a fresh session is returned unchanged."""
    session = Session(session_id="test-session-1")
    query = "How long does a regular customer have to return an unused backpack?"
    result = session.contextualize_query(query)
    assert result == query


def test_canada_shipping_followup():
    """Verify that follow-up 'What about Canada...' preserves international shipping context."""
    session = Session(session_id="test-session-2")
    session.add_user_message("Do you ship internationally?")
    session.add_assistant_message("Aster & Row currently ships internationally only to Canada.")

    followup = "What about Canada, and how long does it take?"
    contextualized = session.contextualize_query(followup)

    assert "international shipping" in contextualized.lower()
    assert "canada" in contextualized.lower()
    assert "how long" in contextualized.lower()


def test_canada_duties_followup():
    """Verify follow-up regarding duties inherits Canada shipping context."""
    session = Session(session_id="test-session-3")
    session.add_user_message("Do you ship to Canada?")
    session.add_assistant_message("Yes, Canada is our only international shipping destination.")

    followup = "What about duties and taxes?"
    contextualized = session.contextualize_query(followup)

    assert "canada" in contextualized.lower()
    assert "duties and taxes" in contextualized.lower()


def test_trailplus_return_followup():
    """Verify follow-up about final-sale items preserves TrailPlus return policy context."""
    session = Session(session_id="test-session-4")
    session.add_user_message("What is the return window for TrailPlus members?")
    session.add_assistant_message("TrailPlus members receive a 45-day return window from delivery.")

    followup = "Does that apply to final-sale items?"
    contextualized = session.contextualize_query(followup)

    assert "trailplus" in contextualized.lower()
    assert "return" in contextualized.lower()
    assert "final-sale" in contextualized.lower()


def test_order_id_pronoun_resolution():
    """Verify pronoun follow-up ('when will it arrive?') resolves to previous order ID."""
    session = Session(session_id="test-session-5")
    session.add_user_message("Where is ORD-1007?")
    session.add_assistant_message("Your order ORD-1007 has shipped with UPS.")

    followup = "When will it arrive?"
    contextualized = session.contextualize_query(followup)

    assert "ORD-1007" in contextualized
    assert "arrive" in contextualized.lower()


def test_bounded_history_limits_turns():
    """Verify that history does not grow indefinitely and trims older turns."""
    session = Session(session_id="test-session-6", max_history_turns=3)

    for i in range(1, 6):
        session.add_user_message(f"User message {i}")
        session.add_assistant_message(f"Assistant response {i}")

    # 3 turns = 6 messages max
    history = session.get_history()
    assert len(history) == 6
    assert history[0].content == "User message 3"
    assert history[-1].content == "Assistant response 5"


def test_session_isolation_no_cross_talk():
    """Verify that session A never leaks context to session B."""
    manager = SessionManager()
    session_a = manager.get_or_create("session-A")
    session_b = manager.get_or_create("session-B")

    # Session A discusses ORD-1007
    session_a.add_user_message("Where is ORD-1007?")
    session_a.add_assistant_message("ORD-1007 is in transit.")

    # Session B asks a pronoun question without prior context
    followup_in_b = "When will it arrive?"
    contextualized_b = session_b.contextualize_query(followup_in_b)

    # Session B must NOT contain ORD-1007
    assert "ORD-1007" not in contextualized_b
    assert contextualized_b == followup_in_b


def test_standalone_query_in_later_turn_not_polluted():
    """Verify that an unrelated standalone query in a subsequent turn is not polluted."""
    session = Session(session_id="test-session-7")
    session.add_user_message("Do you ship internationally?")
    session.add_assistant_message("Yes, to Canada.")

    standalone_query = "What is the warranty period for backpacks?"
    result = session.contextualize_query(standalone_query)

    assert result == standalone_query
    assert "international" not in result.lower()
    assert "canada" not in result.lower()


def test_session_manager_lifecycle():
    """Verify session manager creation, retrieval, and deletion."""
    manager = SessionManager()
    assert len(manager) == 0

    s1 = manager.get_or_create("s1")
    s2 = manager.get_or_create("s2")
    assert len(manager) == 2

    # Retrieval returns the exact same instance
    s1_again = manager.get_or_create("s1")
    assert s1 is s1_again

    manager.delete("s1")
    assert len(manager) == 1

    manager.clear_all()
    assert len(manager) == 0
