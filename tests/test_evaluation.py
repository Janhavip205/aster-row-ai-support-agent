"""Unit tests for the deterministic evaluation suite runner."""

from pathlib import Path
from typing import Any

import pytest

from src.agent.core import AsterRowAgent
from src.agent.mock_client import MockGenAIClient
from src.config import CUSTOM_CASES_PATH, VISIBLE_CASES_PATH
from src.evaluation.runner import (
    check_concept,
    evaluate_case,
    load_all_evaluation_cases,
    load_cases,
    run_evaluation,
)


@pytest.fixture
def agent() -> AsterRowAgent:
    return AsterRowAgent(genai_client=MockGenAIClient())


def test_all_visible_cases_loaded():
    """Verify that all 15 visible evaluation cases are present and loaded."""
    cases = load_cases(VISIBLE_CASES_PATH)
    assert len(cases) == 15, f"Expected exactly 15 visible cases, got {len(cases)}"
    for c in cases:
        assert "id" in c
        assert "category" in c
        assert "messages" in c
        assert "expect" in c


def test_custom_cases_loaded():
    """Verify that at least 5 original custom cases are present."""
    cases = load_cases(CUSTOM_CASES_PATH)
    assert len(cases) >= 5, f"Expected at least 5 custom cases, got {len(cases)}"
    assert len(cases) == 6


def test_all_cases_execute_and_pass(agent: AsterRowAgent):
    """Verify that all 21 evaluation cases (15 visible + 6 custom) execute and pass."""
    report = run_evaluation(agent=agent, verbose=False)

    assert report.total_cases == 21, f"Expected 21 cases, got {report.total_cases}"
    assert report.failed_cases == 0, f"Expected 0 failures, got {report.failed_cases}. Failures: {[c.case_id for c in report.case_results if not c.passed]}"
    assert report.passed_cases == 21
    assert report.pass_rate == 100.0


def test_multi_turn_session_preservation(agent: AsterRowAgent):
    """Verify that multi-turn cases execute all messages in the same session."""
    canada_case = {
        "id": "canada-test-session",
        "category": "conversation",
        "messages": [
            {"role": "user", "content": "Do you ship internationally?"},
            {"role": "user", "content": "What about Canada, and how long does it take?"},
        ],
        "expect": {
            "must_include_concepts": [
                "Canada is supported",
                "5–9 business days",
            ],
            "required_sources": ["06-international-shipping.md"],
            "tool": "not_called",
            "handoff": False,
        },
    }

    result = evaluate_case(agent, canada_case)
    assert result.passed is True
    assert len(result.user_turns) == 2


def test_concept_checker_behavior():
    """Verify deterministic concept matching logic."""
    assert check_concept("We offer a 30 calendar days window from delivery", "30 calendar days")
    assert check_concept("Active members receive 45 days", "45 calendar days")
    assert check_concept("Final sale items that arrive damaged or defective can be reviewed", "final sale does not block damaged-item review")
    assert check_concept("Please report within 7 days", "report within 7 days")
    assert check_concept("A human support review before approval is needed", "human review before approval")
    assert check_concept("Aster & Row ships only to Canada", "Canada is supported")
    assert check_concept("We do not offer a lifetime warranty on any products", "no lifetime warranty")
    assert check_concept("Official sources conflict on cleaning guidance", "current official sources conflict")

    # Negative non-matches
    assert not check_concept("Everything is covered for 60 days", "30 calendar days")
    assert not check_concept("We ship worldwide to all countries", "shipping to Germany is not currently available")


def test_failed_case_reporting_clarity(agent: AsterRowAgent):
    """Verify that assertion failures produce clear, actionable failure reasons."""
    failing_case = {
        "id": "synthetic-failure-case",
        "category": "testing",
        "messages": [{"role": "user", "content": "What is the return window?"}],
        "expect": {
            "must_include": ["IMPOSSIBLE_NON_EXISTENT_PHRASE_12345"],
            "handoff": True,  # Will fail because actual answer has handoff=False
        },
    }

    result = evaluate_case(agent, failing_case)
    assert result.passed is False
    assert len(result.failure_reasons) >= 2
    assert any("IMPOSSIBLE_NON_EXISTENT_PHRASE" in r for r in result.failure_reasons)
    assert any("Handoff mismatch" in r for r in result.failure_reasons)


def test_summary_calculation(agent: AsterRowAgent):
    """Verify that category summaries aggregate totals, pass counts, and pass rates properly."""
    report = run_evaluation(agent=agent, verbose=False)

    total_from_categories = sum(c.total for c in report.category_summaries)
    passed_from_categories = sum(c.passed for c in report.category_summaries)

    assert total_from_categories == report.total_cases
    assert passed_from_categories == report.passed_cases
    assert len(report.category_summaries) > 0
    for cat in report.category_summaries:
        assert cat.total > 0
        assert cat.pass_rate >= 0.0
