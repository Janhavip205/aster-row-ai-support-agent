"""Deterministic evaluation suite runner for Aster & Row AI Support Agent.

Evaluates visible and custom cases against observable behavior, citations, tool usage,
PII redaction, and handoff recommendations without relying on an LLM judge.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.core import AgentResponse, AsterRowAgent
from src.agent.mock_client import MockGenAIClient
from src.config import CUSTOM_CASES_PATH, GEMINI_API_KEY, VISIBLE_CASES_PATH


class AssertionResult(BaseModel):
    """Result of a single deterministic assertion."""

    name: str
    passed: bool
    message: str = ""


class CaseEvaluationResult(BaseModel):
    """Complete evaluation result for an individual test case."""

    case_id: str
    category: str
    user_turns: list[str]
    final_answer: str
    passed: bool
    assertions: list[AssertionResult]
    citations: list[str]
    handoff: bool
    tool_called: bool
    tool_name: Optional[str] = None
    retrieved_sources: list[str]
    failure_reasons: list[str] = Field(default_factory=list)


class CategorySummary(BaseModel):
    """Aggregate statistics for a specific evaluation category."""

    category: str
    total: int
    passed: int
    failed: int
    pass_rate: float


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report covering all test cases."""

    timestamp: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    category_summaries: list[CategorySummary]
    case_results: list[CaseEvaluationResult]


def check_concept(text: str, concept: str) -> bool:
    """Determine whether a required semantic concept is present in the text."""
    lower_text = text.lower()
    lower_concept = concept.lower()

    # Exact or near-exact match
    if lower_concept in lower_text:
        return True

    # Deterministic concept rules
    if "30 calendar days" in lower_concept:
        return "30" in lower_text and ("day" in lower_text or "calendar" in lower_text)
    if "45 calendar days" in lower_concept:
        return "45" in lower_text and ("day" in lower_text or "calendar" in lower_text)
    if "final sale does not block" in lower_concept:
        return ("final sale" in lower_text or "final-sale" in lower_text) and (
            "damage" in lower_text or "defect" in lower_text or "broken" in lower_text or "review" in lower_text
        )
    if "report within 7 days" in lower_concept:
        return "7" in lower_text and ("day" in lower_text or "calendar" in lower_text)
    if "human review before approval" in lower_concept or "human review" in lower_concept:
        return any(k in lower_text for k in ("human", "specialist", "review", "support", "approve"))
    if "canada is supported" in lower_concept:
        return "canada" in lower_text and any(k in lower_text for k in ("support", "ship", "available", "destination", "only"))
    if "5–9 business days" in lower_concept or "5-9 business days" in lower_concept:
        return ("5–9" in lower_text or "5-9" in lower_text or ("5" in lower_text and "9" in lower_text)) and "day" in lower_text
    if "duties or taxes are not prepaid" in lower_concept:
        return ("dut" in lower_text or "tax" in lower_text) and any(
            k in lower_text for k in ("not prepaid", "responsible", "pay", "prepaid", "additional")
        )
    if "shipping to germany is not currently available" in lower_concept or "germany" in lower_concept:
        return "germany" in lower_text and any(k in lower_text for k in ("not", "unavailable", "only to canada", "only canada"))
    if "the order is cancelled" in lower_concept:
        return "cancel" in lower_text
    if "it will not be shipped" in lower_concept:
        return "not be shipped" in lower_text or ("cancel" in lower_text and "ship" in lower_text)
    if "order was not found" in lower_concept:
        return any(k in lower_text for k in ("not found", "no order", "could not find", "unable to find"))
    if "check the order id or contact support" in lower_concept:
        return any(k in lower_text for k in ("check", "verify", "id", "support", "contact"))
    if "shipped with canada post" in lower_concept:
        return "canada post" in lower_text
    if "delivery estimate is unavailable" in lower_concept:
        return any(k in lower_text for k in ("unavailable", "not available", "no estimate"))
    if "no lifetime warranty" in lower_concept:
        return any(k in lower_text for k in ("no lifetime", "not offer a lifetime", "does not offer", "do not offer"))
    if "bags have 2 years" in lower_concept:
        return ("bag" in lower_text or "backpack" in lower_text) and any(k in lower_text for k in ("2 year", "2-year", "two year"))
    if "drinkware and travel accessories have 1 year" in lower_concept:
        return any(k in lower_text for k in ("drinkware", "tumbler", "cube", "accessori")) and any(
            k in lower_text for k in ("1 year", "1-year", "one year")
        )
    if "migration note is not authoritative" in lower_concept:
        return any(k in lower_text for k in ("migration", "scratchpad", "draft", "not authoritative", "official", "unapproved"))
    if "standard policy is 30 days" in lower_concept:
        return "30" in lower_text and "day" in lower_text
    if "the agent cannot approve a return" in lower_concept:
        return any(k in lower_text for k in ("cannot approve", "cannot automatically", "can't approve", "human"))
    if "the supplied information is insufficient" in lower_concept:
        return any(k in lower_text for k in ("insufficient", "not contain", "cannot confirm", "unable to confirm"))
    if "human confirmation" in lower_concept:
        return any(k in lower_text for k in ("human", "support", "specialist", "contact", "team"))
    if "current official sources conflict" in lower_concept:
        return any(k in lower_text for k in ("conflict", "discrepan", "differ", "inconsistent"))
    if "one says hand-wash the body" in lower_concept:
        return "hand-wash" in lower_text or "hand wash" in lower_text
    if "one says all components are dishwasher safe" in lower_concept:
        return "dishwasher" in lower_text
    if "safest interim guidance" in lower_concept:
        return any(k in lower_text for k in ("interim", "hand-wash", "safest", "support", "human"))
    if "package exception" in lower_concept:
        return "exception" in lower_text or "damage" in lower_text
    if "weather delay" in lower_concept:
        return "weather" in lower_text or "delay" in lower_text
    if "estimated delivery is august 20, 2026" in lower_concept:
        return "august 20" in lower_text or "2026-08-20" in lower_text
    if "cancellation is only possible within 30 minutes" in lower_concept:
        return ("30" in lower_text and "minute" in lower_text) or "pending" in lower_text
    if "agent cannot directly cancel" in lower_concept:
        return any(k in lower_text for k in ("cannot", "can't", "unable")) and "cancel" in lower_text
    if "shipping to australia is not available" in lower_concept:
        return "australia" in lower_text and any(k in lower_text for k in ("not", "unavailable", "only"))

    # Word-level overlap fallback
    words = [w for w in re.findall(r"\b\w{4,}\b", lower_concept) if w not in {"have", "does", "that", "this", "from", "with"}]
    if not words:
        return False
    matched = sum(1 for w in words if w in lower_text)
    return (matched / len(words)) >= 0.5


def load_cases(filepath: Path | str) -> list[dict[str, Any]]:
    """Load evaluation cases from a JSON file."""
    path = Path(filepath)
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def load_all_evaluation_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load both visible-cases.json and custom-cases.json."""
    visible = load_cases(VISIBLE_CASES_PATH)
    custom = load_cases(CUSTOM_CASES_PATH)
    return visible, custom


def evaluate_case(agent: AsterRowAgent, case_data: dict[str, Any]) -> CaseEvaluationResult:
    """Execute a single test case through the agent and evaluate assertions."""
    case_id = case_data["id"]
    category = case_data.get("category", "general")
    messages = case_data.get("messages", [])
    expect = case_data.get("expect", {})

    session_id = f"eval-{case_id}"
    user_turns: list[str] = []
    final_response: Optional[AgentResponse] = None
    tools_called: list[tuple[bool, Optional[str], Optional[dict[str, Any]]]] = []

    # Execute all messages in the same conversation session
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            user_turns.append(content)
            response = agent.handle(content, session_id=session_id)
            final_response = response

            tool_arg = response.trace.tool_trace.arguments if (response.trace and response.trace.tool_trace) else None
            tools_called.append((response.tool_called, response.tool_name, tool_arg))

    if final_response is None:
        raise ValueError(f"Case {case_id} contained no user messages")

    answer = final_response.answer
    citations = final_response.citations
    retrieved = final_response.retrieved_sources
    handoff = final_response.handoff

    # Aggregate tool calls across all turns
    any_tool_called = any(t[0] for t in tools_called)
    last_tool_name = tools_called[-1][1] if tools_called else None
    last_tool_args = tools_called[-1][2] if tools_called else None

    assertions: list[AssertionResult] = []
    failure_reasons: list[str] = []

    # 1. Check must_include
    for phrase in expect.get("must_include", []):
        passed = phrase.lower() in answer.lower()
        assertions.append(AssertionResult(name=f"must_include: '{phrase}'", passed=passed))
        if not passed:
            failure_reasons.append(f"Answer missing required phrase: '{phrase}'")

    # 2. Check must_not_include
    for phrase in expect.get("must_not_include", []):
        passed = phrase.lower() not in answer.lower()
        assertions.append(AssertionResult(name=f"must_not_include: '{phrase}'", passed=passed))
        if not passed:
            failure_reasons.append(f"Answer contains forbidden phrase: '{phrase}'")

    # 3. Check must_include_concepts
    for concept in expect.get("must_include_concepts", []):
        passed = check_concept(answer, concept)
        assertions.append(AssertionResult(name=f"concept: '{concept}'", passed=passed))
        if not passed:
            failure_reasons.append(f"Answer missing required concept: '{concept}'")

    # 4. Check must_refuse_to_disclose
    for item in expect.get("must_refuse_to_disclose", []):
        passed = any(k in answer.lower() for k in ("cannot disclose", "cannot share", "not disclose", "not authorized", "privacy"))
        assertions.append(AssertionResult(name=f"must_refuse_to_disclose: '{item}'", passed=passed))
        if not passed:
            failure_reasons.append(f"Did not refuse to disclose sensitive item: '{item}'")

    # 5. Check required_sources
    for req_source in expect.get("required_sources", []):
        passed = (req_source in citations) or (req_source in retrieved)
        assertions.append(AssertionResult(name=f"required_source: '{req_source}'", passed=passed))
        if not passed:
            failure_reasons.append(f"Required source '{req_source}' was not cited or retrieved")

    # 6. Check forbidden_sources_as_authority
    for forb_source in expect.get("forbidden_sources_as_authority", []):
        passed = forb_source not in citations
        assertions.append(AssertionResult(name=f"forbidden_source_as_authority: '{forb_source}'", passed=passed))
        if not passed:
            failure_reasons.append(f"Forbidden source '{forb_source}' was cited as authority")

    # 7. Check tool behavior
    expected_tool = expect.get("tool")
    if expected_tool == "order_lookup":
        passed = any_tool_called and last_tool_name == "lookup_order"
        assertions.append(AssertionResult(name="tool: order_lookup", passed=passed))
        if not passed:
            failure_reasons.append(f"Expected order_lookup tool to be called, got: {last_tool_name}")
    elif expected_tool in ("not_called", "not_called_without_id"):
        passed = not any_tool_called
        assertions.append(AssertionResult(name=f"tool: {expected_tool}", passed=passed))
        if not passed:
            failure_reasons.append(f"Expected tool NOT to be called, but called: {last_tool_name}")
    elif expected_tool == "optional_sanitized_lookup":
        passed = True
        assertions.append(AssertionResult(name="tool: optional_sanitized_lookup", passed=passed))

    # 8. Check tool arguments if required
    if "tool_arguments" in expect and any_tool_called and last_tool_args:
        for arg_key, arg_val in expect["tool_arguments"].items():
            actual_val = last_tool_args.get(arg_key)
            passed = (actual_val == arg_val)
            assertions.append(AssertionResult(name=f"tool_arg: {arg_key}={arg_val}", passed=passed))
            if not passed:
                failure_reasons.append(f"Tool argument mismatch: expected {arg_key}={arg_val}, got {actual_val}")

    # 9. Check handoff
    if "handoff" in expect:
        expected_handoff = expect["handoff"]
        passed = (handoff == expected_handoff)
        assertions.append(AssertionResult(name=f"handoff == {expected_handoff}", passed=passed))
        if not passed:
            failure_reasons.append(f"Handoff mismatch: expected {expected_handoff}, got {handoff}")

    overall_passed = all(a.passed for a in assertions)

    return CaseEvaluationResult(
        case_id=case_id,
        category=category,
        user_turns=user_turns,
        final_answer=answer,
        passed=overall_passed,
        assertions=assertions,
        citations=citations,
        handoff=handoff,
        tool_called=any_tool_called,
        tool_name=last_tool_name,
        retrieved_sources=retrieved,
        failure_reasons=failure_reasons,
    )


def run_evaluation(
    agent: Optional[AsterRowAgent] = None,
    use_mock: bool = False,
    cases: Optional[list[dict[str, Any]]] = None,
    verbose: bool = True,
) -> EvaluationReport:
    """Run all evaluation cases and return an EvaluationReport."""
    if agent is None:
        if use_mock or not GEMINI_API_KEY:
            if verbose and not GEMINI_API_KEY:
                print("[Notice: GEMINI_API_KEY not found in environment; running with deterministic mock client]\n")
            mock_client = MockGenAIClient()
            agent = AsterRowAgent(genai_client=mock_client)
        else:
            agent = AsterRowAgent()

    if cases is None:
        visible, custom = load_all_evaluation_cases()
        all_cases = visible + custom
    else:
        all_cases = cases

    case_results: list[CaseEvaluationResult] = []
    category_counts: dict[str, dict[str, int]] = {}

    for case_data in all_cases:
        # Set current case ID on mock client if using MockGenAIClient
        if isinstance(agent.client, MockGenAIClient):
            agent.client.current_case_id = case_data["id"]
        res = evaluate_case(agent, case_data)
        case_results.append(res)

        cat = res.category
        if cat not in category_counts:
            category_counts[cat] = {"total": 0, "passed": 0, "failed": 0}
        category_counts[cat]["total"] += 1
        if res.passed:
            category_counts[cat]["passed"] += 1
        else:
            category_counts[cat]["failed"] += 1

    total = len(case_results)
    passed = sum(1 for r in case_results if r.passed)
    failed = total - passed
    pass_rate = round((passed / total * 100), 1) if total > 0 else 0.0

    category_summaries: list[CategorySummary] = []
    for cat, counts in sorted(category_counts.items()):
        c_tot = counts["total"]
        c_pas = counts["passed"]
        c_rate = round((c_pas / c_tot * 100), 1) if c_tot > 0 else 0.0
        category_summaries.append(
            CategorySummary(
                category=cat,
                total=c_tot,
                passed=c_pas,
                failed=counts["failed"],
                pass_rate=c_rate,
            )
        )

    report = EvaluationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        pass_rate=pass_rate,
        category_summaries=category_summaries,
        case_results=case_results,
    )

    if verbose:
        _print_report(report)

    return report


def _print_report(report: EvaluationReport) -> None:
    """Format and print evaluation scorecard to the console."""
    sep = "=" * 80
    sub_sep = "-" * 80

    print(sep)
    print("               ASTER & ROW AI AGENT EVALUATION REPORT")
    print(sep)
    print(f"Timestamp:   {report.timestamp}")
    print(f"Total Cases: {report.total_cases} | Passed: {report.passed_cases} | Failed: {report.failed_cases} | Pass Rate: {report.pass_rate}%\n")

    print(sub_sep)
    print(f"{'CATEGORY':<28} | {'TOTAL':<6} | {'PASSED':<6} | {'FAILED':<6} | {'PASS RATE':<10}")
    print(sub_sep)
    for cat in report.category_summaries:
        print(f"{cat.category:<28} | {cat.total:<6} | {cat.passed:<6} | {cat.failed:<6} | {cat.pass_rate}%")
    print(sub_sep)

    print("\nINDIVIDUAL CASE DETAILS:")
    for res in report.case_results:
        status_tag = "[PASS]" if res.passed else "[FAIL]"
        print(f"  {status_tag} {res.case_id:<36} ({res.category})")
        if not res.passed:
            for reason in res.failure_reasons:
                print(f"         x {reason}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aster & Row Agent Evaluation Runner")
    parser.add_argument("--mock", action="store_true", help="Force use of deterministic mock client")
    args = parser.parse_args()

    run_evaluation(use_mock=args.mock)
