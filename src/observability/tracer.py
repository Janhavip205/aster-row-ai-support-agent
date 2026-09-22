"""Structured observability and debug tracing for Aster & Row.

Captures sanitized operational telemetry, retrieval scores, tool execution metadata,
citations, and latency without exposing sensitive credentials, PII, or internal data.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.config import DEBUG, GEMINI_API_KEY

# Blacklisted sensitive keys and substrings that must never appear in trace telemetry
FORBIDDEN_KEY_PATTERNS = {
    "api_key",
    "gemini_api_key",
    "password",
    "secret",
    "token",
    "customer",
    "email",
    "shipping_address",
    "address",
    "risk_score",
    "warehouse_note",
    "support_tags",
    "internal",
    "system_prompt",
}


def sanitize_value(val: Any) -> Any:
    """Recursively sanitize objects to prevent PII, credentials, or internal notes leakage."""
    if isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            if str(k).lower() in FORBIDDEN_KEY_PATTERNS:
                continue
            sanitized[k] = sanitize_value(v)
        return sanitized
    elif isinstance(val, list):
        return [sanitize_value(item) for item in val]
    elif isinstance(val, str):
        # Scrub any occurrence of active API key if set
        if GEMINI_API_KEY and GEMINI_API_KEY in val:
            val = val.replace(GEMINI_API_KEY, "[REDACTED_API_KEY]")
        # Scrub email pattern if present
        val = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", val)
        return val
    return val


class RetrievalScoreTrace(BaseModel):
    """Telemetry item for an individual retrieved passage."""

    source: str = Field(description="Source citation (filename > heading)")
    score: float = Field(description="Cosine similarity score")
    status: str = Field(description="Document status (e.g. active)")
    authority: str = Field(description="Document policy authority (e.g. official)")


class ToolExecutionTrace(BaseModel):
    """Telemetry item for an operational tool execution."""

    tool_name: str = Field(description="Name of tool executed")
    arguments: dict[str, Any] = Field(description="Sanitized input arguments (e.g. order_id)")
    status: str = Field(description="Result status: success, not_found, or exception")
    found: Optional[bool] = Field(default=None, description="Whether record was found")


class TraceRecord(BaseModel):
    """Structured telemetry record for a single customer-agent interaction turn."""

    timestamp: str = Field(description="ISO-8601 UTC timestamp")
    session_id: str = Field(description="Unique conversation session identifier")
    turn_index: int = Field(description="1-based turn number within session")
    user_query: str = Field(description="Raw user query")
    contextualized_query: str = Field(description="Clarified retrieval query")
    retrieved_sources: list[str] = Field(default_factory=list, description="List of source citations")
    retrieval_scores: list[RetrievalScoreTrace] = Field(default_factory=list, description="Passages with scores")
    tool_called: bool = Field(default=False, description="Whether a tool was invoked")
    tool_trace: Optional[ToolExecutionTrace] = Field(default=None, description="Tool execution telemetry")
    final_answer: str = Field(description="Customer-facing answer text")
    citations: list[str] = Field(default_factory=list, description="Citations extracted from response")
    handoff: bool = Field(default=False, description="Whether human handoff was recommended")
    error: Optional[str] = Field(default=None, description="Error message if a failure occurred")
    latency_ms: float = Field(description="Total interaction latency in milliseconds")

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to a sanitized dictionary with numeric redaction for final_answer."""
        raw_dict = self.model_dump()
        # Redact any numeric values (e.g., risk scores) in the final answer to avoid PII leakage
        if isinstance(raw_dict.get("final_answer"), str):
            raw_dict["final_answer"] = re.sub(r"\b\d+\b", "[REDACTED]", raw_dict["final_answer"])
        return sanitize_value(raw_dict)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize trace to a sanitized JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class AgentTracer:
    """Manages creation, logging, and storage of structured traces."""

    def __init__(self, debug: bool = DEBUG, logger: Optional[logging.Logger] = None) -> None:
        self.debug: bool = debug
        self.logger = logger or logging.getLogger("aster_row.tracer")
        self._trace_history: list[TraceRecord] = []

    def create_trace(
        self,
        session_id: str,
        turn_index: int,
        user_query: str,
        contextualized_query: str,
        retrieved_scores: list[RetrievalScoreTrace],
        retrieved_sources: list[str],
        tool_called: bool,
        tool_trace: Optional[ToolExecutionTrace],
        final_answer: str,
        citations: list[str],
        handoff: bool,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> TraceRecord:
        """Create, record, and optionally log a structured TraceRecord."""
        record = TraceRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            turn_index=turn_index,
            user_query=user_query,
            contextualized_query=contextualized_query,
            retrieved_sources=retrieved_sources,
            retrieval_scores=retrieved_scores,
            tool_called=tool_called,
            tool_trace=tool_trace,
            final_answer=final_answer,
            citations=citations,
            handoff=handoff,
            error=error,
            latency_ms=round(latency_ms, 2),
        )

        self._trace_history.append(record)

        if self.debug:
            self.logger.debug("Agent Trace: %s", record.to_json())

        return record

    def get_last_trace(self) -> Optional[TraceRecord]:
        """Return the most recent trace record, if any."""
        return self._trace_history[-1] if self._trace_history else None

    def get_traces_for_session(self, session_id: str) -> list[TraceRecord]:
        """Return all traces associated with a specific session ID."""
        return [t for t in self._trace_history if t.session_id == session_id]

    def clear(self) -> None:
        """Clear trace history (primarily for test resets)."""
        self._trace_history.clear()
