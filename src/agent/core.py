"""Core agent orchestrator for Aster & Row.

Coordinates multi-turn session memory, metadata-aware knowledge base retrieval,
secure order lookup tool execution, Gemini response generation, and structured tracing.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_order_result,
    format_retrieved_knowledge,
)
from src.agent.session import SessionManager
from src.config import DEBUG, GEMINI_API_KEY, GEMINI_MODEL
from src.kb.retriever import KBRetriever, RetrievalResult
from src.observability.tracer import (
    AgentTracer,
    RetrievalScoreTrace,
    ToolExecutionTrace,
    TraceRecord,
)
from src.tools.order_tool import lookup_order, normalize_order_id


class AgentResponse(BaseModel):
    """Structured response model capturing the answer, citations, and operational metadata."""

    answer: str = Field(description="Customer-facing answer text")
    citations: list[str] = Field(default_factory=list, description="List of source filenames cited")
    handoff: bool = Field(default=False, description="Whether human assistance/escalation is recommended")
    tool_called: bool = Field(default=False, description="Whether an operational tool was executed")
    tool_name: Optional[str] = Field(default=None, description="Name of the tool executed, if any")
    retrieved_sources: list[str] = Field(default_factory=list, description="All KB sources retrieved for context")
    trace: Optional[TraceRecord] = Field(default=None, description="Sanitized observability trace record")


class AsterRowAgent:
    """Aster & Row AI Support Agent orchestrator."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        retriever: Optional[KBRetriever] = None,
        session_manager: Optional[SessionManager] = None,
        tracer: Optional[AgentTracer] = None,
        genai_client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_MODEL
        self.retriever = retriever or KBRetriever()
        self.session_manager = session_manager or SessionManager()
        self.tracer = tracer or AgentTracer(debug=DEBUG)
        self.client = genai_client

        # Initialize official google-genai client if API key is available and no client injected
        if self.client is None and self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                self._client_init_error = str(e)
            else:
                self._client_init_error = None
        else:
            self._client_init_error = None

    def handle(self, user_message: str, session_id: str = "default") -> AgentResponse:
        """Handle a user message within a session and generate a structured AgentResponse.

        Args:
            user_message: The text input from the customer.
            session_id: Unique conversation identifier.

        Returns:
            AgentResponse containing the response text, citations, and handoff flag.
        """
        start_time = time.perf_counter()
        trimmed_message = user_message.strip()

        # 1. Obtain isolated session and contextualize query
        session = self.session_manager.get_or_create(session_id)
        turn_index = (len(session.history) // 2) + 1

        if not trimmed_message:
            default_greeting = "Hello! How can I assist you with Aster & Row products, shipping, returns, or orders today?"
            latency_ms = (time.perf_counter() - start_time) * 1000
            trace = self.tracer.create_trace(
                session_id=session_id,
                turn_index=turn_index,
                user_query=trimmed_message,
                contextualized_query="",
                retrieved_scores=[],
                retrieved_sources=[],
                tool_called=False,
                tool_trace=None,
                final_answer=default_greeting,
                citations=[],
                handoff=False,
                latency_ms=latency_ms,
            )
            return AgentResponse(
                answer=default_greeting,
                citations=[],
                handoff=False,
                tool_called=False,
                tool_name=None,
                retrieved_sources=[],
                trace=trace,
            )

        contextualized_query = session.contextualize_query(trimmed_message)

        # 2. Determine order lookup requirements
        order_id_match = re.search(r"\b(ORD[\s_-]?\d+)\b", contextualized_query, re.IGNORECASE)
        is_order_inquiry = any(
            phrase in trimmed_message.lower()
            for phrase in (
                "where is my order",
                "where's my order",
                "track my order",
                "track my package",
                "order status",
                "status of my order",
                "when will my order",
                "check my order",
                "where is the order",
            )
        )

        tool_called = False
        tool_name: Optional[str] = None
        tool_trace: Optional[ToolExecutionTrace] = None
        order_data: Optional[dict[str, Any]] = None
        needs_handoff = False

        # Case A: User asks "where is my order?" without providing an order ID
        if is_order_inquiry and not order_id_match:
            clarification_text = (
                "I would be glad to check your order status! Could you please provide your "
                "order ID (for example, ORD-1007)?"
            )
            session.add_user_message(trimmed_message)
            session.add_assistant_message(clarification_text)
            latency_ms = (time.perf_counter() - start_time) * 1000
            trace = self.tracer.create_trace(
                session_id=session_id,
                turn_index=turn_index,
                user_query=trimmed_message,
                contextualized_query=contextualized_query,
                retrieved_scores=[],
                retrieved_sources=[],
                tool_called=False,
                tool_trace=None,
                final_answer=clarification_text,
                citations=[],
                handoff=False,
                latency_ms=latency_ms,
            )
            return AgentResponse(
                answer=clarification_text,
                citations=[],
                handoff=False,
                tool_called=False,
                tool_name=None,
                retrieved_sources=[],
                trace=trace,
            )

        # Case B: Identifiable order ID found
        if order_id_match:
            raw_order_id = normalize_order_id(order_id_match.group(1))
            order_data = lookup_order(raw_order_id)
            tool_called = True
            tool_name = "lookup_order"
            if order_data.get("handoff"):
                needs_handoff = True

            status_str = "not_found" if not order_data.get("found") else (
                "exception" if order_data.get("status") == "exception" else "success"
            )
            tool_trace = ToolExecutionTrace(
                tool_name="lookup_order",
                arguments={"order_id": raw_order_id},
                status=status_str,
                found=order_data.get("found"),
            )

        # Check for order privacy/PII probes
        is_privacy_probe = bool(
            re.search(
                r"\b(email|address|shipping address|risk score|internal note|warehouse note)\b",
                trimmed_message,
                re.IGNORECASE,
            )
        )
        if is_privacy_probe and order_id_match:
            needs_handoff = True

        # 3. Retrieve relevant knowledge base content
        retrieval_results = self.retriever.retrieve(
            contextualized_query,
            top_k=4,
            authoritative_only=True,
        )
        retrieved_sources = [r.chunk.filename for r in retrieval_results]
        retrieval_score_traces = [
            RetrievalScoreTrace(
                source=r.source,
                score=r.score,
                status=r.chunk.status,
                authority=r.chunk.policy_authority,
            )
            for r in retrieval_results
        ]

        # 4. Detect genuine active-source conflicts (e.g. Breeze Tumbler dishwasher vs hand-wash)
        retrieved_files = {r.chunk.filename for r in retrieval_results}
        if "11-product-care.md" in retrieved_files and "12-breeze-tumbler-product-card.md" in retrieved_files:
            if "tumbler" in contextualized_query.lower() or "dishwasher" in contextualized_query.lower():
                needs_handoff = True

        # Detect damaged final-sale review (human review required before approval)
        if "03-final-sale-and-promotions.md" in retrieved_files and "04-damaged-or-wrong-items.md" in retrieved_files:
            if "damaged" in contextualized_query.lower() or "broken" in contextualized_query.lower() or "zipper" in contextualized_query.lower():
                needs_handoff = True

        # 5. Format prompt with XML delimiters
        retrieved_xml = format_retrieved_knowledge(retrieval_results)
        order_xml = format_order_result(order_data)
        user_prompt = build_user_prompt(
            user_message=trimmed_message,
            retrieved_knowledge_xml=retrieved_xml,
            order_result_xml=order_xml,
            history=session.get_history(),
        )

        # 6. Generate model response
        gen_error: Optional[str] = None
        answer = self._generate_response(user_prompt)
        if "unable to process your request due to a technical error" in answer:
            gen_error = "Model generation error"

        # 7. Add user & assistant turns to session history
        session.add_user_message(trimmed_message)
        session.add_assistant_message(answer)

        # 8. Extract citations and evaluate handoff flag
        citations = self._extract_citations(answer, retrieved_sources)

        # Check for handoff indicators in response text
        lower_ans = answer.lower()
        if any(
            phrase in lower_ans
            for phrase in (
                "contact support",
                "customer support",
                "human support",
                "support specialist",
                "recommend contacting",
                "connect with our support",
                "support review is required",
                "insufficient information",
                "cannot confirm",
                "human confirmation",
            )
        ):
            needs_handoff = True

        # 9. Record trace
        latency_ms = (time.perf_counter() - start_time) * 1000
        trace = self.tracer.create_trace(
            session_id=session_id,
            turn_index=turn_index,
            user_query=trimmed_message,
            contextualized_query=contextualized_query,
            retrieved_scores=retrieval_score_traces,
            retrieved_sources=retrieved_sources,
            tool_called=tool_called,
            tool_trace=tool_trace,
            final_answer=answer,
            citations=citations,
            handoff=needs_handoff,
            latency_ms=latency_ms,
            error=gen_error,
        )

        return AgentResponse(
            answer=answer,
            citations=citations,
            handoff=needs_handoff,
            tool_called=tool_called,
            tool_name=tool_name,
            retrieved_sources=retrieved_sources,
            trace=trace,
        )

    def get_last_trace(self) -> Optional[TraceRecord]:
        """Return the most recent observability trace."""
        return self.tracer.get_last_trace()

    def _generate_response(self, user_prompt: str) -> str:
        """Call Gemini model to generate customer response, with safe fallbacks."""
        if self.client is None:
            if not self.api_key:
                return (
                    "Aster & Row AI Assistant configuration error: GEMINI_API_KEY is not set. "
                    "Please set your GEMINI_API_KEY in the environment or .env file to enable responses."
                )
            return (
                "Aster & Row AI Assistant service is temporarily unavailable. "
                "Please connect with human customer support."
            )

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                ),
            )
            return response.text.strip()
        except Exception as err:
            return (
                "I apologize, but I am currently unable to process your request due to a technical error. "
                "Please contact Aster & Row customer support for assistance."
            )

    @staticmethod
    def _extract_citations(answer: str, retrieved_sources: list[str]) -> list[str]:
        """Extract referenced markdown files from the generated answer and retrieved context."""
        found_citations: set[str] = set()

        # Match markdown filenames in text e.g. '01-returns-policy-current.md'
        file_matches = re.findall(r"\b(\d{2}-[a-z0-9\-]+\.md)\b", answer)
        for match in file_matches:
            found_citations.add(match)

        # If answer quotes [filename > heading]
        bracket_matches = re.findall(r"\[(\d{2}-[a-z0-9\-]+\.md)\s*>", answer)
        for match in bracket_matches:
            found_citations.add(match)

        # Retain order matching retrieved sources
        ordered_citations = [src for src in retrieved_sources if src in found_citations]
        # Append any remaining matches
        for c in found_citations:
            if c not in ordered_citations:
                ordered_citations.append(c)

        return ordered_citations
