"""Multi-turn session memory and deterministic query contextualization.

Manages isolated conversation sessions and resolves anaphoric references, pronouns,
and elliptical follow-up queries using bounded recent conversation context.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional
from pydantic import BaseModel, Field

from src.tools.order_tool import normalize_order_id


class Message(BaseModel):
    """Represents a single conversation turn."""

    role: str = Field(description="Role of the sender: 'user' or 'assistant'")
    content: str = Field(description="Message body text")


class Session:
    """Manages conversation history and query contextualization for an individual session."""

    def __init__(self, session_id: Optional[str] = None, max_history_turns: int = 6) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.max_history_turns: int = max_history_turns
        self.history: list[Message] = []

    def add_user_message(self, content: str) -> None:
        """Append a user message, maintaining bounded history."""
        self.history.append(Message(role="user", content=content.strip()))
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant response, maintaining bounded history."""
        self.history.append(Message(role="assistant", content=content.strip()))
        self._trim_history()

    def _trim_history(self) -> None:
        """Retain only the most recent turns (2 messages per turn: user + assistant)."""
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_history(self, limit: Optional[int] = None) -> list[Message]:
        """Return conversation messages up to limit."""
        if limit is not None:
            return self.history[-limit:]
        return list(self.history)

    def clear(self) -> None:
        """Clear all conversation history for this session."""
        self.history.clear()

    def contextualize_query(self, query: str) -> str:
        """Contextualize a follow-up query using recent conversation turns.

        Resolves pronouns ('it', 'that', 'they'), elliptical queries ('what about Canada?'),
        and references to prior subjects (such as order IDs or return policies).

        Returns:
            The clarified query string suitable for retrieval.
        """
        trimmed_query = query.strip()
        if not trimmed_query:
            return ""

        # Initial turn: return query unchanged
        user_messages = [m for m in self.history if m.role == "user"]
        if not user_messages:
            return trimmed_query

        # Check for follow-up indicators
        lower_q = trimmed_query.lower()
        words = lower_q.split()

        has_pronoun = bool(
            re.search(r"\b(it|its|they|them|their|that|this|these|those)\b", lower_q)
        )
        is_elliptical = any(
            lower_q.startswith(phrase)
            for phrase in (
                "what about",
                "how about",
                "how long",
                "how much",
                "when ",
                "where ",
                "and ",
                "does that",
                "is that",
                "can that",
                "what if",
            )
        ) or len(words) <= 4

        # Look for order ID in previous user message
        prev_user_text = " ".join([m.content for m in user_messages[-2:]])
        order_match = re.search(r"\b(ORD[\s_-]?\d+)\b", prev_user_text, re.IGNORECASE)
        prev_order_id = normalize_order_id(order_match.group(1)) if order_match else None

        # Check if the query refers to an order without repeating the ID
        current_has_order = bool(re.search(r"\bORD[\s_-]?\d+\b", trimmed_query, re.IGNORECASE))
        if prev_order_id and not current_has_order and (has_pronoun or is_elliptical or "arrive" in lower_q or "status" in lower_q or "order" in lower_q):
            return f"{prev_order_id} {trimmed_query}"

        # If it's a standalone complete question without pronouns or ellipsis, don't alter it
        if not (has_pronoun or is_elliptical):
            return trimmed_query

        # Extract topical antecedents from recent turns
        antecedents: list[str] = []

        # 1. Geographic / shipping antecedents
        if re.search(r"\binternational(?:ly)?\b", prev_user_text, re.IGNORECASE):
            if "international" not in lower_q:
                antecedents.append("international shipping")
        elif re.search(r"\bshipping\b", prev_user_text, re.IGNORECASE) and "shipping" not in lower_q:
            antecedents.append("shipping")

        if re.search(r"\bcanada\b", prev_user_text, re.IGNORECASE) and "canada" not in lower_q:
            antecedents.append("to Canada")

        # 2. Membership / Policy antecedents
        if re.search(r"\btrailplus\b", prev_user_text, re.IGNORECASE) and "trailplus" not in lower_q:
            antecedents.append("TrailPlus")

        if re.search(r"\breturn\b", prev_user_text, re.IGNORECASE) and "return" not in lower_q:
            antecedents.append("return policy")

        if re.search(r"\bwarranty\b", prev_user_text, re.IGNORECASE) and "warranty" not in lower_q:
            antecedents.append("warranty")

        if re.search(r"\bbreeze tumbler\b", prev_user_text, re.IGNORECASE) and "tumbler" not in lower_q:
            antecedents.append("Breeze Tumbler")

        if antecedents:
            context_prefix = " ".join(antecedents)
            return f"{context_prefix} {trimmed_query}"

        return trimmed_query


class SessionManager:
    """Manages multiple independent, isolated sessions."""

    def __init__(self, max_history_turns: int = 6) -> None:
        self.max_history_turns: int = max_history_turns
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """Retrieve an existing session by ID or create a new isolated session."""
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                session_id=session_id,
                max_history_turns=self.max_history_turns,
            )
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        """Remove a session from memory."""
        self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """Remove all active sessions (primarily for test resets)."""
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)
