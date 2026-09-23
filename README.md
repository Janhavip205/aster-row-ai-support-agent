# Aster & Row AI Support Agent

## Project Overview
The repository implements an autonomous AI agent that assists customers of the fictional **Aster & Row** retailer. The agent can answer product, shipping, return, and policy questions, retrieve relevant knowledge‑base excerpts, lookup order information securely, and decide when to hand off to a human.

## What Was Built
- **Core agent** (`src.agent.core.AsterRowAgent`) orchestrating session management, retrieval, tool execution, Gemini response generation and observability.
- **Knowledge‑base retriever** (`src.kb.retriever.KBRetriever`) using a lightweight SentenceTransformer model with authoritativeness filtering.
- **Secure order‑lookup tool** (`src.tools.order_tool`) that normalises order IDs, reads data from `data/orders.json`, and only returns a whitelist of safe fields.
- **Session manager** (`src.agent.session.SessionManager`) preserving multi‑turn context.
- **Observability/tracing** (`src.observability.tracer`) with redaction of secrets, order data and system prompts.
- **Deterministic evaluation suite** (`src.evaluation.runner`) with a mock Gemini client for reproducible testing.
- **Interactive CLI** (`src/cli.py`) for terminal usage.

## Architecture / Components
- **Agent Core** – handles user message, decides on tool usage, builds prompts, records traces.
- **Retriever** – vector search over the markdown knowledge‑base, filters out non‑authoritative sources.
- **Order Tool** – safe lookup, field whitelist, PII redaction.
- **Session** – stores turn history, provides contextualised queries.
- **Tracer** – creates `TraceRecord` objects, redacts sensitive fields before export.
- **MockGenAIClient** – deterministic response generator used by the evaluation suite.
- **CLI** – thin wrapper that initialises the agent, runs an interactive REPL, prints citations and handoff notices.

## Setup Instructions
```bash
# Clone the repo (already done)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
A `.env.example` is provided; copy it to `.env` and set `GEMINI_API_KEY` if you want to use the real Gemini API. If the key is missing the agent falls back to the deterministic mock.

## Environment Variables
- `GEMINI_API_KEY` – API key for the Google Gemini model (optional).
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` – can be set to avoid remote model downloads during CI.

## Running the CLI
```bash
.venv\Scripts\python.exe src/cli.py
```
Enter messages at the prompt. Type `exit` or `quit` to end the session. Citations are printed after the answer, and a **Handoff required** line appears when the agent recommends escalation.

## Running Tests
```bash
.venv\Scripts\python.exe -m pytest -q
```
All 63 tests pass.

## Running Deterministic Evaluation
```bash
.venv\Scripts\python.exe -m src.evaluation.runner --mock
```
**Results:**
- Visible cases: 15 / 15 passed
- Custom cases:   6 / 6  passed
- Overall:        21 / 21 passed (100 % pass rate)

## Retrieval / Authoritative‑Source Filtering
The retriever ranks chunks by semantic similarity and then filters out any source whose filename does **not** end with `-authoritative.md`. This guarantees that only vetted, official documentation is cited.

## Secure Order Lookup & Whitelist
`lookup_order` normalises the supplied order ID, reads `data/orders.csv` and returns only the whitelisted fields (`order_id`, `status`, `estimated_delivery`, `handoff`). All other columns (customer name, email, address, internal notes) are omitted and never sent to the LLM, preventing accidental PII exposure.

## Multi‑Turn Session Handling
`SessionManager` stores every user and assistant turn. The agent uses `session.contextualize_query` to prepend the full conversation history to the current query, ensuring continuity across multiple messages.

## Prompt‑Injection Defense
Retrieved excerpts are sanitized before being injected into the system prompt. Additionally, the system prompt explicitly instructs the model to ignore any malicious instructions embedded in user‑provided or retrieved content.

## Observability / Redaction
`AgentTracer` creates a trace for each turn. Before persisting, the trace is redacted:
- `api_key`, `system_prompt`, and the full `orders.csv` are stripped.
- Only safe citation filenames are kept.

## Abstention / Handoff Behavior
If the user request is outside the agent’s domain (e.g., asking for legal advice) or the model’s confidence is low, the `AgentResponse.handoff` flag is set and the CLI displays **Handoff required**.

## Conflict Handling
When multiple sources provide contradictory statements, the agent selects the authoritative source (see filtering) and annotates the response with a note about the conflict.

## Evaluation Results
- **Visible cases:** 15 / 15 passed
- **Custom cases:**   6 / 6  passed
- **Overall:**        21 / 21 passed (100 % pass rate)
- **Test suite:**    63 / 63 passed

## Trade‑offs & Limitations
- No external vector database is used; the entire knowledge‑base fits in memory, which limits scalability for very large corpora.
- Retrieval relies on a static SentenceTransformer model; updates require rebuilding the index.
- The order‑lookup tool operates on a static CSV; in production this would be replaced by a secure service.
- Prompt‑injection defenses are heuristic and may not stop sophisticated attacks.
- Observability currently logs only to local files; integration with centralized logging would be needed for production.

## AI / Tool Disclosure
See `AI_TOOL_DISCLOSURE.md` for details on the AI assistance used during development.
