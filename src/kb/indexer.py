"""Knowledge Base Indexer for Aster & Row.

Parses Markdown documents and YAML frontmatter, preserving document metadata,
status, authority, and section hierarchies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from src.config import KB_DIR


class DocumentChunk(BaseModel):
    """Represents an indexed section of a policy document with full metadata."""

    chunk_id: str = Field(description="Unique identifier: filename#heading-slug")
    filename: str = Field(description="Source markdown file name")
    document_id: str = Field(description="Frontmatter document_id (e.g. RET-2026-01)")
    title: str = Field(description="Document title")
    heading: str = Field(description="Section heading (e.g. Standard return window)")
    content: str = Field(description="Body text of the section")
    status: str = Field(description="Document status: active, superseded, or draft")
    policy_authority: str = Field(description="Policy authority: official or none")
    audience: str = Field(default="customer", description="Target audience: customer or internal")
    customer_answering: bool = Field(
        default=True,
        description="Whether this document can be used as authority for customer answers",
    )
    supersedes: Optional[str] = Field(default=None, description="Previous document_id superseded")
    superseded_by: Optional[str] = Field(default=None, description="Newer document_id that supersedes this")
    effective_date: Optional[str] = Field(default=None, description="Effective date")
    last_reviewed: Optional[str] = Field(default=None, description="Last review date")
    superseded_date: Optional[str] = Field(default=None, description="Date superseded")

    @property
    def source_citation(self) -> str:
        """Formatted source citation for agent responses: 'filename > heading'."""
        return f"{self.filename} > {self.heading}"

    @property
    def is_customer_authoritative(self) -> bool:
        """Returns True if the document is active, official, and customer-answering."""
        return (
            self.status == "active"
            and self.policy_authority == "official"
            and self.customer_answering is True
        )

    @property
    def embed_text(self) -> str:
        """Text used for semantic embedding, combining title, heading, and content."""
        return f"# {self.title}\n## {self.heading}\n{self.content}"


def _slugify(text: str) -> str:
    """Create a URL-safe slug from a heading string."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text)


def parse_markdown_document(filepath: Path | str) -> list[DocumentChunk]:
    """Parse a single Markdown file with YAML frontmatter into DocumentChunks.

    Preserves frontmatter metadata on each chunk and splits content by section headings.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge-base file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    filename = path.name

    # Expect YAML frontmatter delimited by ---
    if not raw_text.startswith("---"):
        raise ValueError(f"File {filename} does not contain valid YAML frontmatter delimiter '---'")

    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"File {filename} contains malformed frontmatter")

    frontmatter_raw = parts[1].strip()
    body_raw = parts[2].strip()

    meta = yaml.safe_load(frontmatter_raw) or {}

    document_id = str(meta.get("document_id", filename))
    title = str(meta.get("title", filename))
    status = str(meta.get("status", "draft")).lower()
    policy_authority = str(meta.get("policy_authority", "none")).lower()
    audience = str(meta.get("audience", "customer")).lower()
    customer_answering = bool(meta.get("customer_answering", True))
    supersedes = str(meta["supersedes"]) if meta.get("supersedes") else None
    superseded_by = str(meta["superseded_by"]) if meta.get("superseded_by") else None
    effective_date = str(meta["effective_date"]) if meta.get("effective_date") else None
    last_reviewed = str(meta["last_reviewed"]) if meta.get("last_reviewed") else None
    superseded_date = str(meta["superseded_date"]) if meta.get("superseded_date") else None

    chunks: list[DocumentChunk] = []

    # Split body by markdown section headings '## '
    sections = re.split(r"\n(?=##\s+)", body_raw)

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        lines = sec.split("\n")
        first_line = lines[0].strip()

        if first_line.startswith("##"):
            heading = first_line.lstrip("#").strip()
            content = "\n".join(lines[1:]).strip()
        elif first_line.startswith("#"):
            heading = "Overview"
            # Remove title line from content, keep any introductory body paragraphs
            body_lines = [line for line in lines[1:] if line.strip() and not line.strip().startswith("#")]
            content = "\n".join(body_lines).strip()
        else:
            heading = "General"
            content = sec

        if not content:
            continue

        chunk_id = f"{filename}#{_slugify(heading)}"

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                filename=filename,
                document_id=document_id,
                title=title,
                heading=heading,
                content=content,
                status=status,
                policy_authority=policy_authority,
                audience=audience,
                customer_answering=customer_answering,
                supersedes=supersedes,
                superseded_by=superseded_by,
                effective_date=effective_date,
                last_reviewed=last_reviewed,
                superseded_date=superseded_date,
            )
        )

    return chunks


def load_all_chunks(kb_dir: Path | str = KB_DIR) -> list[DocumentChunk]:
    """Load and index all Markdown files from the knowledge-base directory."""
    directory = Path(kb_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge-base directory not found: {directory}")

    chunks: list[DocumentChunk] = []
    # Sort files for deterministic ordering
    for file_path in sorted(directory.glob("*.md")):
        doc_chunks = parse_markdown_document(file_path)
        chunks.extend(doc_chunks)

    return chunks
