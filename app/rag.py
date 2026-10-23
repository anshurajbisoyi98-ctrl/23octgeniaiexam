"""Answer questions from the PDF with citations, or reply exactly "Not found".

Three layers decide "Not found":
  1. Relevance gate   - if the best retrieved chunk is not similar enough, stop (no LLM call).
  2. Grounded prompt  - the LLM may only use the retrieved context and must say found=false otherwise.
  3. Citation check   - every quote the LLM gives must really appear in a retrieved chunk of that page.
"""

from __future__ import annotations

import re
import time
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.ingest import get_embeddings

NOT_FOUND = "Not found"


class Citation(BaseModel):
    page: int = Field(description="Page number from the context header")
    section: str = Field(description="Section name from the context header")
    quote: str = Field(description="A short phrase copied word-for-word from that context chunk")


class LLMAnswer(BaseModel):
    found: bool = Field(description="True only if the context contains the answer")
    answer: str = Field(default="", description="The answer, using only the context")
    citations: list[Citation] = Field(default_factory=list)


class RAGResult(BaseModel):
    answer: str
    found: bool
    citations: list[Citation] = []
    latency_ms: int = 0
    top_score: float = 0.0


PROMPT = """You answer questions about the GE document GER-3620 "Heavy-Duty Gas Turbine \
Operating and Maintenance Considerations" using ONLY the context below.

Rules:
- Use only facts written in the context. Do not use outside knowledge, even if you know the answer.
- If the context does not contain the answer, return found=false with an empty answer and no citations.
- If found, write a clear answer in 2-5 sentences, or a short list when the question asks for
  several items. Keep the document's numbers and units exactly as written.
- Each context chunk starts with a header like [Section | page N]. For every citation copy the page
  and section from that header, and a quote of 5-20 words copied exactly from the chunk text.

Context:
{context}

Question: {question}"""


@lru_cache
def get_store() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_llm(settings: Settings) -> BaseChatModel:
    """Create the chat model chosen by LLM_PROVIDER. Temperature 0 for repeatable answers."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your .env file.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model, google_api_key=settings.google_api_key, temperature=0,
            timeout=60, max_retries=2,
        )
    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0, timeout=60)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use gemini or groq.")


def retrieve(question: str, k: int) -> list[tuple[Document, float]]:
    """Return the top-k chunks with cosine similarity (1.0 = identical)."""
    results = get_store().similarity_search_with_score(question, k=k)
    return [(doc, 1.0 - distance) for doc, distance in results]


def normalise(text: str) -> str:
    """Lower-case, unify quotes/dashes and collapse whitespace so quotes can be compared."""
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[–—]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def check_citations(citations: list[Citation], docs: list[Document]) -> list[Citation]:
    """Keep only citations whose quote appears in a retrieved chunk from the cited page.

    The section shown to the user is taken from that chunk's metadata, not from the LLM.
    """
    verified: list[Citation] = []
    for citation in citations:
        quote = normalise(citation.quote)
        if len(quote) < 8:
            continue
        for doc in docs:
            if doc.metadata["page"] == citation.page and quote in normalise(doc.page_content):
                fixed = Citation(page=citation.page, section=doc.metadata["section"], quote=citation.quote)
                if fixed not in verified:
                    verified.append(fixed)
                break
    return verified


def answer_question(question: str) -> RAGResult:
    settings = get_settings()
    start = time.perf_counter()

    def done(result: RAGResult) -> RAGResult:
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        return result

    question = question.strip()
    if not question:
        return done(RAGResult(answer=NOT_FOUND, found=False))

    # Layer 1: relevance gate.
    hits = retrieve(question, settings.top_k)
    top_score = hits[0][1] if hits else 0.0
    if top_score < settings.relevance_threshold:
        return done(RAGResult(answer=NOT_FOUND, found=False, top_score=top_score))

    # Layer 2: strict grounded prompt with structured (JSON) output.
    docs = [doc for doc, _ in hits]
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    llm = get_llm(settings).with_structured_output(LLMAnswer)
    reply: LLMAnswer = llm.invoke(PROMPT.format(context=context, question=question))
    if not reply.found or not reply.answer.strip():
        return done(RAGResult(answer=NOT_FOUND, found=False, top_score=top_score))

    # Layer 3: citation check.
    citations = check_citations(reply.citations, docs)
    if not citations:
        return done(RAGResult(answer=NOT_FOUND, found=False, top_score=top_score))

    return done(RAGResult(answer=reply.answer.strip(), found=True, citations=citations, top_score=top_score))


def format_result(result: RAGResult) -> str:
    """Plain-text form used by the CLI and the eval report."""
    if not result.found:
        return NOT_FOUND
    sources = "; ".join(dict.fromkeys(f"Page {c.page} — {c.section}" for c in result.citations))
    return f"Answer: {result.answer}\nSource: {sources}"
