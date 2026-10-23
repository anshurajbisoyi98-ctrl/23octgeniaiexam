"""FastAPI backend. Serves the API, the PDF and the frontend from one process.

Run with:  uvicorn app.server:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag import RAGResult, answer_question, get_store

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_store()  # open ChromaDB and load the embedding model once, before the first request
    yield


app = FastAPI(title="Gas Turbine Maintenance Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.post("/api/ask", response_model=RAGResult)
async def ask(request: AskRequest) -> RAGResult:
    try:
        # The RAG pipeline is blocking (embeddings + LLM call), so run it off the event loop.
        return await run_in_threadpool(answer_question, request.question)
    except RuntimeError as error:  # e.g. missing API key
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {error}") from error


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks": get_store()._collection.count(),
        "llm_provider": settings.llm_provider,
    }


@app.get("/pdf")
def pdf() -> FileResponse:
    # Served inline so the browser opens it at /pdf#page=N
    return FileResponse(settings.pdf_path, media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})


# The frontend is mounted last so the API routes above take priority.
app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
