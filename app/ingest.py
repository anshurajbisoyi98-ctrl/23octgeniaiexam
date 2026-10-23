"""Ingest the PDF: read pages, clean them, find section headings, chunk, and store in ChromaDB.

Run with:  python -m app.ingest
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings

# Section headings in this PDF are printed in the accent colour at 10 pt (subsection) or 14 pt (section).
HEADING_COLOR = 0x006061
BODY_SIZE = 9.0
# Running header/footer live in the top 20 pt and bottom 50 pt of every page.
TOP_MARGIN, BOTTOM_MARGIN = 20, 745
# Sections that only list titles (no content) and would produce misleading citations.
SKIP_SECTIONS = {"List of Figures"}
CAPTION_RE = re.compile(r"^Figure\s+([A-G]-)?\d+\.")


@dataclass
class Line:
    page: int  # 1-based page number, as shown by a PDF viewer
    text: str
    size: float
    color: int
    y0: float
    is_heading: bool = False


@dataclass
class Table:
    page: int
    y0: float
    text: str  # rows rendered as "row label — column: value; ..."
    caption: str


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Local sentence-embedding model. BGE works best with a short instruction on queries."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "Represent this sentence for searching relevant passages: ",
        },
    )


def is_toc_page(page: pymupdf.Page) -> bool:
    """Table-of-contents pages are full of dot leaders ('. . . .')."""
    return page.get_text().count(". . . .") > 5


def good_tables(page: pymupdf.Page) -> list:
    """Keep only real tables. Charts are also detected as 'tables' but are mostly empty cells."""
    kept = []
    for table in page.find_tables().tables:
        cells = [c for row in table.extract() for c in row]
        filled = [c for c in cells if c and c.strip()]
        if table.row_count >= 3 and table.col_count >= 3 and len(filled) / len(cells) >= 0.45:
            kept.append(table)
    return kept


def clean_cell(cell: str | None) -> str:
    """Normalise cell text; table extraction sometimes drops spaces ('HotGasPath', 'Combustion(DLN)')."""
    text = re.sub(r"\s+", " ", cell or "").strip()
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])\(", " (", text)
    text = text.replace("Typeof", "Type of").replace("/ ", "/")
    return text


def render_table(rows: list[list[str | None]]) -> str:
    """Turn a table into self-describing lines so each value keeps its row and column label.

    A row with an empty first cell, or one that repeats the first header cell
    (e.g. 'Type of Inspection'), is a header row for the rows below it.
    """
    rows = [[clean_cell(c) for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    header_marker = rows[0][0]
    header = rows[0]
    out: list[str] = []
    for row in rows[1:]:
        if not row[0] or (header_marker and row[0] == header_marker):
            # Fill blank header cells from the left (merged cells).
            header = [c or (header[i] if i < len(header) else "") for i, c in enumerate(row)]
            continue
        label = row[0] or "(row)"
        pairs = [
            f"{header[i] if i < len(header) and header[i] else f'col {i}'}: {value}"
            for i, value in enumerate(row[1:], start=1)
            if value
        ]
        if pairs:
            out.append(f"{label} — " + "; ".join(pairs))
    return "\n".join(out)


def read_pdf(settings: Settings) -> tuple[list[Line], list[Table]]:
    """Read every page into cleaned lines (body text + headings) and rendered tables."""
    doc = pymupdf.open(settings.pdf_path)
    lines: list[Line] = []
    tables: list[Table] = []

    for index, page in enumerate(doc):
        page_no = index + 1
        if is_toc_page(page):
            continue

        page_tables = good_tables(page)
        boxes = [pymupdf.Rect(t.bbox) for t in page_tables]

        page_lines: list[Line] = []
        for block in page.get_text("dict")["blocks"]:
            for raw in block.get("lines", []):
                text = "".join(span["text"] for span in raw["spans"]).strip()
                y0 = raw["bbox"][1]
                if not text or y0 < TOP_MARGIN or y0 > BOTTOM_MARGIN:
                    continue  # empty, running header, page number or footer
                if any(box.contains(pymupdf.Rect(raw["bbox"])) for box in boxes):
                    continue  # text inside a table is added separately below
                span = raw["spans"][0]
                size = round(span["size"], 1)
                is_heading = (
                    span["color"] == HEADING_COLOR
                    and 9.9 <= size <= 14.5
                    and re.search(r"[a-z]", text) is not None  # skips column labels like '6B. 03'
                )
                # Drop chart tick labels: small text with almost no letters.
                if not is_heading and abs(size - BODY_SIZE) > 0.3 and len(re.findall(r"[A-Za-z]", text)) < 2:
                    continue
                page_lines.append(Line(page_no, text, size, span["color"], y0, is_heading))
        lines.extend(page_lines)

        for table, box in zip(page_tables, boxes):
            # The figure caption is the nearest "Figure N." line below the table.
            below = [l for l in page_lines if CAPTION_RE.match(l.text) and l.y0 >= box.y1 - 2]
            caption = min(below, key=lambda l: l.y0).text if below else ""
            tables.append(Table(page_no, box.y0, render_table(table.extract()), caption))

    return lines, tables


def join_lines(texts: list[str]) -> str:
    """Join PDF lines into flowing text, keeping bullets on their own line."""
    out = ""
    for text in texts:
        text = text.replace("\t", " ")
        if text.startswith(("•", "–")):
            out += "\n" + text
        elif out.endswith("-"):
            out += text  # keep compound words like 'GE Vernova-approved' together
        else:
            out += " " + text
    return re.sub(r"[ ]{2,}", " ", out).strip()


def build_sections(lines: list[Line]) -> list[tuple[int, str, str]]:
    """Group lines into (page, section, text) segments. Sections carry over page breaks."""
    segments: list[tuple[int, str, list[str]]] = []
    section = "Cover"
    previous_heading = False
    for line in lines:
        if line.is_heading:
            # Headings that wrap over two lines are merged ("Digital Solutions for" + "Asset Management").
            section = f"{section} {line.text}" if previous_heading else line.text
            previous_heading = True
            continue
        previous_heading = False
        if section in SKIP_SECTIONS:
            continue
        if segments and segments[-1][0] == line.page and segments[-1][1] == section:
            segments[-1][2].append(line.text)
        else:
            segments.append((line.page, section, [line.text]))
    return [(page, sec, join_lines(texts)) for page, sec, texts in segments]


def table_section(table: Table, lines: list[Line], segments: list[tuple[int, str, str]]) -> str:
    """Pick the section a table belongs to.

    Rule: the section whose text most recently mentions the figure (e.g. 'Figure 36 lists ...')
    on or before the table's page. Falls back to the heading active on that page.
    """
    match = re.match(r"Figure\s+([A-G]-)?(\d+)", table.caption)
    if match:
        figure_ref = re.compile(rf"Figure\s+{re.escape((match.group(1) or '') + match.group(2))}\b")
        for page, section, text in reversed(segments):
            if page <= table.page and figure_ref.search(text.replace(table.caption, "")):
                return section
    # Fallback: last heading above the table on the same page, else the section carried in.
    section = "Cover"
    for line in lines:
        if line.page > table.page or (line.page == table.page and line.y0 > table.y0):
            break
        if line.is_heading:
            section = line.text
    return section


def make_chunks(settings: Settings) -> list[Document]:
    lines, tables = read_pdf(settings)
    segments = build_sections(lines)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n", ". ", " ", ""],
    )
    source = settings.pdf_path.name
    docs: list[Document] = []

    for page, section, text in segments:
        for piece in splitter.split_text(text):
            if len(piece) < 40:
                continue  # stray fragments (lone labels) add noise
            docs.append(Document(
                page_content=f"[{section} | page {page}]\n{piece}",
                metadata={"page": page, "section": section, "source": source, "kind": "text"},
            ))

    for table in tables:
        section = table_section(table, lines, segments)
        title = f"{table.caption}\n" if table.caption else ""
        # Large tables are split by rows; every piece repeats the caption so it stays understandable.
        rows, piece = table.text.split("\n"), []
        for row in rows + [None]:
            if row is None or (piece and len("\n".join(piece + [row])) > settings.chunk_size * 2):
                if piece:
                    docs.append(Document(
                        page_content=f"[{section} | page {table.page} | table]\n{title}" + "\n".join(piece),
                        metadata={"page": table.page, "section": section, "source": source,
                                  "kind": "table", "figure": table.caption},
                    ))
                piece = []
            if row is not None:
                piece.append(row)

    for number, doc in enumerate(docs):
        doc.metadata["chunk_id"] = f"p{doc.metadata['page']:02d}-{doc.metadata['kind']}-{number:04d}"
    return docs


def ingest() -> int:
    settings = get_settings()
    if not settings.pdf_path.exists():
        raise SystemExit(f"PDF not found: {settings.pdf_path}")
    docs = make_chunks(settings)
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Idempotent: wipe the old collection, then add chunks with fixed ids.
    store.delete_collection()
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    return len(docs)


if __name__ == "__main__":
    count = ingest()
    print(f"Indexed {count} chunks into {get_settings().chroma_dir}")
