"""
ingest.py — Baseline RAG ingestion pipeline for SEC Filings Intelligence POC.

Downloads 10-K / 10-Q filings for UAL, DAL, AAL, LUV from SEC EDGAR (free,
no API key — just a descriptive User-Agent), chunks the primary documents,
embeds them locally with a HuggingFace sentence-transformer, and persists
the vectors to a local ChromaDB store at ./chroma_db.

Run:
    python ingest.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICKERS = ["UAL", "DAL", "AAL", "LUV"]
FORM_LOOKBACK_YEARS = {"10-K": 5, "10-Q": 3}

FILINGS_DIR = Path("./sec-edgar-filings")
CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "sec_filings"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_FILINGS_PER_FORM = int(os.getenv("MAX_FILINGS_PER_FORM", "0") or "0")

EDGAR_COMPANY_NAME = os.getenv("EDGAR_COMPANY_NAME")
EDGAR_EMAIL = os.getenv("EDGAR_EMAIL")

# Fallback: accept a single combined SEC_USER_AGENT="Name email@x.com" value too.
if not EDGAR_EMAIL:
    combined = os.getenv("SEC_USER_AGENT", "")
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", combined)
    if match:
        EDGAR_EMAIL = match.group(0)
        EDGAR_COMPANY_NAME = (
            EDGAR_COMPANY_NAME or combined[: match.start()].strip().strip('"') or "SEC Filings POC"
        )

EDGAR_COMPANY_NAME = EDGAR_COMPANY_NAME or "SEC Filings POC"

if not EDGAR_EMAIL:
    raise SystemExit(
        "Missing EDGAR_EMAIL in .env.\n"
        "SEC EDGAR requires a descriptive User-Agent with a real contact email, e.g.:\n"
        '  EDGAR_COMPANY_NAME="Your Name Your School"\n'
        '  EDGAR_EMAIL="you@example.com"\n'
    )


def download_filings() -> None:
    """Download 10-K and 10-Q filings for each ticker via sec-edgar-downloader."""
    from sec_edgar_downloader import Downloader

    dl = Downloader(EDGAR_COMPANY_NAME, EDGAR_EMAIL, str(FILINGS_DIR.parent))

    for ticker in TICKERS:
        for form, years in FORM_LOOKBACK_YEARS.items():
            kwargs = {"after": f"{2026 - years}-01-01"}
            if MAX_FILINGS_PER_FORM:
                kwargs["limit"] = MAX_FILINGS_PER_FORM
            print(f"[download] {ticker} {form} (last {years}y)...")
            try:
                dl.get(form, ticker, **kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {ticker} {form} download failed: {exc}")


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def accession_to_year(accession_dir_name: str) -> str:
    # e.g. 0000100517-25-000046 -> "20" + "25" -> 2025 (best-effort)
    parts = accession_dir_name.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        yy = int(parts[1])
        return str(2000 + yy)
    return "unknown"


def load_filing_documents() -> list:
    """Walk ./sec-edgar-filings/{ticker}/{form}/{accession}/ and build LlamaIndex Documents."""
    from llama_index.core import Document

    documents: list[Document] = []

    if not FILINGS_DIR.exists():
        print(f"[warn] {FILINGS_DIR} does not exist — did download_filings() run?")
        return documents

    for ticker_dir in sorted(FILINGS_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for form_dir in sorted(ticker_dir.iterdir()):
            if not form_dir.is_dir():
                continue
            form = form_dir.name
            for accession_dir in sorted(form_dir.iterdir()):
                if not accession_dir.is_dir():
                    continue

                primary_matches = list(accession_dir.glob("primary-document.*"))
                text = None
                source_file = None

                if primary_matches:
                    source_file = primary_matches[0]
                    raw = source_file.read_text(errors="ignore")
                    text = html_to_text(raw) if source_file.suffix in (".htm", ".html") else raw
                else:
                    fallback = accession_dir / "full-submission.txt"
                    if fallback.exists():
                        source_file = fallback
                        raw = fallback.read_text(errors="ignore")
                        text = html_to_text(raw)

                if not text or not text.strip():
                    continue

                year = accession_to_year(accession_dir.name)
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "ticker": ticker,
                            "form": form,
                            "accession": accession_dir.name,
                            "year": year,
                            "source_file": str(source_file),
                        },
                    )
                )
                print(f"  [loaded] {ticker} {form} {accession_dir.name} ({len(text):,} chars)")

    return documents


def build_vector_index(documents: list) -> None:
    """Chunk, embed, and persist documents into a local Chroma collection."""
    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    if not documents:
        raise SystemExit("No documents loaded — nothing to index. Check the download step above.")

    print(f"[embed] loading local embedding model: {EMBED_MODEL_NAME} ...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[index] chunking {len(documents)} documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    print(f"[done] Chroma collection '{CHROMA_COLLECTION}' persisted at {CHROMA_DIR}")
    print(f"       Total chunks in collection: {chroma_collection.count()}")


def main() -> None:
    print("=" * 70)
    print("SEC Filings Intelligence — Ingestion (ingest.py)")
    print(f"User-Agent: {EDGAR_COMPANY_NAME} <{EDGAR_EMAIL}>")
    print("=" * 70)

    download_filings()
    documents = load_filing_documents()
    build_vector_index(documents)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
        