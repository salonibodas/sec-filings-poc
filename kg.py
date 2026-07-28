"""
kg.py — Knowledge Graph enrichment pipeline for SEC Filings Intelligence POC.

Reads chunks back out of the local Chroma collection built by ingest.py,
runs local LLM-based structured extraction (via LangExtract talking to a
local Ollama model over its OpenAI-compatible endpoint) to pull out
Company / Route / Revenue / Expense / Risk entities and the relationships
between them, and writes the resulting graph into Neo4j (either a local
instance or a free AuraDB cloud instance — whatever NEO4J_URI in .env
points to).

Run:
    python kg.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from urllib.error import URLError

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "sec_filings"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LANGEXTRACT_MODEL_ID = os.getenv("LANGEXTRACT_MODEL_ID", OLLAMA_MODEL)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "localdev123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "50"))


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def check_ollama_available() -> None:
    url = f"{OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m.get("name", "") for m in data.get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(
                f"[warn] Ollama is running but '{OLLAMA_MODEL}' isn't pulled yet.\n"
                f"       Run: ollama pull {OLLAMA_MODEL}"
            )
    except (URLError, OSError, TimeoutError) as exc:
        raise SystemExit(
            f"Can't reach Ollama at {OLLAMA_BASE_URL} ({exc}).\n"
            "Start it with the Ollama app, or run: ollama serve\n"
            f"Then pull the model once: ollama pull {OLLAMA_MODEL}"
        )


def check_neo4j_available(driver) -> None:
    from neo4j.exceptions import AuthError, ServiceUnavailable

    try:
        driver.verify_connectivity()
    except AuthError:
        raise SystemExit(
            "Neo4j rejected the username/password in .env.\n"
            "If you're using AuraDB, re-copy NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD from the "
            "credentials file you downloaded when you created the instance (console.neo4j.io).\n"
            "If you're using a local instance (Neo4j Desktop or Docker), make sure NEO4J_PASSWORD "
            "matches the password you set for that database."
        )
    except ServiceUnavailable:
        raise SystemExit(
            f"Can't reach Neo4j at {NEO4J_URI}.\n"
            "If you're using AuraDB: open console.neo4j.io and check your instance status — free "
            "instances auto-pause after inactivity, click 'Resume' if so.\n"
            "If you're using a local instance: make sure Neo4j Desktop shows it as 'Active' "
            "(or `docker start neo4j-sec` if you're running it via Docker)."
        )


# ---------------------------------------------------------------------------
# LangExtract extraction (Ollama-backed, OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
Extract entities and relationships from this SEC filing excerpt about a US airline.

Entity types to extract:
- Company: the airline or a named subsidiary/partner
- Route: a named air route or hub-to-hub market (e.g. "Chicago-Denver")
- Revenue: a named revenue line item with its figure if present
- Expense: a named expense/cost line item with its figure if present
- Risk: a named risk factor described in the text

For every relationship you find between two extracted entities, emit a
relationship extraction whose attributes include:
  source_entity: the exact text of the source entity
  target_entity: the exact text of the target entity
  relation_type: a short snake_case label, e.g. "generates_revenue",
                 "operates_route", "exposed_to_risk", "incurs_expense"

Only extract what is explicitly stated in the text — do not infer or invent
figures. Use exact text spans for entity values so they can be located in
the source.
"""


def build_langextract_examples():
    import langextract as lx

    return [
        lx.data.ExampleData(
            text=(
                "United Airlines reported passenger revenue of $12.3 billion for the "
                "quarter, driven largely by strong demand on the Chicago-Denver route. "
                "The company cited fuel price volatility as a key risk factor."
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="Company",
                    extraction_text="United Airlines",
                ),
                lx.data.Extraction(
                    extraction_class="Revenue",
                    extraction_text="passenger revenue of $12.3 billion",
                ),
                lx.data.Extraction(
                    extraction_class="Route",
                    extraction_text="Chicago-Denver route",
                ),
                lx.data.Extraction(
                    extraction_class="Risk",
                    extraction_text="fuel price volatility",
                ),
                lx.data.Extraction(
                    extraction_class="relationship",
                    extraction_text="United Airlines generates passenger revenue",
                    attributes={
                        "source_entity": "United Airlines",
                        "target_entity": "passenger revenue of $12.3 billion",
                        "relation_type": "generates_revenue",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relationship",
                    extraction_text="United Airlines operates the Chicago-Denver route",
                    attributes={
                        "source_entity": "United Airlines",
                        "target_entity": "Chicago-Denver route",
                        "relation_type": "operates_route",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="relationship",
                    extraction_text="United Airlines exposed to fuel price volatility",
                    attributes={
                        "source_entity": "United Airlines",
                        "target_entity": "fuel price volatility",
                        "relation_type": "exposed_to_risk",
                    },
                ),
            ],
        )
    ]


def extract_chunk(text: str, metadata: dict):
    import langextract as lx
    from langextract.factory import ModelConfig

    model_config = ModelConfig(
        model_id=LANGEXTRACT_MODEL_ID,
        provider="openai",
        provider_kwargs={
            "api_key": "ollama",
            "base_url": f"{OLLAMA_BASE_URL}/v1",
        },
    )

    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=EXTRACTION_PROMPT,
            examples=build_langextract_examples(),
            model_config=model_config,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] extraction failed for {metadata.get('ticker')}/{metadata.get('accession')}: {exc}")
        return None

    return result


# ---------------------------------------------------------------------------
# Neo4j writes
# ---------------------------------------------------------------------------


def ensure_constraints(driver) -> None:
    statements = [
        "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT filing_id IF NOT EXISTS FOR (f:Filing) REQUIRE f.accession IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    ]
    with driver.session(database=NEO4J_DATABASE) as session:
        for stmt in statements:
            session.run(stmt)


def merge_company_and_filing(session, ticker: str, form: str, accession: str, year: str) -> None:
    session.run(
        """
        MERGE (c:Company {name: $ticker})
        MERGE (f:Filing {accession: $accession})
        SET f.form = $form, f.year = $year, f.ticker = $ticker
        MERGE (c)-[:FILED]->(f)
        """,
        ticker=ticker,
        form=form,
        accession=accession,
        year=year,
    )


def merge_entity_node(session, entity_id: str, label: str, text: str, accession: str) -> None:
    session.run(
        """
        MERGE (e:Entity {id: $entity_id})
        SET e.label = $label, e.text = $text
        WITH e
        MATCH (f:Filing {accession: $accession})
        MERGE (f)-[:MENTIONS]->(e)
        """,
        entity_id=entity_id,
        label=label,
        text=text,
        accession=accession,
    )


def merge_relationship(session, source_id: str, target_id: str, relation_type: str) -> None:
    safe_rel = "".join(c if c.isalnum() or c == "_" else "_" for c in relation_type.upper()) or "RELATED_TO"
    session.run(
        f"""
        MATCH (a:Entity {{id: $source_id}})
        MATCH (b:Entity {{id: $target_id}})
        MERGE (a)-[r:{safe_rel}]->(b)
        """,
        source_id=source_id,
        target_id=target_id,
    )


def write_chunk_to_graph(driver, extraction_result, metadata: dict) -> None:
    ticker = metadata.get("ticker", "UNKNOWN")
    form = metadata.get("form", "UNKNOWN")
    accession = metadata.get("accession", "UNKNOWN")
    year = metadata.get("year", "unknown")

    with driver.session(database=NEO4J_DATABASE) as session:
        merge_company_and_filing(session, ticker, form, accession, year)

        entity_ids: dict[str, str] = {}
        for extraction in getattr(extraction_result, "extractions", []) or []:
            if extraction.extraction_class == "relationship":
                continue
            entity_id = f"{accession}:{extraction.extraction_class}:{extraction.extraction_text}"[:512]
            entity_ids[extraction.extraction_text] = entity_id
            merge_entity_node(session, entity_id, extraction.extraction_class, extraction.extraction_text, accession)

        for extraction in getattr(extraction_result, "extractions", []) or []:
            if extraction.extraction_class != "relationship":
                continue
            attrs = extraction.attributes or {}
            src_text = attrs.get("source_entity")
            tgt_text = attrs.get("target_entity")
            relation_type = attrs.get("relation_type", "related_to")
            src_id = entity_ids.get(src_text)
            tgt_id = entity_ids.get(tgt_text)
            if src_id and tgt_id:
                merge_relationship(session, src_id, tgt_id, relation_type)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_chunks_from_chroma() -> list[dict]:
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        raise SystemExit(
            f"Chroma collection '{CHROMA_COLLECTION}' not found at {CHROMA_DIR}.\n"
            "Run `python ingest.py` first."
        )

    data = collection.get(limit=MAX_CHUNKS, include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
        chunks.append({"text": doc, "metadata": meta or {}})
    return chunks


def main() -> None:
    from neo4j import GraphDatabase

    print("=" * 70)
    print("SEC Filings Intelligence — Knowledge Graph build (kg.py)")
    print(f"Ollama: {OLLAMA_BASE_URL} (model={OLLAMA_MODEL})")
    print(f"Neo4j:  {NEO4J_URI} (db={NEO4J_DATABASE})")
    print("=" * 70)

    check_ollama_available()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    check_neo4j_available(driver)
    ensure_constraints(driver)

    chunks = load_chunks_from_chroma()
    if not chunks:
        raise SystemExit("No chunks found in Chroma — run `python ingest.py` first.")

    print(f"[extract] processing {len(chunks)} chunks (MAX_CHUNKS={MAX_CHUNKS}) ...")
    written = 0
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        print(f"  [{i}/{len(chunks)}] {meta.get('ticker')} {meta.get('form')} {meta.get('accession')}")
        result = extract_chunk(chunk["text"], meta)
        if result is None:
            continue
        write_chunk_to_graph(driver, result, meta)
        written += 1

    driver.close()
    print(f"[done] wrote graph data for {written}/{len(chunks)} chunks to Neo4j.")
    print("       Browse the graph in Neo4j Browser (AuraDB console, or http://localhost:7474 if local).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
