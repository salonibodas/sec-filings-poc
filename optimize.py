"""
optimize.py — Token optimization + 3-way evaluation for SEC Filings Intelligence POC.

Compares three RAG variants on a benchmark question set (UAL / DAL / AAL / LUV,
single-doc / multi-doc / causal-reasoning questions):

  1. Baseline  — plain vector retrieval over the Chroma index from ingest.py
  2. KG        — vector retrieval + one-hop Neo4j graph traversal from kg.py
  3. Optimized — tighter top_k + context trimming + LLMLingua compression +
                 a terse "Caveman" system prompt to cut tokens further

All generation and judging runs locally through Ollama — zero external API
calls, zero API keys, $0 cost.

Writes results to ./eval_results.json, which app.py reads for the dashboard.

Run:
    python optimize.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from urllib.error import URLError

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "sec_filings"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", OLLAMA_MODEL)
EVAL_MODEL = os.getenv("EVAL_MODEL", OLLAMA_MODEL)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "localdev123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

BASELINE_TOP_K = int(os.getenv("BASELINE_TOP_K", "5"))
OPTIMIZED_TOP_K = int(os.getenv("OPTIMIZED_TOP_K", "3"))
LLMLINGUA_RATE = float(os.getenv("LLMLINGUA_RATE", "0.5"))
MAX_CONTEXT_CHARS_OPTIMIZED = int(os.getenv("MAX_CONTEXT_CHARS_OPTIMIZED", "4000"))

# Fully local => $0. Left overridable only for a hypothetical cloud comparison.
COST_PER_1K_INPUT = float(os.getenv("COST_PER_1K_INPUT_TOKENS", "0.0"))
COST_PER_1K_OUTPUT = float(os.getenv("COST_PER_1K_OUTPUT_TOKENS", "0.0"))

RESULTS_PATH = "./eval_results.json"

TARGETS = {
    "accuracy_improvement_pct": 25,
    "context_precision_improvement_pct": 30,
    "hallucination_reduction_pct": 40,
    "input_token_reduction_pct": 50,
    "output_token_reduction_pct": 40,
    "cost_reduction_pct": 50,
    "latency_reduction_pct": 25,
}


# ---------------------------------------------------------------------------
# Benchmark questions (planning.md evaluation dataset — representative subset)
# ---------------------------------------------------------------------------

BENCHMARK_QUESTIONS = [
    {"id": "q1", "category": "single-doc", "ticker": "UAL",
     "question": "What did United Airlines identify as a key risk factor related to fuel costs?"},
    {"id": "q2", "category": "single-doc", "ticker": "DAL",
     "question": "What was Delta Air Lines' reported passenger revenue in its most recent filing?"},
    {"id": "q3", "category": "single-doc", "ticker": "AAL",
     "question": "What operating expenses did American Airlines highlight as significant?"},
    {"id": "q4", "category": "single-doc", "ticker": "LUV",
     "question": "What routes or markets does Southwest Airlines describe as strategically important?"},
    {"id": "q5", "category": "multi-doc", "ticker": "UAL,DAL",
     "question": "How do United and Delta's discussions of labor costs compare across their recent filings?"},
    {"id": "q6", "category": "multi-doc", "ticker": "AAL,LUV",
     "question": "How do American and Southwest each describe their exposure to fuel price volatility?"},
    {"id": "q7", "category": "multi-doc", "ticker": "UAL,AAL,DAL,LUV",
     "question": "Across all four airlines, what risk factors appear most consistently in their SEC filings?"},
    {"id": "q8", "category": "causal-reasoning", "ticker": "UAL",
     "question": "How might rising fuel costs affect United Airlines' route profitability based on its filings?"},
    {"id": "q9", "category": "causal-reasoning", "ticker": "DAL",
     "question": "What connection does Delta draw between labor agreements and operating expenses?"},
    {"id": "q10", "category": "causal-reasoning", "ticker": "AAL,LUV",
     "question": "How could a shared risk factor like fuel volatility differently impact American vs. Southwest given their route networks?"},
]


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def check_ollama_available() -> None:
    url = f"{OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            json.loads(resp.read())
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
# LLM + retrieval plumbing
# ---------------------------------------------------------------------------


def get_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(model=GENERATION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def get_chroma_collection():
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(CHROMA_COLLECTION)
    except Exception:
        raise SystemExit(
            f"Chroma collection '{CHROMA_COLLECTION}' not found at {CHROMA_DIR}.\n"
            "Run `python ingest.py` first."
        )


def get_embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"))


def vector_retrieve(collection, embed_model, question: str, top_k: int) -> list[dict]:
    query_embedding = embed_model.get_query_embedding(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        hits.append({"text": doc, "metadata": meta or {}})
    return hits


def graph_retrieve(driver, question: str, tickers: list[str], limit: int = 10) -> list[str]:
    """One-hop traversal: pull entities + their direct relationships for the tickers in question."""
    facts: list[str] = []
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(
            """
            MATCH (c:Company)-[:FILED]->(f:Filing)-[:MENTIONS]->(e:Entity)
            WHERE c.name IN $tickers
            OPTIONAL MATCH (e)-[r]->(e2:Entity)
            RETURN c.name AS ticker, e.label AS label, e.text AS text,
                   type(r) AS rel, e2.text AS related_text
            LIMIT $limit
            """,
            tickers=tickers,
            limit=limit,
        )
        for record in result:
            line = f"[{record['ticker']}] {record['label']}: {record['text']}"
            if record["rel"] and record["related_text"]:
                line += f" --{record['rel']}--> {record['related_text']}"
            facts.append(line)
    return facts


def trim_context(chunks: list[str], max_chars: int) -> str:
    joined = "\n\n---\n\n".join(chunks)
    if len(joined) <= max_chars:
        return joined
    return joined[:max_chars].rsplit("\n", 1)[0] + "\n...[trimmed]"


_compressor = None


def compress_with_llmlingua(text: str, rate: float) -> str:
    global _compressor
    try:
        if _compressor is None:
            from llmlingua import PromptCompressor

            _compressor = PromptCompressor(use_llmlingua2=True)
        result = _compressor.compress_prompt(text, rate=rate)
        return result.get("compressed_prompt", text)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] LLMLingua unavailable ({exc}); falling back to word-budget truncation.")
        words = text.split()
        keep = max(1, int(len(words) * rate))
        return " ".join(words[:keep])


CAVEMAN_SYSTEM_PROMPT = (
    "You terse answer machine. No filler word. No full sentence needed. "
    "Use fragment. State fact only from context. Short as possible."
)
CAVEMAN_USER_SUFFIX = "\n\nAnswer short. Fragments ok. Facts only from context above."


# ---------------------------------------------------------------------------
# Token counting + timing
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# RAG variants
# ---------------------------------------------------------------------------


def run_baseline(llm, collection, embed_model, question: str) -> dict:
    hits = vector_retrieve(collection, embed_model, question, BASELINE_TOP_K)
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."

    start = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start
    answer = response.content

    return {
        "answer": answer,
        "context": context,
        "contexts": [h["text"] for h in hits],
        "input_tokens": count_tokens(prompt),
        "output_tokens": count_tokens(answer),
        "latency_sec": latency,
    }


def run_kg(llm, collection, embed_model, driver, question: str, tickers: list[str]) -> dict:
    hits = vector_retrieve(collection, embed_model, question, BASELINE_TOP_K)
    graph_facts = graph_retrieve(driver, question, tickers)
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    if graph_facts:
        context += "\n\n--- Knowledge Graph Facts ---\n" + "\n".join(graph_facts)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."

    start = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start
    answer = response.content

    return {
        "answer": answer,
        "context": context,
        "contexts": [h["text"] for h in hits] + graph_facts,
        "input_tokens": count_tokens(prompt),
        "output_tokens": count_tokens(answer),
        "latency_sec": latency,
    }


def run_optimized(llm, collection, embed_model, driver, question: str, tickers: list[str]) -> dict:
    hits = vector_retrieve(collection, embed_model, question, OPTIMIZED_TOP_K)
    graph_facts = graph_retrieve(driver, question, tickers, limit=5)

    trimmed = trim_context([h["text"] for h in hits], MAX_CONTEXT_CHARS_OPTIMIZED)
    compressed = compress_with_llmlingua(trimmed, LLMLINGUA_RATE)

    context = compressed
    if graph_facts:
        context += "\n\nKG: " + " | ".join(graph_facts)

    prompt = f"{CAVEMAN_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQ: {question}{CAVEMAN_USER_SUFFIX}"

    start = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start
    answer = response.content

    return {
        "answer": answer,
        "context": context,
        "contexts": [context],
        "input_tokens": count_tokens(prompt),
        "output_tokens": count_tokens(answer),
        "latency_sec": latency,
    }


# ---------------------------------------------------------------------------
# DeepEval judge — local Ollama, JSON-mode via `instructor`
# ---------------------------------------------------------------------------


class OllamaDeepEvalLLM:
    """Minimal DeepEvalBaseLLM-compatible wrapper around a local Ollama model."""

    def __init__(self, model_id: str, base_url: str):
        from openai import AsyncOpenAI, OpenAI

        try:
            import instructor
        except ImportError:
            instructor = None

        self.model_id = model_id
        self._client = OpenAI(api_key="ollama", base_url=base_url)
        self._async_client = AsyncOpenAI(api_key="ollama", base_url=base_url)
        self._instructor_client = None
        self._async_instructor_client = None
        if instructor is not None:
            self._instructor_client = instructor.from_openai(self._client, mode=instructor.Mode.JSON)
            self._async_instructor_client = instructor.from_openai(self._async_client, mode=instructor.Mode.JSON)

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None or self._instructor_client is None:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        return self._instructor_client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            response_model=schema,
            temperature=0,
        )

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None or self._async_instructor_client is None:
            response = await self._async_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        return await self._async_instructor_client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            response_model=schema,
            temperature=0,
        )

    def get_model_name(self) -> str:
        return f"Ollama/{self.model_id}"


# ---------------------------------------------------------------------------
# Evaluation (ragas + deepeval)
# ---------------------------------------------------------------------------


def evaluate_variant(variant_name: str, results: list[dict]) -> dict:
    """Compute faithfulness, context precision, and hallucination for one variant's results."""
    faithfulness_scores: list[float] = []
    context_precision_scores: list[float] = []
    hallucination_scores: list[float] = []

    try:
        from langchain_ollama import ChatOllama
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

        evaluator_llm = LangchainLLMWrapper(ChatOllama(model=EVAL_MODEL, base_url=OLLAMA_BASE_URL, temperature=0))

        ragas_rows = [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"] or [r["context"]],
            }
            for r in results
        ]
        dataset = EvaluationDataset.from_list(ragas_rows)
        ragas_result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), LLMContextPrecisionWithoutReference()],
            llm=evaluator_llm,
        )
        df = ragas_result.to_pandas()
        faithfulness_scores = df["faithfulness"].dropna().tolist()
        context_precision_scores = df["llm_context_precision_without_reference"].dropna().tolist()
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] ragas evaluation failed for {variant_name}: {exc}")

    try:
        from deepeval.metrics import HallucinationMetric
        from deepeval.test_case import LLMTestCase

        deepeval_llm = OllamaDeepEvalLLM(EVAL_MODEL, f"{OLLAMA_BASE_URL}/v1")
        metric = HallucinationMetric(threshold=0.5, model=deepeval_llm, include_reason=False)

        for r in results:
            try:
                test_case = LLMTestCase(
                    input=r["question"],
                    actual_output=r["answer"],
                    context=r["contexts"] or [r["context"]],
                )
                metric.measure(test_case)
                hallucination_scores.append(metric.score)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] hallucination scoring failed for one question: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] deepeval evaluation failed for {variant_name}: {exc}")

    def avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "avg_faithfulness": avg(faithfulness_scores),
        "avg_context_precision": avg(context_precision_scores),
        "avg_hallucination": avg(hallucination_scores),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def summarize_variant(variant_name: str, results: list[dict]) -> dict:
    n = len(results)
    total_input = sum(r["input_tokens"] for r in results)
    total_output = sum(r["output_tokens"] for r in results)
    avg_latency = sum(r["latency_sec"] for r in results) / n if n else 0
    avg_cost = ((total_input / 1000) * COST_PER_1K_INPUT + (total_output / 1000) * COST_PER_1K_OUTPUT) / n if n else 0

    eval_scores = evaluate_variant(variant_name, results)

    return {
        "avg_input_tokens": round(total_input / n, 1) if n else 0,
        "avg_output_tokens": round(total_output / n, 1) if n else 0,
        "avg_latency_sec": round(avg_latency, 3),
        "avg_cost_usd": round(avg_cost, 6),
        **eval_scores,
    }


def pct_change(baseline: float | None, other: float | None, lower_is_better: bool = False) -> float | None:
    if baseline in (None, 0) or other is None:
        return None
    change = (other - baseline) / baseline * 100
    return round(-change if lower_is_better else change, 1)


def main() -> None:
    from neo4j import GraphDatabase

    print("=" * 70)
    print("SEC Filings Intelligence — Optimization + Evaluation (optimize.py)")
    print(f"Ollama: {OLLAMA_BASE_URL} (generation={GENERATION_MODEL}, eval={EVAL_MODEL})")
    print(f"Neo4j:  {NEO4J_URI} (db={NEO4J_DATABASE})")
    print("=" * 70)

    check_ollama_available()

    llm = get_llm()
    collection = get_chroma_collection()
    embed_model = get_embed_model()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    check_neo4j_available(driver)

    baseline_results, kg_results, optimized_results = [], [], []

    for i, q in enumerate(BENCHMARK_QUESTIONS, start=1):
        question = q["question"]
        tickers = [t.strip() for t in q["ticker"].split(",")]
        print(f"[{i}/{len(BENCHMARK_QUESTIONS)}] {question}")

        base = run_baseline(llm, collection, embed_model, question)
        base.update({"id": q["id"], "category": q["category"], "question": question})
        baseline_results.append(base)

        kg = run_kg(llm, collection, embed_model, driver, question, tickers)
        kg.update({"id": q["id"], "category": q["category"], "question": question})
        kg_results.append(kg)

        opt = run_optimized(llm, collection, embed_model, driver, question, tickers)
        opt.update({"id": q["id"], "category": q["category"], "question": question})
        optimized_results.append(opt)

    driver.close()

    print("[eval] scoring baseline ...")
    baseline_summary = summarize_variant("baseline", baseline_results)
    print("[eval] scoring KG-RAG ...")
    kg_summary = summarize_variant("kg", kg_results)
    print("[eval] scoring optimized ...")
    optimized_summary = summarize_variant("optimized", optimized_results)

    variants = {"baseline": baseline_summary, "kg": kg_summary, "optimized": optimized_summary}

    improvements = {
        "kg_vs_baseline": {
            "faithfulness_pct": pct_change(baseline_summary["avg_faithfulness"], kg_summary["avg_faithfulness"]),
            "context_precision_pct": pct_change(baseline_summary["avg_context_precision"], kg_summary["avg_context_precision"]),
            "hallucination_pct": pct_change(baseline_summary["avg_hallucination"], kg_summary["avg_hallucination"], lower_is_better=True),
        },
        "optimized_vs_baseline": {
            "faithfulness_pct": pct_change(baseline_summary["avg_faithfulness"], optimized_summary["avg_faithfulness"]),
            "context_precision_pct": pct_change(baseline_summary["avg_context_precision"], optimized_summary["avg_context_precision"]),
            "hallucination_pct": pct_change(baseline_summary["avg_hallucination"], optimized_summary["avg_hallucination"], lower_is_better=True),
            "input_tokens_pct": pct_change(baseline_summary["avg_input_tokens"], optimized_summary["avg_input_tokens"], lower_is_better=True),
            "output_tokens_pct": pct_change(baseline_summary["avg_output_tokens"], optimized_summary["avg_output_tokens"], lower_is_better=True),
            "cost_pct": pct_change(baseline_summary["avg_cost_usd"], optimized_summary["avg_cost_usd"], lower_is_better=True),
            "latency_pct": pct_change(baseline_summary["avg_latency_sec"], optimized_summary["avg_latency_sec"], lower_is_better=True),
        },
    }

    per_question = []
    for base, kg, opt in zip(baseline_results, kg_results, optimized_results):
        per_question.append(
            {
                "id": base["id"],
                "category": base["category"],
                "question": base["question"],
                "baseline_answer": base["answer"],
                "kg_answer": kg["answer"],
                "optimized_answer": opt["answer"],
                "baseline_input_tokens": base["input_tokens"],
                "optimized_input_tokens": opt["input_tokens"],
            }
        )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Generation via local Ollama (Neo4j may be local or AuraDB). Cost = $0.",
        "targets": TARGETS,
        "variants": variants,
        "improvements_vs_baseline": improvements,
        "per_question": per_question,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[done] wrote {RESULTS_PATH}")
    print("       Run `streamlit run app.py` to view the dashboard.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
