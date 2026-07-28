
"""
app.py — Streamlit demo dashboard for SEC Filings Intelligence POC.

Reads ./eval_results.json (written by optimize.py) and shows an interactive
comparison of Baseline RAG vs Knowledge-Graph RAG vs Optimized RAG across
accuracy/faithfulness, context precision, hallucination rate, token usage,
cost, and latency. Falls back to placeholder data if optimize.py hasn't
been run yet, so the dashboard always renders.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS_PATH = Path("./eval_results.json")

PLACEHOLDER_DATA = {
    "generated_at": None,
    "note": "Placeholder data — run `python optimize.py` to generate real results.",
    "targets": {
        "accuracy_improvement_pct": 25,
        "context_precision_improvement_pct": 30,
        "hallucination_reduction_pct": 40,
        "input_token_reduction_pct": 50,
        "output_token_reduction_pct": 40,
        "cost_reduction_pct": 50,
        "latency_reduction_pct": 25,
    },
    "variants": {
        "baseline": {
            "avg_faithfulness": 0.62, "avg_context_precision": 0.58, "avg_hallucination": 0.35,
            "avg_input_tokens": 1800, "avg_output_tokens": 220, "avg_latency_sec": 4.2, "avg_cost_usd": 0.0,
        },
        "kg": {
            "avg_faithfulness": 0.74, "avg_context_precision": 0.71, "avg_hallucination": 0.24,
            "avg_input_tokens": 2100, "avg_output_tokens": 230, "avg_latency_sec": 4.8, "avg_cost_usd": 0.0,
        },
        "optimized": {
            "avg_faithfulness": 0.79, "avg_context_precision": 0.76, "avg_hallucination": 0.19,
            "avg_input_tokens": 850, "avg_output_tokens": 130, "avg_latency_sec": 2.9, "avg_cost_usd": 0.0,
        },
    },
    "improvements_vs_baseline": {
        "kg_vs_baseline": {"faithfulness_pct": 19.4, "context_precision_pct": 22.4, "hallucination_pct": 31.4},
        "optimized_vs_baseline": {
            "faithfulness_pct": 27.4, "context_precision_pct": 31.0, "hallucination_pct": 45.7,
            "input_tokens_pct": 52.8, "output_tokens_pct": 40.9, "cost_pct": 0.0, "latency_pct": 31.0,
        },
    },
    "per_question": [],
}

st.set_page_config(page_title="SEC Filings Intelligence — US Airlines", layout="wide", page_icon="✈️")


@st.cache_data(ttl=5)
def load_results() -> dict:
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return PLACEHOLDER_DATA


def metric_card(col, label: str, baseline_val, other_val, unit: str = "", lower_is_better: bool = False):
    if baseline_val is None or other_val is None:
        col.metric(label, "n/a")
        return
    delta = other_val - baseline_val
    delta_color = "inverse" if lower_is_better else "normal"
    col.metric(label, f"{other_val:.2f}{unit}", f"{delta:+.2f}{unit} vs baseline", delta_color=delta_color)


def main() -> None:
    data = load_results()
    variants = data["variants"]
    targets = data["targets"]
    improvements = data.get("improvements_vs_baseline", {})

    st.title("✈️ SEC Filings Intelligence — US Airlines")
    st.caption("Local Ollama LLM + Neo4j + ChromaDB")

    if data.get("generated_at"):
        st.caption(f"Results generated: {data['generated_at']}")
    else:
        st.info(data.get("note", "Placeholder data shown — run `python optimize.py` to generate real results."))

    with st.expander("About this comparison", expanded=False):
        st.markdown(
            """
**Baseline RAG** — plain vector similarity search over chunked SEC filings (10-K/10-Q for
UAL, DAL, AAL, LUV), no graph, no compression.

**KG-RAG** — adds a Neo4j knowledge graph (Company/Route/Revenue/Expense/Risk
entities and relationships extracted with LangExtract) and does one-hop graph traversal
alongside vector search.

**Optimized RAG** — tighter top-k retrieval, context trimming, LLMLingua prompt
compression, and a terse "Caveman" system prompt, aimed at cutting token usage and
latency while holding or improving faithfulness.

All three variants run against the same local Ollama model — the differences below are
purely from retrieval and prompting strategy, not the underlying LLM.
            """
        )

    tab_overview, tab_tokens, tab_explorer, tab_raw = st.tabs(
        ["📊 Overview", "💰 Token & Cost", "🔍 Per-Question Explorer", "🗂️ Raw Data"]
    )

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------
    with tab_overview:
        st.subheader("Quality metrics: Baseline vs KG-RAG vs Optimized RAG")

        b, k, o = variants["baseline"], variants["kg"], variants["optimized"]

        c1, c2, c3 = st.columns(3)
        metric_card(c1, "Faithfulness (baseline)", b["avg_faithfulness"], b["avg_faithfulness"])
        metric_card(c2, "Faithfulness (KG-RAG)", b["avg_faithfulness"], k["avg_faithfulness"])
        metric_card(c3, "Faithfulness (Optimized)", b["avg_faithfulness"], o["avg_faithfulness"])

        c1, c2, c3 = st.columns(3)
        metric_card(c1, "Context Precision (baseline)", b["avg_context_precision"], b["avg_context_precision"])
        metric_card(c2, "Context Precision (KG-RAG)", b["avg_context_precision"], k["avg_context_precision"])
        metric_card(c3, "Context Precision (Optimized)", b["avg_context_precision"], o["avg_context_precision"])

        c1, c2, c3 = st.columns(3)
        metric_card(c1, "Hallucination (baseline)", b["avg_hallucination"], b["avg_hallucination"], lower_is_better=True)
        metric_card(c2, "Hallucination (KG-RAG)", b["avg_hallucination"], k["avg_hallucination"], lower_is_better=True)
        metric_card(c3, "Hallucination (Optimized)", b["avg_hallucination"], o["avg_hallucination"], lower_is_better=True)

        fig = go.Figure()
        metrics = ["avg_faithfulness", "avg_context_precision", "avg_hallucination"]
        labels = ["Faithfulness", "Context Precision", "Hallucination"]
        for name, values in [("Baseline", b), ("KG-RAG", k), ("Optimized", o)]:
            fig.add_trace(go.Bar(name=name, x=labels, y=[values.get(m) or 0 for m in metrics]))
        fig.update_layout(barmode="group", title="Quality metrics by variant", yaxis_title="Score (0-1)")
        st.plotly_chart(fig, use_container_width=True)

        opt_vs_base = improvements.get("optimized_vs_baseline", {})
        st.subheader("Progress vs. planning.md targets")
        target_rows = [
            ("Faithfulness / accuracy", opt_vs_base.get("faithfulness_pct"), targets["accuracy_improvement_pct"]),
            ("Context precision", opt_vs_base.get("context_precision_pct"), targets["context_precision_improvement_pct"]),
            ("Hallucination reduction", opt_vs_base.get("hallucination_pct"), targets["hallucination_reduction_pct"]),
            ("Input token reduction", opt_vs_base.get("input_tokens_pct"), targets["input_token_reduction_pct"]),
            ("Output token reduction", opt_vs_base.get("output_tokens_pct"), targets["output_token_reduction_pct"]),
            ("Latency reduction", opt_vs_base.get("latency_pct"), targets["latency_reduction_pct"]),
        ]
        df_targets = pd.DataFrame(target_rows, columns=["Metric", "Actual (%)", "Target (%)"])
        st.dataframe(df_targets, use_container_width=True, hide_index=True)

        if all((variants[v]["avg_cost_usd"] or 0) == 0 for v in variants):
            st.info("Cost is $0 across all variants because generation runs entirely on a local model (Ollama) — no paid API calls.")

    # ------------------------------------------------------------------
    # Token & Cost
    # ------------------------------------------------------------------
    with tab_tokens:
        st.subheader("Token usage by variant")

        fig_tokens = go.Figure()
        for name, values in [("Baseline", b), ("KG-RAG", k), ("Optimized", o)]:
            fig_tokens.add_trace(
                go.Bar(name=name, x=["Input tokens", "Output tokens"],
                       y=[values["avg_input_tokens"], values["avg_output_tokens"]])
            )
        fig_tokens.update_layout(barmode="group", title="Avg tokens per question", yaxis_title="Tokens")
        st.plotly_chart(fig_tokens, use_container_width=True)

        fig_latency = go.Figure()
        fig_latency.add_trace(
            go.Bar(
                x=["Baseline", "KG-RAG", "Optimized"],
                y=[b["avg_latency_sec"], k["avg_latency_sec"], o["avg_latency_sec"]],
            )
        )
        fig_latency.update_layout(title="Avg latency per question (sec)", yaxis_title="Seconds")
        st.plotly_chart(fig_latency, use_container_width=True)

        st.subheader("If this were running on a paid API instead")
        st.caption("This POC uses $0 generation cost (local Ollama). Enter hypothetical rates to see what the token savings would be worth on a paid provider.")

        col1, col2, col3 = st.columns(3)
        rate_in = col1.number_input("$ per 1K input tokens", min_value=0.0, value=0.0005, step=0.0001, format="%.4f")
        rate_out = col2.number_input("$ per 1K output tokens", min_value=0.0, value=0.0015, step=0.0001, format="%.4f")
        queries_per_month = col3.number_input("Queries per month", min_value=1, value=10000, step=1000)

        def hypothetical_monthly_cost(values):
            per_query = (values["avg_input_tokens"] / 1000) * rate_in + (values["avg_output_tokens"] / 1000) * rate_out
            return per_query * queries_per_month

        base_cost = hypothetical_monthly_cost(b)
        opt_cost = hypothetical_monthly_cost(o)
        savings = base_cost - opt_cost

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline — monthly cost", f"${base_cost:,.2f}")
        c2.metric("Optimized — monthly cost", f"${opt_cost:,.2f}")
        c3.metric("Estimated monthly savings", f"${savings:,.2f}", f"{(savings / base_cost * 100) if base_cost else 0:.1f}%")

    # ------------------------------------------------------------------
    # Per-Question Explorer
    # ------------------------------------------------------------------
    with tab_explorer:
        st.subheader("Per-question answers across variants")
        per_question = data.get("per_question", [])
        if not per_question:
            st.warning("No per-question data yet — run `python optimize.py` to populate this tab.")
        else:
            categories = sorted({q["category"] for q in per_question})
            selected_cat = st.selectbox("Filter by category", ["All"] + categories)
            filtered = per_question if selected_cat == "All" else [q for q in per_question if q["category"] == selected_cat]

            for q in filtered:
                with st.expander(f"[{q['category']}] {q['question']}"):
                    st.markdown(f"**Baseline** ({q.get('baseline_input_tokens', '?')} input tokens)")
                    st.write(q["baseline_answer"])
                    st.markdown("**KG-RAG**")
                    st.write(q["kg_answer"])
                    st.markdown(f"**Optimized** ({q.get('optimized_input_tokens', '?')} input tokens)")
                    st.write(q["optimized_answer"])

    # ------------------------------------------------------------------
    # Raw Data
    # ------------------------------------------------------------------
    with tab_raw:
        st.subheader("Raw eval_results.json")
        st.json(data)


if __name__ == "__main__":
    main()
    
        