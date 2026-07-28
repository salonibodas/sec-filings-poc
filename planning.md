# SEC Filings Intelligence for US Airlines — GenAI POC

**Knowledge Graph + RAG + Token Optimization**
Improve multi-document answer quality while cutting token cost by 40–60%.

---

## 1. Business Problem

Airline executives and analysts need answers drawn from multiple SEC filings (10-K, 10-Q, 8-K, earnings calls, investor presentations). Relevant information is scattered across documents and hard to connect, especially for causal, comparative, or trend-based questions.

Example: *"Why did operating margins decline at United despite passenger growth?"* — no single section of a single filing answers this; it requires connecting revenue, cost, and risk-factor sections, potentially across filings.

The POC combines a Knowledge Graph with RAG and token-optimization techniques to answer these questions accurately and cheaply.

---

## 2. Goals & Success Targets

| Target Metric | Goal | Evaluation Tool |
|---|---|---|
| Answer Accuracy | **+25%** | Human review (spot check) |
| Context Precision | +30% | RAGAS |
| Faithfulness | +25% | RAGAS |
| Hallucination Rate | -40% | DeepEval |
| **Token Consumption** | **-50%** | Token/cost tracking |
| Output Tokens (avg) | -40% | Token/cost tracking |
| Cost per Query | -50% | Token/cost tracking |
| Response Latency | -25% | Token/cost tracking |

All targets are measured relative to the Week 1 baseline RAG system. The two headline goals for this POC are **+25% answer accuracy** and **-50% token consumption**, achieved by Week 4 via the KG + Optimization pipeline.

---

## 3. Dataset & Sources

**Source:** SEC EDGAR (via SEC EDGAR API)

**Airlines covered (tickers):**
- United Airlines Holdings (**UAL**)
- Delta Air Lines (**DAL**)
- American Airlines Group (**AAL**)
- Southwest Airlines (**LUV**)

**Documents to ingest:**

| Document Type | Coverage |
|---|---|
| 10-K | Last 5 years |
| 10-Q | Last 3 years |
| 8-K | Major events |
| Earnings call transcripts | Last 12 quarters |
| Investor presentations | Available decks |

**Total:** 100+ documents across 4 airlines · **Format:** PDF, TXT

---

## 4. Solution Architecture

1. **Data Ingestion** — SEC EDGAR API → raw filings (PDF/TXT)
2. **Document Processing** — text extraction (OCR if needed), chunking, metadata enrichment
3. **Knowledge Extraction** — entity extraction (LangExtract), relationship extraction
4. **Knowledge Storage** — vector database (ChromaDB / Pinecone) + knowledge graph (Neo4j AuraDB)
5. **Hybrid Retrieval** — vector search (semantic) + graph search (relationships)
6. **Compression Layer** — context compression (LangChain), prompt compression (LLMLingua), Caveman prompting, query rewriting
7. **LLM Options** — OpenAI/GPT-4o, Mistral 7B, Llama 3
8. **Response Optimization** — Caveman-style concise responses, structured output ("only what matters")
9. **Evaluation & Monitoring** — RAGAS (quality), DeepEval (hallucination), Weights & Biases (experiment tracking), Streamlit dashboard

### Tech Stack (all free/low-cost, cloud-based)

| Layer | Tool |
|---|---|
| Development environment | Google Colab |
| LLMs & models | Hugging Face Hub |
| RAG framework | LlamaIndex |
| Vector database | ChromaDB (free) |
| Knowledge graph | Neo4j AuraDB (free) |
| Entity/relation extraction | LangExtract |
| Prompt compression | LLMLingua |
| Compression retriever | LangChain |
| Evaluation | RAGAS, DeepEval |
| Experiment tracking | Weights & Biases |
| Application UI | Streamlit |

**Estimated POC cost:** $0–$100 (open models / low-cost APIs).

---

## 5. 4-Week Roadmap

### Week 1 — Baseline RAG

- Collect filings via SEC EDGAR API for UAL, DAL, AAL, LUV
- Parse and clean documents; chunk and create embeddings
- Store in vector DB (ChromaDB)
- Build baseline RAG system (retrieval + generation, no graph)
- Draft the benchmark question set (see Section 6)

**Deliverable:** Baseline SEC Filings Assistant
**Output:** Baseline metrics — accuracy, tokens, latency, cost — used as the comparison floor for all later weeks

### Week 2 — Knowledge Graph Enhancement

- Extract entities (Company, Route, Aircraft, Revenue, Expense, Risk, etc.)
- Extract relationships between entities
- Build the knowledge graph in Neo4j AuraDB
- Implement hybrid retrieval (vector + graph)
- Compare answer quality vs. Week 1 baseline, especially on multi-document and causal questions

**Deliverable:** KG-Enhanced Assistant
**Output:** Improved precision, recall, and answer quality vs. baseline

### Week 3 — Token Optimization

- Implement Caveman prompting (short, direct, no fluff)
- Context compression (top-k + re-rank + summarize)
- Query rewriting for better retrieval
- Document summary cache (use summaries instead of full text where possible)
- Prompt compression via LLMLingua
- Response compression (structured output: key points, numbers, tables only)
- Track token usage and cost per query throughout

**Deliverable:** Optimized Assistant (Low Cost)
**Output:** Reduced tokens and cost with minimal quality loss

**Expected token reduction by technique:**

| Technique | Expected Reduction |
|---|---|
| Caveman prompting | 20–40% |
| Context compression | 30–50% |
| Query rewriting | 10–20% |
| KG-guided retrieval | 40–60% |
| Document summarization cache | 50–70% |
| Structured response compression | 30–50% |
| **Overall target** | **40–60%** |

### Week 4 — Evaluation & Finalization

- Finalize the evaluation set (target: 300 analyst-style questions, expanded from the seed set in Section 6)
- Run all three system variants — Baseline RAG vs. KG RAG vs. KG + Optimization — on the same question set
- Score with RAGAS (faithfulness, context precision) and DeepEval (hallucination rate)
- Build the token/cost/quality comparison dashboard in Streamlit
- Prepare the final report and executive demo (use the 5 Gold Standard questions in Section 6.5)

**Deliverable:** Final Report & Executive Demo
**Output:** Documented quality improvement and cost reduction results against the Section 2 targets

---

## 6. Evaluation Dataset

The evaluation set is organized by reasoning complexity, deliberately structured to expose where baseline RAG struggles and where the Knowledge Graph adds value. Each category below is tagged with its type: **Single-Document**, **Multi-Document**, or **Causal Reasoning**.

### 6.1 Single-Document Questions (Baseline — Type: Single-Document)

Answerable from one section of one filing; any competent RAG system should get these right. Used to confirm the baseline system works before layering on complexity.

**Financial:**
1. What was United's (UAL) operating revenue in FY2024?
2. What percentage of Delta's (DAL) revenue came from passenger operations?
3. What were Southwest's (LUV) fuel expenses in 2024?
4. How much debt did American Airlines (AAL) repay in 2024?
5. What was United's (UAL) CASM (Cost per Available Seat Mile)?

**Risk Factors:**
6. What cyber risk factors did Delta (DAL) identify?
7. What cybersecurity risks were disclosed by United (UAL)?
8. What risks did Southwest (LUV) mention regarding fuel prices?

### 6.2 Multi-Section Questions (Type: Single-Document, cross-section)

Still confined to one filing, but require connecting different sections within it (e.g., MD&A + risk factors). A harder single-document tier that starts to expose baseline RAG's chunk-isolation weakness.

9. Why did United's (UAL) operating margin decline despite revenue growth?
10. How did fuel costs impact profitability? (per airline)
11. What factors contributed to increased labor expenses? (per airline)
12. How do debt redemption rules affect future earnings? (per airline)
13. How does fleet modernization affect long-term margins? (per airline)

### 6.3 Multi-Document Questions (Type: Multi-Document)

Require synthesizing information across multiple filings or time periods for the same airline. This is where the Knowledge Graph's value should become visible against the baseline.

14. How has Delta's (DAL) strategy changed over the last 5 years?
15. What recurring risk factors appeared across United's (UAL) last three annual reports?
16. How has discussion of labor challenges evolved since COVID? (per airline)
17. What common themes appear across recent earnings calls? (per airline)
18. How has Southwest's (LUV) fleet strategy changed over time?

### 6.4 Cross-Company Comparison Questions (Type: Multi-Document, cross-entity)

Require pulling facts from multiple airlines' filings and comparing them — strong candidates for executive demos since they showcase graph relationships across companies.

19. Compare fuel risk management strategies of United (UAL) and Delta (DAL).
20. Compare fleet modernization investments across all four airlines (UAL, DAL, AAL, LUV).
21. Which airline appears most dependent on international traffic?
22. Compare cybersecurity risk disclosures between Delta (DAL) and Southwest (LUV).
23. How do debt levels compare across all four major US airlines?
24. Which airline disclosed the highest number of strategic risks?

### 6.5 Analyst-Level Questions (Type: Multi-Document, synthesis)

Modeled on questions a Wall Street analyst would ask; combine financial, risk, and strategic reasoning across documents and companies.

25. What are the primary drivers of margin pressure at United (UAL)?
26. What risks could negatively impact Delta's (DAL) future earnings?
27. Which airline is best positioned for international growth?
28. How are airlines mitigating labor cost inflation?
29. What common themes emerge from recent earnings calls across all four airlines?
30. What factors are most likely to impact 2025 profitability?

### 6.6 Causal Reasoning Questions (Type: Causal Reasoning)

Require tracing chains of cause and effect — the strongest demonstration of Knowledge Graph value, since these depend on explicit entity relationships rather than semantic similarity alone.

31. Why did fuel expenses increase? (per airline)
32. Why are labor costs rising? (per airline)
33. What chain of events caused higher maintenance costs?
34. Why are maintenance expenses increasing?
35. What factors led to reduced capacity?

**Reference causal chain (KG example):** Flight Delay → Passenger Rebooking → Fleet Availability → Cost Impact

### 6.7 Gold Standard Demo Questions

Reserved for the Week 4 executive demo — each spans multiple companies/documents and requires causal reasoning, so together they showcase multi-document synthesis, relationship mapping, KG value, token reduction, and executive-level insight in one sitting.

1. Why did United's operating margin decline despite passenger growth? *(Multi-Section + Causal)*
2. Compare labor-related risks across United, Delta, and Southwest. *(Multi-Document, cross-company)*
3. What factors are most frequently linked to profitability declines across airline filings? *(Multi-Document + Causal)*
4. How has fleet modernization strategy evolved over the last three years? *(Multi-Document, trend)*
5. What are the top operational and financial risks likely to affect airline profitability over the next three years? *(Multi-Document, forward-looking synthesis)*

### 6.8 Scaling to the Full Evaluation Set

Week 4 calls for a 300-question evaluation set. Scale the seed questions above by:
- Repeating each single-document and causal-reasoning template across all four tickers (UAL, DAL, AAL, LUV)
- Repeating each multi-document/comparison template across multiple filing years (FY2020–FY2024) and multiple 2-4 airline combinations
- Maintaining roughly this category mix: 30% single-document, 45% multi-document (including cross-company and analyst-level), 25% causal reasoning — so the set continues to stress-test the KG's advantage over plain RAG rather than skew toward the easy tier

---

## 7. Evaluation Framework

| Metric | Baseline RAG | KG RAG | KG + Optimization (Goal) | Target Improvement | Tool |
|---|---|---|---|---|---|
| Faithfulness | — | — | — | +25% | RAGAS |
| Context Precision | — | — | — | +30% | RAGAS |
| Answer Accuracy | — | — | — | +25% | Human review (spot check) |
| Hallucination Rate | — | — | — | -40% | DeepEval |
| Input Tokens (avg) | — | — | — | -50% | Token tracking |
| Output Tokens (avg) | — | — | — | -40% | Token tracking |
| Cost per Query | — | — | — | -50% | Token tracking |
| Latency (sec) | — | — | — | -25% | Token tracking |

*(Baseline / KG RAG / KG+Optimization columns are populated as each week's results come in — Week 1 fills the Baseline column, Week 2 fills KG RAG, Week 3–4 fill KG+Optimization.)*

---

## 8. Final Deliverables

**Technical:**
- SEC filing ingestion pipeline
- Knowledge Graph (Neo4j)
- Baseline RAG system
- KG-enhanced system
- Token optimization modules
- Evaluation framework
- Token & cost dashboard
- Streamlit demo app

**Business:**
- Quality improvement report
- Cost reduction analysis
- Comparative results (Baseline vs. KG vs. KG+Optimization)
- Architecture & design doc
- Executive presentation
- ROI & impact summary

---

## 9. Why This POC

Demonstrates how Knowledge Graphs enable multi-document reasoning and how token optimization techniques significantly reduce GenAI costs while maintaining or improving answer quality — directly against the +25% accuracy / -50% token targets in Section 2.
