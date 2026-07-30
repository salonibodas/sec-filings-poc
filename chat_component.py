import json
from pathlib import Path
import streamlit as st

RESULTS_PATH = Path("./eval_results.json")

def load_eval_questions():
    """Loads benchmark questions directly from eval_results.json"""
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                data = json.load(f)
                return (
                    data.get("per_question")
                    or data.get("benchmark_dataset")
                    or data.get("questions")
                    or []
                )
        except Exception:
            return []
    return []

def show_chat():
    st.divider()
    st.subheader("💬 SEC Filings Intelligence Assistant (Offline Demo)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about SEC filings or evaluation metrics..."):
        # Display user question
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        questions = load_eval_questions()
        user_words = set(prompt.lower().split())

        matched_q = None
        best_score = 0

        # Keyword overlap matching
        for q in questions:
            q_text = q.get("question", "").lower()
            # Count how many words match
            overlap = sum(1 for word in user_words if word in q_text and len(word) > 3)
            if overlap > best_score:
                best_score = overlap
                matched_q = q

        # If we found a reasonable match (at least 1 key keyword matched)
        if matched_q and best_score >= 1:
            ans = matched_q.get("optimized_answer") or matched_q.get("baseline_answer") or "No answer recorded."
            category = matched_q.get("category", "General")
            
            response = (
                f"**Matched Benchmark Question:** *\"{matched_q.get('question')}\"*\n\n"
                f"**Category:** `{category}`\n\n"
                f"**Optimized RAG Answer:**\n{ans}"
            )
        else:
            response = (
                "I couldn't find a matching question in your local `eval_results.json` dataset.\n\n"
                "**Try asking terms like:** `Delta`, `fuel`, `PRASM`, `United`, or `latency`."
            )

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
