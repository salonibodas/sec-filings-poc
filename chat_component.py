import streamlit as st

def show_chat():
    st.divider()
    st.subheader("💬 SEC Filings Intelligence Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about airlines SEC filings or evaluation metrics..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Load your existing evaluation results
        data = st.session_state.get("eval_data", {})
        questions = data.get("per_question", [])
        
        # Search for a matching question in your local benchmark dataset
        matched_q = next((q for q in questions if prompt.lower() in q["question"].lower()), None)

        if matched_q:
            response = (
                f"**Question:** {matched_q['question']}\n\n"
                f"**Optimized RAG Answer:**\n{matched_q.get('optimized_answer', 'N/A')}\n\n"
                f"*Tokens used: {matched_q.get('optimized_input_tokens', '?')} input tokens.*"
            )
        else:
            response = (
                "I am currently operating in **Offline Demo Mode**. "
                "I can answer questions present in your `eval_results.json` benchmark dataset. "
                "Try asking about fuel risks, PRASM trends, or RAG token latency!"
            )

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
