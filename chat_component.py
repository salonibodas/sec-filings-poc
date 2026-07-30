import os
import streamlit as st
from groq import Groq

def show_chat():
    st.divider()
    st.subheader("💬 SEC Filings Intelligence Assistant (Live Online)")

    # Retrieve API key from Streamlit Secrets or Environment
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

    if not api_key:
        st.warning("Please configure your `GROQ_API_KEY` in Streamlit App Secrets to enable the live chatbot.")
        return

    client = Groq(api_key=api_key)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Ask anything about airline SEC filings, RAG, or financial metrics..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call Live Groq API (Llama 3 model)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Format system context
            system_prompt = {
                "role": "system",
                "content": (
                    "You are an expert AI assistant specializing in SEC Filings (10-K/10-Q) for US Airlines "
                    "(UAL, DAL, AAL, LUV) and RAG architecture optimization. Provide clear, accurate, and concise answers."
                )
            }
            
            api_messages = [system_prompt] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            # Stream response live
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=api_messages,
                stream=True,
            )

            full_response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
