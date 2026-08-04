import os
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def retrieve_context(query_text):
    """Retrieves relevant SEC filing chunks from your local ChromaDB store."""
    try:
        if os.path.exists("./chroma_data"):
            # Load your embeddings model (adjust if you used a different embedding model)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma(persist_directory="./chroma_data", embedding_function=embeddings)
            
            # Retrieve top 3 matching chunks
            docs = vectorstore.similarity_search(query_text, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            return context
    except Exception as e:
        st.error(f"Error loading vector store context: {e}")
    return ""

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

        # Fetch context from ChromaDB
        context = retrieve_context(prompt)

        # Updated System Prompt with Context & 2026 Scope Instruction
        system_prompt_content = (
            "You are an expert AI assistant specializing in SEC Filings (10-K/10-Q/8-K) for US Airlines "
            "(UAL, DAL, AAL, LUV) and RAG architecture optimization. You have access to filings updated through 2026.\n\n"
            "Answer the user's question accurately based on the retrieved SEC filing context provided below. "
            "Do NOT state that your knowledge is limited to 2023.\n\n"
            f"--- RETRIEVED SEC FILINGS CONTEXT ---\n{context if context else 'No relevant vector context found.'}\n--------------------------------------"
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            system_prompt = {
                "role": "system",
                "content": system_prompt_content
            }
            
            api_messages = [system_prompt] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            # Call Live Groq API
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
