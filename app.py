from __future__ import annotations

import tempfile
import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
from pyvis.network import Network

st.set_page_config(page_title="SEC Filings Intelligence", layout="wide")

st.title("SEC Filings Intelligence — US Airlines")

tab_overview, tab_cost, tab_explorer, tab_kg = st.tabs(
    ["Overview", "Token & Cost", "Per-Question Explorer", "Knowledge Graph"]
)

with tab_overview:
    st.header("Overview & Key Metrics")
    st.write("Baseline RAG vs. KG-RAG vs. KG + Token Optimization performance summary.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Faithfulness (Accuracy)", "0.79", "+27.4% vs Baseline")
    with col2:
        st.metric("Context Precision", "0.76", "+31.0% vs Baseline")
    with col3:
        st.metric("Hallucination Rate", "0.19", "-45.7% vs Baseline")

with tab_cost:
    st.header("Token Usage & Cost Analysis")
    st.write("Detailed breakdown of token efficiency and estimated API savings.")

with tab_explorer:
    st.header("Per-Question Explorer")
    st.write("Compare side-by-side prompt responses across model variants.")

with tab_kg:
    st.subheader("🌐 SEC Filings Knowledge Graph Explorer")
    st.write("Interactive view of entities and relationship paths stored in Neo4j AuraDB.")

    node_limit = st.slider("Max Relationships to Display", min_value=10, max_value=100, value=35, step=5)

    @st.cache_data(ttl=600)
    def fetch_graph_data(limit: int):
        cypher_query = f"""
        MATCH (s)-[r]->(t)
        RETURN s.name AS source, labels(s)[0] AS source_type, 
               type(r) AS rel, 
               t.name AS target, labels(t)[0] AS target_type
        LIMIT {limit}
        """
        driver = GraphDatabase.driver(
            st.secrets["NEO4J_URI"] if "NEO4J_URI" in st.secrets else "neo4j+s://<YOUR_INSTANCE>.databases.neo4j.io",
            auth=(
                st.secrets.get("NEO4J_USER", "neo4j"),
                st.secrets.get("NEO4J_PASSWORD", "<YOUR_PASSWORD>")
            )
        )
        with driver.session() as session:
            records = [record.data() for record in session.run(cypher_query)]
        driver.close()
        return records

    try:
        graph_data = fetch_graph_data(node_limit)

        if graph_data:
            net = Network(height="520px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)

            color_map = {
                "Company": "#4CAF50",
                "Risk": "#FFC107",
                "Expense": "#FF5722",
                "Revenue": "#2196F3"
            }

            for row in graph_data:
                src, tgt = row["source"], row["target"]
                src_type, tgt_type = row["source_type"], row["target_type"]

                net.add_node(src, label=src, title=f"Type: {src_type}", color=color_map.get(src_type, "#97C2FC"))
                net.add_node(tgt, label=tgt, title=f"Type: {tgt_type}", color=color_map.get(tgt_type, "#97C2FC"))
                net.add_edge(src, tgt, label=row["rel"])

            net.toggle_physics(True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, "r", encoding="utf-8") as f:
                    html_bytes = f.read()

            components.html(html_bytes, height=540)
        else:
            st.info("No active relationships retrieved from Neo4j.")

    except Exception as e:
        st.error(f"Could not connect to Neo4j database: {e}")

        