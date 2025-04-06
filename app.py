import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")


# App title
st.markdown("<h1 style='margin-bottom: 0.5rem;'>SHL Assessment Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; margin-bottom: 2rem;'>Find optimal assessments based on job requirements</p>", unsafe_allow_html=True)

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

# Load product data
df = pd.read_csv("data/product.csv")
documents = df.to_dict(orient="records")
for doc in documents:
    if "description" in doc and isinstance(doc["description"], str):
        doc["description"] = doc["description"][:50]
    else:
        doc["description"] = ""

text_documents = [str(doc) for doc in documents]
doc_map = {str(doc): doc for doc in documents}

def get_recommendations(query):
    all_matches = []
    errors = []
    batch_size = 5

    for i in range(0, len(text_documents), batch_size):
        batch_docs = text_documents[i:i+batch_size]
        try:
            reranked = pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=batch_docs,
                top_n=min(6, len(batch_docs)),
                return_documents=True
            )
            all_matches.extend(reranked.data)
        except Exception as e:
            for doc_text in batch_docs:
                doc = doc_map.get(doc_text, {})
                errors.append(doc.get("name", "Unknown"))

    sorted_matches = sorted(all_matches, key=lambda x: x.score, reverse=True)
    return [doc_map.get(match.document.text, {}) for match in sorted_matches[:10]], errors

# Input selection
input_mode = st.radio("Select Input Type", ("URL", "Query"), horizontal=True)

def display_results(results, errors):
    if results:
        st.subheader("Recommended Assessments")
        # Prepare table data
        table_data = []
        for idx, item in enumerate(results, 1):
            table_data.append({
                "#": idx,
                "Assessment": f"[{item.get('name', '')}]({item.get('url', '')})",
                "Job Levels": item.get("job_levels", ""),
                "Languages": item.get("languages", ""),
                "Duration": item.get("assessment_length", ""),
                "Remote": "✅" if item.get("remote_testing") else "❌",
                "Adaptive IRT": "✅" if item.get("adaptive_irt") else "❌",
                "Test Types": item.get("test_types", "")
            })
        
        # Create and display dataframe
        df = pd.DataFrame(table_data)[["#", "Assessment", "Job Levels", "Languages",
                                      "Duration", "Remote", "Adaptive IRT", "Test Types"]]
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
    
    # Display errors if any
    if errors:
        with st.expander("Missed Assessments (Description Too Large)"):
            for error in errors:
                st.markdown(f"- {error}")

if input_mode == "URL":
    url_input = st.text_input("Enter URL", placeholder="Paste job description URL")
    if st.button("Analyze URL"):
        if url_input:
            with st.spinner("Analyzing URL..."):
                try:
                    response = requests.get(url_input, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    query = text[:1000]
                    results, errors = get_recommendations(query)
                    display_results(results, errors)
                except Exception as e:
                    st.error(f"Failed to fetch or parse content: {e}")
        else:
            st.error("Please provide a URL.")
else:
    query_input = st.text_area("Enter your query", placeholder="e.g., Senior Java Developer", height=150)
    if st.button("Analyze Text"):
        if query_input:
            with st.spinner("Analyzing Text..."):
                results, errors = get_recommendations(query_input)
                display_results(results, errors)
        else:
            st.error("Please provide a query.")