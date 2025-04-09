from flask import Flask, request, jsonify, render_template
import pandas as pd
from pinecone import Pinecone
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import re
load_dotenv()
app = Flask(__name__)
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
df = pd.read_csv("data/product.csv")
documents = df.to_dict(orient="records")
for doc in documents:
    if "description" in doc and isinstance(doc["description"], str):
        doc["description"] = doc["description"][:50]
    else:
        doc["description"] = ""
text_documents = [str(doc) for doc in documents]
doc_map = {str(doc): doc for doc in documents}
def is_url(s):
    if not isinstance(s, str):
        return False
    return re.match(r'^https?://\S+', s) is not None
@app.route('/')
def home():
    return render_template('input.html')
@app.get('/health')
def health_func():
    return jsonify({"status": "healthy"})
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query_input = data.get("query")
    if not query_input:
        return jsonify({"error": "Missing query parameter"}), 400
    query = ""
    if is_url(query_input):
        try:
            response = requests.get(query_input, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            query = text[:1000]
        except Exception as e:
            return jsonify({"error": f"Failed to fetch or parse URL: {str(e)}"}), 500
    else:
        query = query_input[:1000]

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
    top_results = []
    for match in sorted_matches[:10]:
        doc = doc_map.get(match.document.text, {})
        top_results.append({
            "name": doc.get("name", ""),
            "url": doc.get("url", ""),
            "job_levels": doc.get("job_levels", ""),
            "languages": doc.get("languages", ""),
            "assessment_length": doc.get("assessment_length", ""),
            "remote_testing": doc.get("remote_testing", ""),
            "adaptive_irt": doc.get("adaptive_irt", ""),
            "test_types": doc.get("test_types", "")
        })
    return jsonify({"results": top_results})
if __name__ == '__main__':
    app.run(debug=True)