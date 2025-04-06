from flask import Flask, request, jsonify, render_template
import pandas as pd
from pinecone import Pinecone
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()
# Load environment variables from .env file 

app = Flask(__name__)

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

@app.route('/recommend_from_url', methods=['POST'])
def recommend_from_url():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "Missing url parameter"}), 400

    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        query = text[:1000]  # Use only the first 1000 characters for brevity
        #print(f"Extracted query from URL: {query}")
        # Forward the extracted query to /recommend logic
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
                #print(f"Error during reranking: {e}")
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

        return jsonify({"results": top_results, "errors": errors})

    except Exception as e:
        # print(f"Error fetching URL: {e}")
        return jsonify({"error": f"Failed to fetch or parse content: {str(e)}"}), 500

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400

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
            # print(f"Error during reranking: {e}")
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

    return jsonify({"results": top_results, "errors": errors})

if __name__ == '__main__':
    app.run(debug=True)
