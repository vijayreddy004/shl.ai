
# 🔍 SHL Assessment Recommender System

This is an AI-powered recommendation system built with **Flask**, **Pinecone**, and **BeautifulSoup**. It leverages **vector reranking** and **AI agents** to recommend the most relevant SHL assessments based on a user's natural language query or job description URL.

---

## 🚀 Overview

This app helps users find the best-fit **SHL assessments** by analyzing a query or web page content and reranking preloaded assessment descriptions using **Pinecone’s RAG-ready rerank API** powered by **LLM-based vector embeddings**.

Key highlights:
- 🧠 Uses AI agent (reranker) to rerank documents based on relevance
- 🌐 Accepts free-form text **or** a URL to a job description
- 🧾 Outputs a list of the **top 10** matching SHL products
- 📊 Uses a preloaded CSV of SHL product metadata

---

## 🧬 How It Works

1. **Data Preparation**  
   SHL assessment data is stored in a CSV file (`data/product.csv`) and converted to dictionary records. Descriptions are truncated for efficiency.

2. **User Input**  
   Users provide either:
   - A **free-text** description of a role
   - A **URL** pointing to a job description

3. **AI Agent: Reranker**  
   - The system batches candidate documents and calls **Pinecone's rerank API** (`bge-reranker-v2-m3`)
   - The API uses **vector embeddings** and LLM-based comparison to score relevance

4. **Results**  
   The top 10 ranked assessments are returned, along with metadata like:
   - Name
   - Test types
   - Job levels
   - Languages
   - Remote testing support

---

## 🛠️ Tech Stack

| Component       | Description                                              |
|----------------|----------------------------------------------------------|
| **Flask**       | Lightweight Python web framework                         |
| **Pinecone**    | Vector database and reranking API                        |
| **bge-reranker-v2-m3** | Open-source LLM reranker model used via Pinecone         |
| **BeautifulSoup** | Parses HTML from job description URLs                   |
| **dotenv**      | Loads environment variables like Pinecone API key        |
| **pandas**      | Handles CSV data of SHL assessments                      |

---

## 📁 File Structure

```
.
├── app.py                  # Main Flask app
├── data/
│   └── product.csv         # SHL assessments metadata
├── templates/
│   └── input.html          # Input form for user queries
├── .env                    # Store your PINECONE_API_KEY here
└── README.md               # You are here
```

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/shl-recommender.git
cd shl-recommender
pip install -r requirements.txt
```

Add a `.env` file:

```
PINECONE_API_KEY=your-api-key-here
```

Run the app:

```bash
python app.py
```

---

## 🔗 API Endpoints

- `/` – Web UI form
- `/recommend` – POST endpoint for JSON-based recommendations
- `/health` – Health check endpoint

**Example Request:**
```json
POST /recommend
{
  "query": "https://example.com/job-description"
}
```

---

## 🤖 About the AI Agent

This system uses Pinecone's rerank API as an **AI agent**. Instead of matching documents via simple similarity, this agent **understands** both the query and the documents, evaluating them with **contextual awareness**. It's ideal for complex search tasks like assessment mapping and skills alignment.

---

## 📌 Future Improvements

- Integrate full-text vector search for hybrid search + rerank
- Add support for uploading PDFs of job descriptions
- Save recommendation history in a database
- Add frontend enhancements (filters, sorting, etc.)

---

## 🧠 Learn More

- [Pinecone Rerank API](https://docs.pinecone.io/docs/rerank)
- [bge-reranker-v2-m3 Model](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [SHL Assessments](https://www.shl.com/)

---
