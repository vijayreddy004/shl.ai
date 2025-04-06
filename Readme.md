Great! Here's an updated `README.md` that now includes the **PDF ingestion and embedding workflow** using LangChain and Chroma. This section documents how PDFs are processed, split, embedded using HuggingFace models, and stored in a Chroma vector database.

---

# 🔍 SHL Assessment Recommendation System

This application recommends SHL assessments based on either a **natural language query** or **web page content**, and uses a hybrid of:
- 🧠 Pinecone Rerank API (for relevance ranking from product catalog)
- 📄 LangChain-based pipeline (for PDF ingestion, embedding & retrieval)

---

## 🚀 Features

- 🔗 **URL Input:** Extracts content from a web page and recommends relevant assessments.
- 💬 **Query Input:** Accepts natural language input for assessment recommendations.
- 📁 **PDF Ingestion:** Automatically loads and OCR-processes SHL PDFs to store them in a Chroma vector DB.
- 🧠 **Embeddings:** Uses HuggingFace multilingual MiniLM for document similarity.
- 📊 **Product Catalog Integration:** Matches PDFs to metadata from `product.csv`.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/shl-recommender.git
cd shl-recommender
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Make sure the following additional packages are installed for PDF and LangChain processing:
```bash
pip install langchain unstructured chromadb beautifulsoup4 sentence-transformers
```

---

## 📁 Directory Structure

```
shl-recommender/
├── app.py                     # Flask app for query and URL recommendation
├── embed_pdfs.py              # Script to load, embed, and persist PDFs to Chroma
├── .env                       # Environment variables
├── data/
│   └── product.csv            # Product catalog with metadata
├── downloads/                 # Folder with SHL PDF brochures
├── chroma_db/                 # Persisted Chroma vectorstore
├── templates/
│   └── input.html             # Frontend form
├── requirements.txt
└── README.md
```

---

## 📤 PDF Ingestion & Embedding (`embed_pdfs.py`)

This script processes SHL product brochures in the `downloads/` folder:

### 🔄 Steps:
1. **Load PDFs** with OCR using `UnstructuredPDFLoader`.
2. **Match filenames** with metadata in `product.csv`.
3. **Combine PDF text** into a single document per file.
4. **Split documents** into 2,000-character chunks.
5. **Generate embeddings** using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
6. **Store results** in a persistent Chroma vectorstore (`chroma_db/`).

### 🧾 Example Metadata Enrichment:
Each document gets metadata from `product.csv` like:
```json
{
  "name": "Logical Reasoning Assessment",
  "url": "https://shl.com/logical-test",
  "remote_testing": "Yes",
  "adaptive_irt": "No",
  "duration": "20 mins",
  "test_type": "Cognitive"
}
```

---

## 💻 Running the App

### 1. Ingest PDFs (One-time setup)
```bash
python embed_pdfs.py
```

### 2. Start the Flask App
```bash
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📡 API Endpoints

### `POST /recommend`
Query SHL assessments by natural language input.
```json
{
  "query": "We need a cognitive test for graduates"
}
```

### `POST /recommend_from_url`
Provide a webpage URL to extract job descriptions or requirements.
```json
{
  "url": "https://company.com/job/software-engineer"
}
```

---

## 📚 Technologies Used

- **Flask** – Web framework
- **Pinecone** – Re-ranking API
- **LangChain** – PDF processing and text splitting
- **Chroma** – Vector database
- **HuggingFace Transformers** – Embedding model
- **BeautifulSoup** – HTML parsing
- **OCR via Unstructured** – PDF text extraction

---

## 📝 Future Enhancements

- Use PDF-based retrieval to complement rerank results.
- Add UI for uploading and querying PDFs directly.
- Visualize result scores and metadata in a friendlier format.

---

Let me know if you want this `README.md` split into separate files (e.g., `README`, `docs/setup.md`, `docs/api.md`) or formatted for GitHub Pages!