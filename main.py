from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document
import pandas as pd
import os
combined_df = pd.read_csv('data/product.csv')
documents = []
for filename in os.listdir('downloads'):
    if filename.endswith('.pdf'):
        loader = UnstructuredPDFLoader(os.path.join('downloads', filename), strategy="ocr_only")
        pages = loader.load()
        full_text = " ".join([page.page_content for page in pages])
        pdf_name = os.path.splitext(filename)[0].strip()
        meta_matches = combined_df[combined_df['name'].apply(lambda x: pdf_name.startswith(x.strip()))]
        if not meta_matches.empty:
            meta_row = meta_matches.iloc[0]
            metadata = {
                'name': meta_row['name'],
                'url': meta_row['url'],
                'remote_testing': meta_row['remote_testing'],
                'adaptive_irt': meta_row['adaptive_irt'],
                'duration': meta_row['assessment_length'],
                'test_type': meta_row['test_types']
            }
            print(f"Loaded {filename} with metadata: {metadata}")
            doc = Document(page_content=full_text, metadata=metadata)
            documents.append(doc)
if documents:
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    filtered_texts = filter_complex_metadata(texts)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma.from_documents(filtered_texts, embeddings, persist_directory="chroma_db")
    vectorstore.persist()