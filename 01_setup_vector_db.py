from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

# --- Configuration ---
PDF_PATH = "data/CRAG_paper.pdf"
DB_PATH = "./chroma_db"

def build_vector_db():
    """Builds and persists a Chroma vector database from a PDF document."""
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF not found at {PDF_PATH}. Please create a 'data' folder and add your PDF.")
        return

    print("Step 1: Loading document...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Step 2: Splitting documents into chunks...")
    # Smaller chunks for better knowledge strip granularity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    print("Step 3: Creating local embeddings and persisting the vector store...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=DB_PATH
    )
    print(f"\n✓ Vector DB created successfully with {len(docs)} chunks.")

if __name__ == "__main__":
    build_vector_db()
