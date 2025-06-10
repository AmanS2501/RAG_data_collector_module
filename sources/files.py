import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import PyPDF2
from typing import List

FILE_PATHS = [
    "documents/sample.pdf"
    # "documents/manual.txt",
    # "documents/guide.pdf"
]

VECTOR_DB_DIR = "vector_store"

def clean_text(text: str) -> str:
    return ' '.join(text.split())

def read_pdf_file(file_path: str) -> str:
    try:
        print(f"[INFO] Reading PDF: {file_path}")
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return clean_text(text)
    except Exception as e:
        print(f"[ERROR] Failed to read PDF {file_path}: {e}")
        return ""

def read_text_file(file_path: str) -> str:
    try:
        print(f"[INFO] Reading text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return clean_text(text)
    except Exception as e:
        print(f"[ERROR] Failed to read text file {file_path}: {e}")
        return ""

def fetch_file_content(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return ""
    
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return read_pdf_file(file_path)
    elif file_extension in ['.txt', '.md']:
        return read_text_file(file_path)
    else:
        print(f"[ERROR] Unsupported file type: {file_extension}")
        return ""

def load_documents(file_paths: List[str]) -> List[Document]:
    documents = []
    for file_path in file_paths:
        text = fetch_file_content(file_path)
        if text:
            documents.append(Document(
                page_content=text, 
                metadata={"source": file_path, "type": Path(file_path).suffix}
            ))
    return documents

def store_in_vector_db(docs: List[Document], save_path: str):
    print("[INFO] Embedding and saving documents to vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)
    print(f"[SUCCESS] Vector store saved to: {save_path}")

if __name__ == "__main__":
    docs = load_documents(FILE_PATHS)
    print(f"[INFO] Loaded {len(docs)} documents.")
    
    if docs:
        store_in_vector_db(docs, VECTOR_DB_DIR)