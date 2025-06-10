import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import json
from datetime import datetime

MANUAL_INPUTS = [
    {
        "title": "Company Policy",
        "content": "Our company follows strict data privacy guidelines and ensures customer information is protected at all times.",
        "category": "Policy"
    },
    {
        "title": "Product Features",
        "content": "Our AI-powered platform provides real-time analytics, automated reporting, and intelligent insights for business optimization.",
        "category": "Product"
    },
    {
        "title": "Support Guidelines",
        "content": "Customer support is available 24/7 through chat, email, and phone. Response time is guaranteed within 2 hours.",
        "category": "Support"
    }
]

VECTOR_DB_DIR = "vector_store"

def clean_text(text: str) -> str:
    return ' '.join(text.split())

def validate_manual_input(input_data: dict) -> bool:
    required_fields = ['title', 'content']
    for field in required_fields:
        if field not in input_data or not input_data[field].strip():
            print(f"[ERROR] Missing or empty required field: {field}")
            return False
    return True

def process_manual_input(input_data: dict) -> str:
    try:
        if not validate_manual_input(input_data):
            return ""
        
        title = input_data.get('title', '').strip()
        content = input_data.get('content', '').strip()
        category = input_data.get('category', 'General').strip()
        
        print(f"[INFO] Processing manual input: {title}")
        
        # Format the content with metadata
        formatted_content = f"Title: {title}\nCategory: {category}\nContent: {content}"
        return clean_text(formatted_content)
        
    except Exception as e:
        print(f"[ERROR] Failed to process manual input '{input_data.get('title', 'Unknown')}': {e}")
        return ""

def load_documents(manual_inputs: List[dict]) -> List[Document]:
    documents = []
    for i, input_data in enumerate(manual_inputs):
        text = process_manual_input(input_data)
        if text:
            metadata = {
                "source": f"manual_input_{i+1}",
                "title": input_data.get('title', ''),
                "category": input_data.get('category', 'General'),
                "type": "manual",
                "timestamp": datetime.now().isoformat()
            }
            documents.append(Document(page_content=text, metadata=metadata))
    return documents

def store_in_vector_db(docs: List[Document], save_path: str):
    """Embed documents and store them in a FAISS vector DB."""
    print("[INFO] Embedding and saving documents to vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)
    print(f"[SUCCESS] Vector store saved to: {save_path}")

def add_manual_entry(title: str, content: str, category: str = "General") -> dict:
    """Helper function to create a manual entry."""
    return {
        "title": title,
        "content": content,
        "category": category
    }

def interactive_manual_input():
    print("[INFO] Starting interactive manual input mode...")
    entries = []
    
    while True:
        print("\n--- Add Manual Entry ---")
        title = input("Enter title (or 'quit' to finish): ").strip()
        
        if title.lower() == 'quit':
            break
            
        content = input("Enter content: ").strip()
        category = input("Enter category (optional): ").strip() or "General"
        
        if title and content:
            entries.append(add_manual_entry(title, content, category))
            print(f"[SUCCESS] Added entry: {title}")
        else:
            print("[ERROR] Title and content are required!")
    
    return entries

if __name__ == "__main__":
    use_interactive = input("Use interactive mode? (y/n): ").lower().startswith('y')
    
    if use_interactive:
        manual_data = interactive_manual_input()
    else:
        manual_data = MANUAL_INPUTS
    
    docs = load_documents(manual_data)
    print(f"[INFO] Loaded {len(docs)} documents.")
    
    if docs:
        store_in_vector_db(docs, VECTOR_DB_DIR)