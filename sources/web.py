import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# URLs to scrape
URLS = [
    "https://cynayd.com/",
    "https://cynayd.com/service-web",
    "https://cynayd.com/why-us"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

VECTOR_DB_DIR = "vector_store"

def clean_text(text: str) -> str:
    # Clean and normalize text extracted from HTML.
    return ' '.join(text.split())

def fetch_page_text(url: str) -> str:
    # Fetch and parse HTML from a given URL, returning clean visible text.
    try:
        print(f"[INFO] Fetching: {url}")
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.extract()

        return clean_text(soup.get_text())
    except requests.RequestException as e:
        print(f"[ERROR] Request failed for {url}: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to process {url}: {e}")
    return ""

def load_documents(urls: list[str]) -> list[Document]:
    # Load content from a list of URLs and return as Document objects.
    documents = []
    for url in urls:
        text = fetch_page_text(url)
        if text:
            documents.append(Document(page_content=text, metadata={"source": url}))
    return documents

def store_in_vector_db(docs: list[Document], save_path: str):
    # Embed documents and store them in a FAISS vector DB.
    print("[INFO] Embedding and saving documents to vector DB...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)

    print(f"[SUCCESS] Vector store saved to: {save_path}")

if __name__ == "__main__":
    docs = load_documents(URLS)
    print(f"[INFO] Loaded {len(docs)} documents.")

    if docs:
        store_in_vector_db(docs, VECTOR_DB_DIR)