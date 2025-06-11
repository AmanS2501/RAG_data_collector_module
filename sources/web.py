import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from collections import deque
from urllib.parse import urljoin, urlparse

load_dotenv()

#  URL for the crawl
BASE_URL = "https://www.viit.ac.in/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

VECTOR_DB_DIR = "vector_store"

def clean_text(text: str) -> str:
    # Clean and normalize
    return ' '.join(text.split())

def crawl_website(start_url: str) -> list[Document]:
    # Crawl a website 
    documents = []
    urls_to_visit = deque([start_url])
    visited_urls = set()
    base_domain = urlparse(start_url).netloc

    print(f"[INFO] Starting crawl on domain: {base_domain}")

    while urls_to_visit:
        current_url = urls_to_visit.popleft()

        if current_url in visited_urls:
            continue

        print(f"[INFO] Scraping: {current_url}")
        visited_urls.add(current_url)

        try:
            response = requests.get(current_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract and clean text for the document
            for element in soup(["script", "style", "noscript", "header", "footer"]):
                element.extract()
            text = clean_text(soup.get_text())

            if text:
                documents.append(Document(page_content=text, metadata={"source": current_url}))

            # Find all internal links on the page and add them to the queue
            for a_tag in soup.find_all("a", href=True):
                link = a_tag['href']
                full_url = urljoin(current_url, link)
                parsed_url = urlparse(full_url)

                if (parsed_url.scheme in ["http", "https"]) and (parsed_url.netloc == base_domain) and (full_url not in visited_urls):
                    urls_to_visit.append(full_url)

        except requests.RequestException as e:
            print(f"[ERROR] Request failed for {current_url}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to process {current_url}: {e}")

    return documents

def store_in_vector_db(docs: list[Document], save_path: str):
    if not docs:
        print("[WARNING] No documents to store. Skipping vector DB creation.")
        return
        
    print("[INFO] Embedding and saving documents to vector DB...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)

    print(f"[SUCCESS] Vector store saved to: {save_path}")

if __name__ == "__main__":
    docs = crawl_website(BASE_URL)
    print(f"[INFO] Crawled and loaded {len(docs)} documents.")

    if docs:
        store_in_vector_db(docs, VECTOR_DB_DIR)