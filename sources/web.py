import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from collections import deque
from urllib.parse import urljoin, urlparse
from langchain_text_splitters import HTMLHeaderTextSplitter

load_dotenv()

# url
BASE_URL = "https://www.lattice.site/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

VECTOR_DB_DIR = "vector_store"

def clean_text(text: str) -> str:
    return ' '.join(text.split())

def crawl_website(start_url: str) -> list[Document]:
    documents = []
    urls_to_visit = deque([start_url])
    visited_urls = set()
    base_domain = urlparse(start_url).netloc

    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

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
            
            html_chunks = html_splitter.split_text(response.text)
            
            for chunk in html_chunks:
                chunk.page_content = clean_text(chunk.page_content)
                chunk.metadata["source"] = current_url
            
            documents.extend(html_chunks)

            soup = BeautifulSoup(response.text, "html.parser")
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
        
    print(f"[INFO] Embedding and saving {len(docs)} document chunks to vector DB...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)

    print(f"[SUCCESS] Vector store saved to: {save_path}")

if __name__ == "__main__":
    docs = crawl_website(BASE_URL)
    print(f"[INFO] Crawled and chunked site into {len(docs)} documents.")

    if docs:
        store_in_vector_db(docs, VECTOR_DB_DIR)