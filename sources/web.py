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
from RAG_data_collector_module.storage_utils import DocumentStorage
from RAG_data_collector_module.sources.files import fetch_file_content 

load_dotenv()

BASE_URL = input("Enter the base URL to crawl (e.g., https://example.com): ").strip()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

FILE_EXTENSIONS = (".pdf", ".docx", ".pptx", ".xlsx", ".zip", ".rar")

def is_file_url(url: str) -> bool:
    path = urlparse(url).path
    return any(path.lower().endswith(ext) for ext in FILE_EXTENSIONS)

def clean_text(text: str) -> str:
    return ' '.join(text.split())

def crawl_website(start_url: str, max_pages: int = 100) -> list[Document]:
    from time import sleep

    documents = []
    urls_to_visit = deque([start_url])
    visited_urls = set()
    base_domain = urlparse(start_url).netloc
    storage = DocumentStorage()

    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    print(f"[INFO] Starting crawl on domain: {base_domain}")

    while urls_to_visit and len(visited_urls) < max_pages:
        current_url = urls_to_visit.popleft().split("#")[0]

        if current_url in visited_urls:
            continue

        print(f"[INFO] Scraping: {current_url}")
        visited_urls.add(current_url)

        try:
            # 🆕 Handle file links (PDF, DOCX, etc.)
            if is_file_url(current_url):
                file_text = fetch_file_content(current_url)
                if file_text:
                    doc = Document(
                        page_content=clean_text(file_text),
                        metadata={"source": current_url, "type": "file"}
                    )
                    documents.append(doc)
                    storage.save_documents_as_json([doc], filename="partial_documents.json", append=True)
                    
                continue

            response = requests.get(current_url, headers=HEADERS, timeout=10)
            response.raise_for_status()

            html_chunks = html_splitter.split_text(response.text)

            # 🆕 Fallback if chunking fails (e.g., malformed HTML)
            if not html_chunks:
                html_chunks = [Document(page_content=clean_text(response.text), metadata={"source": current_url})]

            for chunk in html_chunks:
                chunk.page_content = clean_text(chunk.page_content)
                chunk.metadata["source"] = current_url

            if html_chunks:
                print(f"[INFO] Saving {len(html_chunks)} chunks from {current_url}")
                try:
                    storage.save_documents_as_json(
                        documents=html_chunks,
                        filename="partial_documents.json",
                        append=True
                    )
                except Exception as e:
                    print(f"[ERROR] Could not save partial data for {current_url}: {e}")

                documents.extend(html_chunks)

            # Discover internal links
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                link = a_tag['href']
                full_url = urljoin(current_url, link).split("#")[0]
                parsed_url = urlparse(full_url)

                if parsed_url.scheme == "http":
                    print(f"[INFO] Forcing HTTPS for: {full_url}")
                    parsed_url = parsed_url._replace(scheme="https")
                    full_url = parsed_url.geturl()

                if (parsed_url.scheme in ["http", "https"]) and \
                   (parsed_url.netloc == base_domain) and \
                   (full_url not in visited_urls):
                    urls_to_visit.append(full_url)

            sleep(0.5)

        except requests.RequestException as e:
            print(f"[ERROR] Request failed for {current_url}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to process {current_url}: {e}")

    return documents


# if __name__ == "__main__":
#     docs = crawl_website(BASE_URL)
#     print(f"[INFO] Crawled and chunked site into {len(docs)} documents.")   This runs only when the script is executed directly