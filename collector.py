from urllib.parse import urljoin
from RAG_data_collector_module.sources import load_file_documents
from RAG_data_collector_module.sources import load_manual_documents
from RAG_data_collector_module.sources import crawl_website
from RAG_data_collector_module.utils import clean_text
from RAG_data_collector_module.utils import chunk_document
from RAG_data_collector_module.config import FILE_PATHS, MANUAL_INPUTS, BASE_URL, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from RAG_data_collector_module.utils import clean_web_content

from RAG_data_collector_module.sources import fetch_robots_txt, fetch_security_txt
from RAG_data_collector_module.storage_utils import DocumentStorage
from langchain_core.documents import Document
from langchain_core.documents import Document
from typing import List


def collect_file_documents() -> List[Document]:
    try:
        raw_docs = load_file_documents(FILE_PATHS)
        cleaned_docs = []
        for doc in raw_docs:
            cleaned_text = clean_text(doc.page_content)
            if cleaned_text:
                new_doc = Document(
                    page_content=cleaned_text,
                    metadata=doc.metadata.copy()
                )
                cleaned_docs.append(new_doc)
        return cleaned_docs
    except Exception as e:
        print(f"[ERROR] Failed to collect file documents: {e}")
        return []



def collect_manual_documents() -> List[Document]:
    try:
        raw_docs = load_manual_documents(MANUAL_INPUTS)
        cleaned_docs = []
        for doc in raw_docs:
            cleaned_text_content = clean_text(doc.page_content)
            if cleaned_text_content:
                new_doc = Document(
                    page_content=cleaned_text_content,
                    metadata=doc.metadata.copy()
                )
                cleaned_docs.append(new_doc)
        return cleaned_docs
    except Exception as e:
        print(f"[ERROR] Failed to collect manual documents: {e}")
        return []


def safe_crawl_website(url: str) -> List[Document]:
    try:
        # Force HTTPS if HTTP is used
        if url.startswith("http://"):
            url = url.replace("http://", "https://")

        # Inject updated URL into crawl_website
        return crawl_website(url)
    except Exception as e:
        print(f"[ERROR] Failed to crawl website: {e}")
        return []


def collect_web_documents() -> List[Document]:
    try:
        raw_docs = safe_crawl_website(BASE_URL)
        cleaned_docs = []

        for doc in raw_docs:
            print("[INFO] Cleaning web document...")
            cleaned_text_content = clean_web_content(doc.page_content)

            if cleaned_text_content:
                print("[INFO] Cleaned web document...")
                new_doc = Document(
                    page_content=cleaned_text_content,
                    metadata=doc.metadata.copy()
                )
                cleaned_docs.append(new_doc)

        return cleaned_docs

    except Exception as e:
        print(f"[ERROR] Failed to collect web documents: {e}")
        return []



def chunk_documents(documents: List[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[Document]:
    try:
        chunked_docs = []
        for doc in documents:
            chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
            chunked_docs.extend(chunks)
        return chunked_docs
    except Exception as e:
        print(f"[ERROR] Failed to chunk documents: {e}")
        return documents


def collect_all_documents(chunk: bool = True) -> List[Document]:
    try:
        print("[INFO] Starting document collection...")

        file_docs = collect_file_documents()
        print(f"[INFO] Collected {len(file_docs)} file documents")

        manual_docs = collect_manual_documents()
        print(f"[INFO] Collected {len(manual_docs)} manual documents")

        web_docs = collect_web_documents()
        print(f"[INFO] Collected {len(web_docs)} web documents")
        all_docs = file_docs + manual_docs + web_docs

        print(f"[INFO] Total documents before chunking: {len(all_docs)}")

        if chunk and all_docs:
            all_docs = chunk_documents(all_docs, )
            print(f"[INFO] Total documents after chunking: {len(all_docs)}")

        return all_docs
    except Exception as e:
        print(f"[ERROR] Failed to collect all documents: {e}")
        return []


# if __name__ == "__main__":
#     print("[INFO] Running collector directly...")
#     collected_docs = collect_all_documents(chunk=True)
#     print(f"[INFO] Collected total {len(collected_docs)} documents after cleaning and chunking.") This runs only when the script is executed directly