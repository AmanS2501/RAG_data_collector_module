import os
from typing import List
from langchain_core.documents import Document

# Use relative imports for package structure
try:
    # Try relative imports first (when imported as part of package)
    from .sources.files import load_documents as load_file_documents
    from .sources.manual import load_documents as load_manual_documents
    from .sources.web import crawl_website
    from .utils.cleaner import clean_text
    from .utils.chunker import chunk_document
    from .config import FILE_PATHS, MANUAL_INPUTS, BASE_URL, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
except ImportError:
    # Fallback for direct execution or when package structure is not recognized
    try:
        from sources.files import load_documents as load_file_documents
        from sources.manual import load_documents as load_manual_documents
        from sources.web import crawl_website
        from utils.cleaner import clean_text
        from utils.chunker import chunk_document
        from config import FILE_PATHS, MANUAL_INPUTS, BASE_URL, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
    except ImportError as e:
        print(f"[ERROR] Could not import required modules: {e}")
        print("[INFO] Make sure you're running from the correct directory and all files are present")
        raise

def collect_file_documents() -> List[Document]:
    """
    Load documents from local files specified in FILE_PATHS,
    clean their content, and return a list of Documents.
    """
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
    """
    Load documents from manual inputs specified in MANUAL_INPUTS,
    clean their content, and return a list of Documents.
    """
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

def collect_web_documents() -> List[Document]:
    """
    Crawl the website starting from BASE_URL to collect documents,
    clean their content, and return a list of Documents.
    """
    try:
        raw_docs = crawl_website(BASE_URL)
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
        print(f"[ERROR] Failed to collect web documents: {e}")
        return []

def chunk_documents(documents: List[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[Document]:
    """
    Chunk all documents using chunk_document function with given size and overlap.
    Returns a list of chunked Documents.
    """
    try:
        chunked_docs = []
        for doc in documents:
            chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
            chunked_docs.extend(chunks)
        return chunked_docs
    except Exception as e:
        print(f"[ERROR] Failed to chunk documents: {e}")
        return documents  # Return original documents if chunking fails

def collect_all_documents(chunk: bool = True) -> List[Document]:
    """
    Collect documents from files, manual inputs, and web crawling.
    Clean and optionally chunk them.
    Return the combined list of Documents ready for storage or processing.
    """
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
            all_docs = chunk_documents(all_docs)
            print(f"[INFO] Total documents after chunking: {len(all_docs)}")

        return all_docs
    except Exception as e:
        print(f"[ERROR] Failed to collect all documents: {e}")
        return []

if __name__ == "__main__":
    print("[INFO] Running collector directly...")
    collected_docs = collect_all_documents(chunk=True)
    print(f"[INFO] Collected total {len(collected_docs)} documents after cleaning and chunking.")