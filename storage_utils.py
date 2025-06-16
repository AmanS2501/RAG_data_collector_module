import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Configuration
STORAGE_DIR = "storage"
VECTOR_DB_DIR = "vector_store"
METADATA_FILE = "metadata.json"

class DocumentStorage:
    # Handles storage operations for documents and metadata.
    
    def __init__(self, storage_dir: str = STORAGE_DIR):
        self.storage_dir = Path(storage_dir)
        self.vector_db_dir = Path(VECTOR_DB_DIR)
        self.metadata_file = self.storage_dir / METADATA_FILE
        self._ensure_directories()
    
    def _ensure_directories(self):
        # Create storage directories if they don't exist.
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Storage directories ensured: {self.storage_dir}, {self.vector_db_dir}")
    
    def save_documents_as_json(self, documents: List[Document], filename: str = "documents.json") -> bool:
        # Save documents as JSON file.
        try:
            file_path = self.storage_dir / filename
            
            # Convert documents to serializable format
            doc_data = []
            for doc in documents:
                doc_dict = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "saved_at": datetime.now().isoformat()
                }
                doc_data.append(doc_dict)
            
            # Save to JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            
            print(f"[SUCCESS] Saved {len(documents)} documents to: {file_path}")
            
            # Update metadata
            self._update_metadata({
                "type": "json_save",
                "filename": filename,
                "document_count": len(documents),
                "saved_at": datetime.now().isoformat(),
                "file_path": str(file_path)
            })
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save documents as JSON: {e}")
            return False
    
    def load_documents_from_json(self, filename: str = "documents.json") -> List[Document]:
        # Load documents from JSON file.
        try:
            file_path = self.storage_dir / filename
            
            if not file_path.exists():
                print(f"[ERROR] JSON file not found: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Convert back to Document objects
            documents = []
            for doc_dict in doc_data:
                doc = Document(
                    page_content=doc_dict.get("page_content", ""),
                    metadata=doc_dict.get("metadata", {})
                )
                documents.append(doc)
            
            print(f"[SUCCESS] Loaded {len(documents)} documents from: {file_path}")
            return documents
            
        except Exception as e:
            print(f"[ERROR] Failed to load documents from JSON: {e}")
            return []
    
    def save_raw_text(self, text: str, filename: str, source_info: Dict[str, Any] = None) -> bool:
        # Save raw text content to file.
        try:
            file_path = self.storage_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"[SUCCESS] Saved raw text to: {file_path}")
            
            # Update metadata
            metadata = {
                "type": "raw_text",
                "filename": filename,
                "text_length": len(text),
                "saved_at": datetime.now().isoformat(),
                "file_path": str(file_path)
            }
            
            if source_info:
                metadata.update(source_info)
            
            self._update_metadata(metadata)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save raw text: {e}")
            return False
    
    def store_in_vector_db(self, documents: List[Document], save_path: str = None) -> bool:
        # Store documents in FAISS vector database.
        try:
            if not documents:
                print("[WARNING] No documents to store. Skipping vector DB creation.")
                return False
            
            save_path = save_path or str(self.vector_db_dir)
            
            print(f"[INFO] Embedding and saving {len(documents)} documents to vector DB...")
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(save_path)
            
            print(f"[SUCCESS] Vector store saved to: {save_path}")
            
            # Update metadata
            self._update_metadata({
                "type": "vector_db",
                "document_count": len(documents),
                "saved_at": datetime.now().isoformat(),
                "vector_db_path": save_path,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            })
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to store in vector DB: {e}")
            return False
    
    def load_vector_db(self, load_path: str = None) -> Optional[FAISS]:
        # Load FAISS vector database.
        try:
            load_path = load_path or str(self.vector_db_dir)
            
            if not Path(load_path).exists():
                print(f"[ERROR] Vector DB path not found: {load_path}")
                return None
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
            
            print(f"[SUCCESS] Loaded vector DB from: {load_path}")
            return vectorstore
            
        except Exception as e:
            print(f"[ERROR] Failed to load vector DB: {e}")
            return None
    
    def save_processing_log(self, log_data: Dict[str, Any], log_filename: str = "processing_log.json") -> bool:
        # Save processing log with URLs/files processed.
        try:
            log_path = self.storage_dir / log_filename
            
            # Load existing log if it exists
            existing_logs = []
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
            
            # Add new log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                **log_data
            }
            existing_logs.append(log_entry)
            
            # Save updated log
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, indent=2, ensure_ascii=False)
            
            print(f"[SUCCESS] Processing log saved to: {log_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save processing log: {e}")
            return False
    
    def _update_metadata(self, metadata: Dict[str, Any]):
        # Update metadata file with operation information.
        try:
            # Load existing metadata
            existing_metadata = []
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            
            # Add new metadata entry
            existing_metadata.append(metadata)
            
            # Save updated metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"[ERROR] Failed to update metadata: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        # Get storage statistics.
        try:
            stats = {
                "storage_directory": str(self.storage_dir),
                "vector_db_directory": str(self.vector_db_dir),
                "files_in_storage": 0,
                "total_storage_size": 0,
                "vector_db_exists": self.vector_db_dir.exists(),
                "metadata_entries": 0,
                "last_updated": None
            }
            
            # Count files and calculate size
            if self.storage_dir.exists():
                for file_path in self.storage_dir.rglob('*'):
                    if file_path.is_file():
                        stats["files_in_storage"] += 1
                        stats["total_storage_size"] += file_path.stat().st_size
            
            # Get metadata info
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    stats["metadata_entries"] = len(metadata)
                    if metadata:
                        stats["last_updated"] = metadata[-1].get("saved_at")
            
            return stats
            
        except Exception as e:
            print(f"[ERROR] Failed to get storage stats: {e}")
            return {}
    
    def cleanup_old_files(self, days_old: int = 30) -> bool:
        # Clean up files older than specified days.
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            removed_count = 0
            
            for file_path in self.storage_dir.rglob('*'):
                if file_path.is_file():
                    file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_modified < cutoff_date:
                        file_path.unlink()
                        removed_count += 1
            
            print(f"[SUCCESS] Cleaned up {removed_count} old files")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to cleanup old files: {e}")
            return False

def store_documents(documents: List[Document], storage_format: str = "both") -> bool:
    # Convenience function to store documents.
    storage = DocumentStorage()
    
    success = True
    
    if storage_format in ["json", "both"]:
        success &= storage.save_documents_as_json(documents)
    
    if storage_format in ["vector", "both"]:
        success &= storage.store_in_vector_db(documents)
    
    return success

def load_stored_documents(source: str = "json") -> List[Document]:
    # Convenience function to load stored documents.
    storage = DocumentStorage()
    
    if source == "json":
        return storage.load_documents_from_json()
    else:
        print("[ERROR] Only JSON loading is supported for documents")
        return []

if __name__ == "__main__":
    # Example usage
    storage = DocumentStorage()
    
    # Print storage statistics
    stats = storage.get_storage_stats()
    print(f"[INFO] Storage Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test with sample documents
    sample_docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "test", "type": "sample"}
        )
    ]
    
    # Test storage operations
    print("\n[INFO] Testing storage operations...")
    storage.save_documents_as_json(sample_docs, "test_documents.json")
    loaded_docs = storage.load_documents_from_json("test_documents.json")
    print(f"[INFO] Loaded {len(loaded_docs)} documents from JSON")