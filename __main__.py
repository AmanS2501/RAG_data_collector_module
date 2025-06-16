import sys
import os
from pathlib import Path

def main():
    """Main function to run the RAG data collector"""
    print("="*60)
    print("RAG DATA COLLECTOR MODULE - MAIN EXECUTION")
    print("="*60)
    
    try:
        # Import the collector module
        from . import collector
        from .storage_utils import store_documents, DocumentStorage


        
        print("[INFO] Starting document collection process...")
        
        # Collect all documents
        documents = collector.collect_all_documents(chunk=True)
        
        if not documents:
            print("[WARNING] No documents were collected!")
            return
        
        print(f"[SUCCESS] Collected {len(documents)} documents total")
        
        # Store documents
        print("[INFO] Storing documents...")
        storage_success = store_documents(documents, storage_format="both")

        
        if storage_success:
            print("[SUCCESS] Documents stored successfully!")
        else:
            print("[ERROR] Failed to store some documents")
        
        # Show storage statistics
        print("\n" + "="*40)
        print("STORAGE STATISTICS")
        print("="*40)
        storage_obj = DocumentStorage()

        stats = storage_obj.get_storage_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("[INFO] Make sure you're running from the correct directory")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)
    
    print("\n[INFO] RAG Data Collector execution completed!")

def show_help():
    """Show help information"""
    help_text = """
RAG Data Collector Module
========================

Usage:
    python -m RAG_data_collector_module [options]

Options:
    --help, -h          Show this help message
    --collect-only      Only collect documents, don't store them
    --no-chunk          Don't chunk the documents
    --stats             Show storage statistics only
    --clean             Clean up old storage files

Examples:
    python -m RAG_data_collector_module
    python -m RAG_data_collector_module --stats
    python -m RAG_data_collector_module --collect-only
    """
    print(help_text)

def show_stats():
    """Show storage statistics"""
    try:
        from . import storage
        storage_obj = storage.DocumentStorage()
        stats = storage_obj.get_storage_stats()
        
        print("="*40)
        print("STORAGE STATISTICS")
        print("="*40)
        for key, value in stats.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"[ERROR] Could not get stats: {e}")

def collect_only():
    """Only collect documents without storing"""
    try:
        from . import collector
        print("[INFO] Collecting documents only (no storage)...")
        documents = collector.collect_all_documents(chunk=True)
        print(f"[SUCCESS] Collected {len(documents)} documents")
        return documents
    except Exception as e:
        print(f"[ERROR] Collection failed: {e}")
        return []

def clean_storage():
    """Clean up old storage files"""
    try:
        from . import storage
        storage_obj = storage.DocumentStorage()
        success = storage_obj.cleanup_old_files(days_old=30)
        if success:
            print("[SUCCESS] Storage cleanup completed")
        else:
            print("[ERROR] Storage cleanup failed")
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h']:
            show_help()
        elif arg == '--stats':
            show_stats()
        elif arg == '--collect-only':
            collect_only()
        elif arg == '--clean':
            clean_storage()
        elif arg == '--no-chunk':
            try:
                from . import collector
                from . import storage
                print("[INFO] Collecting documents without chunking...")
                documents = collector.collect_all_documents(chunk=False)
                if documents:
                    storage.store_documents(documents, storage_format="both")
                    print(f"[SUCCESS] Processed {len(documents)} documents without chunking")
            except Exception as e:
                print(f"[ERROR] Failed: {e}")
        else:
            print(f"[ERROR] Unknown argument: {arg}")
            show_help()
    else:
        # Run main function if no arguments
        main()