__version__ = "0.1.0"
__author__ = "Aman Shaikh"

import sys
import os

print("Initializing RAG Data Collector Module...")

# We'll collect available module names here
available_modules = []

# Delay imports until after initialization
def initialize_modules():
    global config, storage, sources, utils, collector

    try:
        import RAG_data_collector_module.config as config
        available_modules.append("config")
        print("✓ Config module loaded")
    except ImportError as e:
        print(f"⚠ Could not import config module: {e}")

    try:
        import RAG_data_collector_module.storage_utils as storage
        available_modules.append("storage")
        print("✓ Storage module loaded")
    except ImportError as e:
        print(f"⚠ Could not import storage module: {e}")

    try:
        import RAG_data_collector_module.sources as sources
        available_modules.append("sources")
        print("✓ Sources package loaded")
    except ImportError as e:
        print(f"⚠ Could not import sources package: {e}")

    try:
        import RAG_data_collector_module.utils as utils
        available_modules.append("utils")
        print("✓ Utils package loaded")
    except ImportError as e:
        print(f"⚠ Could not import utils package: {e}")

    try:
        import RAG_data_collector_module.collector as collector
        available_modules.append("collector")
        print("✓ Collector module loaded")
    except ImportError as e:
        print(f"⚠ Could not import collector module: {e}")

initialize_modules()

# Test function
def hello():
    return f"RAG Data Collector Module v{__version__} is working!"

def list_modules():
    return available_modules

# Safe proxy functions
# def load_file_documents(*args, **kwargs):
#     try:
#         return sources.load_file_documents(*args, **kwargs)
#     except:
#         print("Sources module not available")
#         return []

# def chunk_document(*args, **kwargs):
#     try:
#         return utils.chunk_document(*args, **kwargs)
#     except:
#         print("Utils module not available")
#         return []

# def clean_text(*args, **kwargs):
#     try:
#         return utils.clean_text(*args, **kwargs)
#     except:
#         print("Utils module not available")
#         return ""

# def collect_all_documents(*args, **kwargs):
#     try:
#         return collector.collect_all_documents(*args, **kwargs)
#     except:
#         print("Collector module not available")
#         return []

# def store_documents(*args, **kwargs):
#     try:
#         return storage.store_documents(*args, **kwargs)
#     except:
#         print("Storage module not available")
#         return False

__all__ = [
    "hello", "list_modules", "load_file_documents", "chunk_document",
    "clean_text", "collect_all_documents", "store_documents",
    "config", "storage", "sources", "utils", "collector"
] + available_modules

print(f"RAG Data Collector Module v{__version__} initialized successfully!")
print(f"Available modules: {', '.join(available_modules) if available_modules else 'None'}")

if __name__ == "__main__":
    print("="*50)
    print("RAG DATA COLLECTOR MODULE")
    print("="*50)
    print(f"Version: {__version__}")
    print(f"Available modules: {len(available_modules)}")

    print(f"\nTest function result: {hello()}") #runs only if executed directly
