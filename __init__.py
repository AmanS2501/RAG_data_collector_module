# __init__.py for RAG data collector module

"""
RAG Data Collector Module
A package for collecting, processing, and storing documents for RAG systems.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

import sys
import os

print("Initializing RAG Data Collector Module...")

# Available modules list
available_modules = []

# Import from root level modules
try:
    from . import config
    available_modules.append("config")
    print("✓ Config module loaded")
except ImportError as e:
    print(f"⚠ Could not import config module: {e}")

try:
    from . import storage_utils
    available_modules.append("storage")
    print("✓ Storage module loaded")
except ImportError as e:
    print(f"⚠ Could not import storage module: {e}")

# Import from sources subfolder
try:
    from . import sources
    available_modules.append("sources")
    print("✓ Sources package loaded")
except ImportError as e:
    print(f"⚠ Could not import sources package: {e}")

# Import from utils subfolder  
try:
    from . import utils
    available_modules.append("utils")
    print("✓ Utils package loaded")
except ImportError as e:
    print(f"⚠ Could not import utils package: {e}")

# Import collector (this should work now)
try:
    from . import collector
    available_modules.append("collector")
    print("✓ Collector module loaded")
except ImportError as e:
    print(f"⚠ Could not import collector module: {e}")

# Simple test function
def hello():
    """Test function to verify package is working"""
    return f"RAG Data Collector Module v{__version__} is working!"

def list_modules():
    """List all available modules"""
    return available_modules

# Make key functions available at package level
def load_file_documents(*args, **kwargs):
    """Load documents from files"""
    try:
        return sources.load_file_documents(*args, **kwargs)
    except (NameError, AttributeError):
        print("Sources module not available")
        return []

def chunk_document(*args, **kwargs):
    """Chunk a document"""
    try:
        return utils.chunk_document(*args, **kwargs)
    except (NameError, AttributeError):
        print("Utils module not available")
        return []

def clean_text(*args, **kwargs):
    """Clean text"""
    try:
        return utils.clean_text(*args, **kwargs)
    except (NameError, AttributeError):
        print("Utils module not available")
        return ""

def collect_all_documents(*args, **kwargs):
    """Collect all documents"""
    try:
        return collector.collect_all_documents(*args, **kwargs)
    except (NameError, AttributeError):
        print("Collector module not available")
        return []

def store_documents(*args, **kwargs):
    """Store documents"""
    try:
        return storage.store_documents(*args, **kwargs)
    except (NameError, AttributeError):
        print("Storage module not available")
        return False

# Define what gets exported when using 'from module import *'
__all__ = [
    "hello", 
    "list_modules", 
    "load_file_documents", 
    "chunk_document", 
    "clean_text",
    "collect_all_documents",
    "store_documents",
    "config",
    "storage", 
    "sources",
    "utils",
    "collector"
] + available_modules

print(f"RAG Data Collector Module v{__version__} initialized successfully!")
print(f"Available modules: {', '.join(available_modules) if available_modules else 'None'}")

# If running directly, provide some basic info
if __name__ == "__main__":
    print("\n" + "="*50)
    print("RAG DATA COLLECTOR MODULE")
    print("="*50)
    print(f"Version: {__version__}")
    print(f"Available modules: {len(available_modules)}")
    
    if available_modules:
        print("\nLoaded modules:")
        for module in available_modules:
            print(f"  • {module}")
    else:
        print("\nNo modules were loaded successfully.")
        print("This might indicate issues with the module files.")
    
    print(f"\nTest function result: {hello()}")
    print("\nTo use this module properly, import it from Python:")
    print("  import RAG_data_collector_module")
    print("  RAG_data_collector_module.hello()")
    
    # Test available functions
    print("\nTesting available functions:")
    try:
        print(f"  collect_all_documents: {'✓' if 'collector' in available_modules else '✗'}")
        print(f"  load_file_documents: {'✓' if 'sources' in available_modules else '✗'}")
        print(f"  chunk_document: {'✓' if 'utils' in available_modules else '✗'}")
        print(f"  clean_text: {'✓' if 'utils' in available_modules else '✗'}")
        print(f"  store_documents: {'✓' if 'storage' in available_modules else '✗'}")
    except:
        pass