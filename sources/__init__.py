__version__ = "0.1.0"

# Import modules
try:
    from . import files
    from . import manual
    from . import web
    
    # Make key functions available at package level
    from .files import load_documents as load_file_documents, fetch_file_content
    from .manual import load_documents as load_manual_documents, add_manual_entry
    from .web import crawl_website, BASE_URL, HEADERS
    from .manual import MANUAL_INPUTS
    from .files import DEFAULT_FILE_PATHS
    from . import robots  # ✅ NEW
    from .robots import fetch_robots_txt, fetch_security_txt# ✅ NEW



    __all__ = [
        "files",
        "manual", 
        "web",
        "load_file_documents",
        "fetch_file_content",
        "load_manual_documents", 
        "add_manual_entry",
        "crawl_website",
        "robots",
        "fetch_robots_txt",
        "fetch_security_txt",
        "robots"
    ]
    
    print("✓ Sources package loaded successfully")
    
except ImportError as e:
    print(f"⚠ Error loading sources package: {e}")
    __all__ = []