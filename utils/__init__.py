# utils/__init__.py

"""
Utils package for RAG Data Collector Module
Contains utility modules for chunking and cleaning text data
"""

__version__ = "0.1.0"

# Import modules
try:
    from . import chunker
    from . import cleaner
    
    # Make key functions available at package level
    from .chunker import (
        chunk_document,
        chunk_text_by_size,
        chunk_by_sentences,
        chunk_by_words,
        chunk_by_paragraphs,
        smart_chunking
    )
    from .cleaner import (
        clean_text,
        remove_html_tags,
        normalize_whitespace,
        remove_special_characters,
        clean_pdf_text,
        clean_web_content,
        remove_urls,
        remove_emails,
        clean_manual_input
    )
    
    __all__ = [
        "chunker",
        "cleaner",
        "chunk_document",
        "chunk_text_by_size", 
        "chunk_by_sentences",
        "chunk_by_words",
        "chunk_by_paragraphs",
        "smart_chunking",
        "clean_text",
        "remove_html_tags",
        "normalize_whitespace",
        "remove_special_characters",
        "clean_pdf_text",
        "clean_web_content",
        "remove_urls",
        "remove_emails",
        "clean_manual_input"
    ]
    
    print("✓ Utils package loaded successfully")
    
except ImportError as e:
    print(f"⚠ Error loading utils package: {e}")
    __all__ = []