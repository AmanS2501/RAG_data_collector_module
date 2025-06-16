import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# File paths and directories
FILE_PATHS = [
    "documents/sample.pdf"
    # "documents/manual.txt",
    # "documents/guide.pdf"
]

VECTOR_DB_DIR = "vector_store"
STORAGE_DIR = "storage"
METADATA_FILE = "metadata.json"

# Web scraping configuration
BASE_URL = "https://www.lattice.site/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Manual input configuration
MANUAL_INPUTS = [
    {
        "title": "Company Policy",
        "content": "Our company follows strict data privacy guidelines and ensures customer information is protected at all times.",
        "category": "Policy"
    },
    {
        "title": "Product Features",
        "content": "Our AI-powered platform provides real-time analytics, automated reporting, and intelligent insights for business optimization.",
        "category": "Product"
    },
    {
        "title": "Support Guidelines",
        "content": "Customer support is available 24/7 through chat, email, and phone. Response time is guaranteed within 2 hours.",
        "category": "Support"
    }
]

# Embedding and vector store configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text processing configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
MIN_CHUNK_SIZE = 100

# HTML splitting configuration
HTML_HEADERS_TO_SPLIT = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

# Request configuration
REQUEST_TIMEOUT = 10

# File type support
SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.md']
SUPPORTED_PDF_EXTENSIONS = ['.pdf']
SUPPORTED_FILE_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS + SUPPORTED_PDF_EXTENSIONS