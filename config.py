import os
from dotenv import load_dotenv

load_dotenv()

# ✅ Import from source modules using absolute paths
from RAG_data_collector_module.sources.files import DEFAULT_FILE_PATHS
from RAG_data_collector_module.sources.manual import MANUAL_INPUTS
from RAG_data_collector_module.sources.web import BASE_URL

# ✅ Export them clearly
FILE_PATHS = DEFAULT_FILE_PATHS
# Already a list from sources/files.py

# Manual Inputs
MANUAL_INPUTS = MANUAL_INPUTS
# Already defined in sources/manual.py

# Base URL (scraping)
BASE_URL = BASE_URL

# Common Config
VECTOR_DB_DIR = "vector_store"
STORAGE_DIR = "storage"
METADATA_FILE = "metadata.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
MIN_CHUNK_SIZE = 100

HTML_HEADERS_TO_SPLIT = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

REQUEST_TIMEOUT = 10

SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.md']
SUPPORTED_PDF_EXTENSIONS = ['.pdf']
SUPPORTED_FILE_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS + SUPPORTED_PDF_EXTENSIONS
