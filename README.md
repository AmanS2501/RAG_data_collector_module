# 🧠 RAG Data Collector Module

A modular, pluggable data ingestion and processing pipeline for **Retrieval-Augmented Generation (RAG)** systems.  
Collects, cleans, chunks, and vectorizes unstructured content from PDFs, text files, manual entries, and web sources.

---

## 📦 Features

- 📂 Ingests content from:
  - PDFs, `.txt`, `.md` files
  - Hardcoded/manual knowledge entries
  - Websites using a configurable base URL

- 🧹 Cleans & normalizes text using regex, HTML tag removal, etc.

- 🔪 Smart chunking by:
  - Character size
  - Sentence
  - Paragraph

- 🧠 Embeds and stores documents in a **FAISS vector DB** using **HuggingFace Sentence Transformers**

- ⚙️ Fully configurable via `config.py`

- 💻 CLI support for scripted runs

---

## 🧰 Tech Stack

| Layer           | Tool/Library                       |
|----------------|------------------------------------|
| Embeddings      | `sentence-transformers` (MiniLM)   |
| Vector DB       | `FAISS`                            |
| Parsing         | `PyPDF2`, `requests`, `BeautifulSoup` |
| Framework       | `LangChain`, `Python`              |

---

## 📁 Project Structure

RAG_data_collector_module/

├── main.py # CLI runner

├── init.py # Package bootstrap

├── collector.py # Document collection logic

├── config.py # Central config

├── storage.py # JSON & FAISS storage

├── sources/ # Ingestion modules

│ ├── files.py # PDF, text file input

│ ├── robots.py 

│ ├── manual.py # Hardcoded content

│ └── web.py # Web scraping

├── utils/ # Preprocessing

│ ├── cleaner.py # Cleaning rules

│ └── chunker.py # Chunking strategies

└── vector_store/ # Output vector DB



---

## 🚀 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/AmanS2501/RAG_data_collector_module.git
cd RAG_data_collector_module (You have run this as a package from root folder)

conda create -n chatbotenv python=3.10
conda activate chatbotenv
pip install -r requirements.txt
cd ..

python -m RAG_data_collector_module

--collect-only   # Skip vector storing
--no-chunk       # Skip chunking
--clean          # Remove old files and vectors
--stats          # Print collected metadata
