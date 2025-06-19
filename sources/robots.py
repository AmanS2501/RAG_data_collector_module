import requests
from urllib.parse import urljoin
from dotenv import load_dotenv
load_dotenv()

ROBOTS_PATH = "/robots.txt"
WELLKNOWN_PATH = "/.well-known/security.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (RAGDataCollectorBot/1.0)"
}

def fetch_robots_txt(base_url: str) -> str:
    """Fetch robots.txt content from the website"""
    robots_url = urljoin(base_url, ROBOTS_PATH)
    try:
        response = requests.get(robots_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        print(f"[SUCCESS] Fetched robots.txt from {robots_url}")
        return response.text
    except Exception as e:
        print(f"[ERROR] Failed to fetch robots.txt: {e}")
        return ""

def fetch_security_txt(base_url: str) -> str:
    """Fetch .well-known/security.txt content from the website"""
    security_url = urljoin(base_url, WELLKNOWN_PATH)
    try:
        response = requests.get(security_url, headers=HEADERS, timeout=10)
        if response.status_code == 404:
            print(f"[INFO] .well-known/security.txt not found (404)")
            return ""
        response.raise_for_status()
        print(f"[SUCCESS] Fetched security.txt from {security_url}")
        return response.text
    except Exception as e:
        print(f"[ERROR] Failed to fetch .well-known/security.txt: {e}")
        return ""

