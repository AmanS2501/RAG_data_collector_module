import re
from typing import Optional
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    # Clean and normalize text by removing extra whitespace.
    if not text:
        return ""
    return ' '.join(text.split())

def remove_html_tags(html_content: str) -> str:
    # Remove HTML tags from content using BeautifulSoup.
    if not html_content:
        return ""
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"[ERROR] Failed to remove HTML tags: {e}")
        return html_content

def normalize_whitespace(text: str) -> str:
    # Normalize whitespace in text.
    if not text:
        return ""
    # Replace multiple whitespace characters with single space
    normalized = re.sub(r'\s+', ' ', text)
    return normalized.strip()

def remove_special_characters(text: str, keep_basic_punctuation: bool = True) -> str:
    # Remove special characters from text.
    if not text:
        return ""
    
    if keep_basic_punctuation:
        # Keep basic punctuation: . , ! ? ; : ( ) [ ] { } " ' /
        cleaned = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\'/]', ' ', text)
    else:
        # Keep only alphanumeric characters and spaces
        cleaned = re.sub(r'[^\w\s]', ' ', text)
    
    return normalize_whitespace(cleaned)

def clean_pdf_text(text: str) -> str:
    # Clean text extracted from PDF files.
    if not text:
        return ""
    
    # Remove common PDF artifacts
    text = re.sub(r'\x0c', ' ', text)  # Remove form feed characters
    text = re.sub(r'\n+', ' ', text)   # Replace multiple newlines with space
    text = re.sub(r'\t+', ' ', text)   # Replace tabs with space
    
    # Remove page numbers (assuming they're standalone numbers)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Clean and normalize
    return clean_text(text)

def clean_web_content(html_content: str) -> str:
    # Clean content scraped from web pages.
    if not html_content:
        return ""
    
    # First remove HTML tags
    text = remove_html_tags(html_content)
    
    # Remove common web artifacts
    text = re.sub(r'Cookie Policy|Privacy Policy|Terms of Service', '', text, flags=re.IGNORECASE)
    text = re.sub(r'©\s*\d{4}.*?All Rights Reserved', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Subscribe|Newsletter|Follow us', '', text, flags=re.IGNORECASE)
    
    # Clean and normalize
    return clean_text(text)

def remove_urls(text: str) -> str:
    # Remove URLs from text.
    if not text:
        return ""
    
    # Remove URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    
    return normalize_whitespace(text)

def remove_emails(text: str) -> str:
    # Remove email addresses from text.
    if not text:
        return ""
    
    # Remove email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '', text)
    
    return normalize_whitespace(text)

def clean_manual_input(title: str, content: str, category: str = "General") -> str:
    # Clean and format manual input data.
    title = clean_text(title.strip()) if title else ""
    content = clean_text(content.strip()) if content else ""
    category = clean_text(category.strip()) if category else "General"
    
    if not title or not content:
        return ""
    
    formatted_content = f"Title: {title}\nCategory: {category}\nContent: {content}"
    return formatted_content