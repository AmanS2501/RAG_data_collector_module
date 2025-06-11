import re
from typing import List
from langchain_core.documents import Document

def chunk_text_by_size(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    # Split text into chunks of specified size with overlap.
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        
        chunk = text[start:end]
        chunks.append(chunk.strip())
        
        if end == text_length:
            break
        
        start = end - overlap
        if start <= 0:
            start = end
    
    return [chunk for chunk in chunks if chunk]

def chunk_by_sentences(text: str, max_chunk_size: int = 1000) -> List[str]:
    # Split text into chunks by sentences, respecting max chunk size.
    if not text:
        return []
    
    # Split by sentences using multiple delimiters
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence would exceed max size
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # If single sentence is too long, split it
                if len(sentence) > max_chunk_size:
                    word_chunks = chunk_by_words(sentence, max_chunk_size)
                    chunks.extend(word_chunks)
                else:
                    chunks.append(sentence)
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_by_words(text: str, max_chunk_size: int = 1000) -> List[str]:
    # Split text into chunks by words, respecting max chunk size.
    if not text:
        return []
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        
        if current_size + word_size > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                # Single word is too long, add it anyway
                chunks.append(word)
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def chunk_by_paragraphs(text: str, max_chunk_size: int = 1000) -> List[str]:
    # Split text into chunks by paragraphs, respecting max chunk size.
    if not text:
        return []
    
    # Split by paragraphs (double newlines or more)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed max size
        if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:  # +2 for newlines
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # If single paragraph is too long, split it further
                if len(paragraph) > max_chunk_size:
                    para_chunks = chunk_by_sentences(paragraph, max_chunk_size)
                    chunks.extend(para_chunks)
                else:
                    chunks.append(paragraph)
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_document(document: Document, chunk_size: int = 1000, overlap: int = 200, method: str = "size") -> List[Document]:
    # Chunk a document into smaller documents using specified method.
    if not document or not document.page_content:
        return []
    
    text = document.page_content
    
    # Choose chunking method
    if method == "sentences":
        chunks = chunk_by_sentences(text, chunk_size)
    elif method == "words":
        chunks = chunk_by_words(text, chunk_size)
    elif method == "paragraphs":
        chunks = chunk_by_paragraphs(text, chunk_size)
    else:  # default to size-based chunking
        chunks = chunk_text_by_size(text, chunk_size, overlap)
    
    # Create new documents for each chunk
    chunked_docs = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            # Copy metadata and add chunk info
            new_metadata = document.metadata.copy()
            new_metadata.update({
                "chunk_id": i,
                "chunk_method": method,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            })
            
            chunked_docs.append(Document(
                page_content=chunk.strip(),
                metadata=new_metadata
            ))
    
    return chunked_docs

def smart_chunking(text: str, max_chunk_size: int = 1000, min_chunk_size: int = 100) -> List[str]:
    # Intelligent chunking that tries paragraphs first, then sentences, then words.
    if not text:
        return []
    
    # First try paragraph-based chunking
    paragraph_chunks = chunk_by_paragraphs(text, max_chunk_size)
    
    # If paragraphs are too small, combine them
    if all(len(chunk) < min_chunk_size for chunk in paragraph_chunks) and len(paragraph_chunks) > 1:
        return chunk_text_by_size(text, max_chunk_size, overlap=200)
    
    # Check if any paragraph chunks are too large
    final_chunks = []
    for chunk in paragraph_chunks:
        if len(chunk) > max_chunk_size:
            # Split large paragraphs by sentences
            sentence_chunks = chunk_by_sentences(chunk, max_chunk_size)
            final_chunks.extend(sentence_chunks)
        else:
            final_chunks.append(chunk)
    
    return final_chunks