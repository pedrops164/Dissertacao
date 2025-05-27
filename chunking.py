import nltk
from nltk.tokenize import sent_tokenize
import re

# --- Text Chunking Functions ---
def chunk_text(text, max_chunk_size=2000, overlap=100):
    """
    Split text into chunks with a maximum size and optional overlap.
    
    Args:
        text (str): The text to chunk
        max_chunk_size (int): Target size for each chunk (in characters)
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    # Download necessary NLTK data if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # If text is shorter than max_chunk_size, return as is
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the max chunk size,
        # save the current chunk and start a new one
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from the end of the previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                # Try to start the new chunk at a sentence boundary within the overlap region
                overlap_text = current_chunk[-overlap:]
                # Find the last sentence boundary in the overlap text
                last_period = overlap_text.rfind('. ')
                if last_period != -1:
                    # Start with the last complete sentence in the overlap
                    current_chunk = current_chunk[-(overlap - last_period):]
                else:
                    # No sentence boundary found, just use the overlap as is
                    current_chunk = overlap_text
            else:
                current_chunk = ""
        
        # Add the sentence to the current chunk
        current_chunk += " " + sentence
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_with_metadata(text, source=None, max_chunk_size=1000, overlap=100):
    """
    Chunk text and add metadata to each chunk.
    
    Args:
        text (str): The text to chunk
        source (str): Optional source information
        max_chunk_size (int): Maximum size for each chunk
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of dictionaries with text chunks and metadata
    """
    chunks = chunk_text(text, max_chunk_size, overlap)
    
    # Extract title from the first paragraph if possible
    title = "Unknown"
    if text:
        # Try to get title from the first line or paragraph
        first_para = text.split('\n\n')[0] if '\n\n' in text else text.split('\n')[0]
        # Clean up - remove special chars, limit length
        title = re.sub(r'[^\w\s]', '', first_para[:50]).strip()
    
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "text": chunk,
            "metadata": {
                "source": source,
                "title": title,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
        })
    
    return result