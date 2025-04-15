import os
import json
import glob
import numpy as np
import time
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Configuration
DATA_DIR = "arch_wiki_data"
VECTOR_DIR = "arch_wiki_vectors"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient for our use case
CHUNK_SIZE = 512  # Approximate size of text chunks in characters
CHUNK_OVERLAP = 50  # Overlap between chunks to maintain context
MAX_CHUNKS_PER_SECTION = 10  # Limit very large sections

# Create output directory
os.makedirs(VECTOR_DIR, exist_ok=True)

def log_message(message):
    """Write message to console and log file"""
    print(message)
    with open(os.path.join(VECTOR_DIR, "vectorize_log.txt"), "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

def load_wiki_pages():
    """Load all scraped wiki pages from JSON files"""
    log_message("Loading wiki pages...")
    wiki_pages = []
    
    # Get all JSON files in the data directory
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    total_files = len(json_files)
    log_message(f"Found {total_files} wiki pages")
    
    for i, file_path in enumerate(json_files):
        if i % 100 == 0:
            log_message(f"Loading file {i+1}/{total_files}")
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                page_data = json.load(f)
                wiki_pages.append(page_data)
        except Exception as e:
            log_message(f"Error loading {file_path}: {str(e)}")
    
    log_message(f"Loaded {len(wiki_pages)} wiki pages successfully")
    return wiki_pages                          

def split_section_into_chunks(section_text, section_title, page_title, page_url):
    """Split a section into smaller chunks for better semantic search"""
    # Don't split short sections
    if len(section_text) <= CHUNK_SIZE:
        return [{
            "text": section_text,
            "title": section_title,
            "page_title": page_title,
            "url": page_url,
            "chunk_index": 0
        }]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    # Split by paragraphs first
    paragraphs = section_text.split("\n\n")
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) > CHUNK_SIZE and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "title": section_title,
                "page_title": page_title,
                "url": page_url,
                "chunk_index": chunk_index
            })
            chunk_index += 1
            current_chunk = paragraph + "\n\n"
        else:
            current_chunk += paragraph + "\n\n"
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "title": section_title,
            "page_title": page_title,
            "url": page_url,
            "chunk_index": chunk_index
        })
    
    # Limit number of chunks to prevent very large sections from dominating
    if len(chunks) > MAX_CHUNKS_PER_SECTION:
        log_message(f"Limiting chunks for section '{section_title}' from {len(chunks)} to {MAX_CHUNKS_PER_SECTION}")
        chunks = chunks[:MAX_CHUNKS_PER_SECTION]
    
    return chunks

def create_chunks_from_wiki_pages(wiki_pages):
    """Extract chunks from wiki pages for embedding"""
    log_message("Creating text chunks from wiki pages...")
    all_chunks = []
    
    for page in wiki_pages:
        # Check if page is a dictionary with expected keys
        if not isinstance(page, dict) or "title" not in page:
            continue
            
        page_title = page["title"]
        page_url = page["url"]
        
        for section in page["sections"]:
            section_title = section["title"]
            section_content = section["content"]
            
            # Skip empty sections
            if not section_content.strip():
                continue
                
            # Skip "See also", "References", etc.
            if section_title.lower() in ["see also", "references", "external links"]:
                continue
            
            # Create a full reference string
            full_reference = f"{page_title} - {section_title}"
            
            # Split the section into chunks
            section_chunks = split_section_into_chunks(
                section_content, 
                section_title, 
                page_title, 
                page_url
            )
            
            all_chunks.extend(section_chunks)
    
    log_message(f"Created {len(all_chunks)} chunks from {len(wiki_pages)} wiki pages")
    return all_chunks             

             
def create_embeddings(chunks):
    """Create embeddings for all text chunks"""
    log_message(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    log_message("Creating embeddings for chunks...")
    texts = [
        f"Title: {chunk['page_title']} | Section: {chunk['title']}\n\n{chunk['text']}"
        for chunk in chunks
    ]
    
    # Process in batches to manage memory
    batch_size = 32
    embeddings = []
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch = texts[i:batch_end]
        
        log_message(f"Creating embeddings for batch {i+1}-{batch_end} of {total_chunks}")
        batch_embeddings = model.encode(batch, show_progress_bar=True)
        embeddings.extend(batch_embeddings)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    log_message(f"Created embeddings with shape: {embeddings_array.shape}")
    
    return embeddings_array, model

def build_faiss_index(embeddings):
    """Build a FAISS index for fast similarity search"""
    log_message("Building FAISS index...")
    
    # Get dimension of embeddings
    dimension = embeddings.shape[1]
    
    # Create a flat index (exact search)
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    index.add(embeddings)
    
    log_message(f"Built FAISS index with {index.ntotal} vectors")
    return index

def save_vector_database(chunks, embeddings, index, model_name):
    """Save all vector database components to disk"""
    log_message("Saving vector database to disk...")
    
    # Save chunks data
    with open(os.path.join(VECTOR_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    
    # Save embeddings
    np.save(os.path.join(VECTOR_DIR, "embeddings.npy"), embeddings)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(VECTOR_DIR, "wiki.index"))
    
    # Save model name
    with open(os.path.join(VECTOR_DIR, "model_info.json"), "w") as f:
        json.dump({"model": model_name, "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}, f)
    
    log_message("Vector database saved to disk")

def main():
    """Main function to create vector database from wiki content"""
    start_time = time.time()
    log_message(f"Starting vectorization at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load wiki pages
    wiki_pages = load_wiki_pages()
    
    # Create chunks
    chunks = create_chunks_from_wiki_pages(wiki_pages)
    
    # Create embeddings
    embeddings, model = create_embeddings(chunks)
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Save everything
    save_vector_database(chunks, embeddings, index, EMBEDDING_MODEL)
    
    duration = time.time() - start_time
    log_message(f"Vectorization completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")

if __name__ == "__main__":
    main()             
