#!/usr/bin/env python3
"""
Arch Wiki Document Embedder using E5-Large-V2
Embeds your arch_chunks.json and creates a searchable index
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import pickle

class ArchWikiEmbedder:
    def __init__(self, model_name='intfloat/e5-large-v2'):
        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunks = None
        self.index = None
        
    def load_chunks(self, json_file='arch_chunks.json'):
        """Load your arch wiki chunks"""
        print(f"Loading chunks from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"Loaded {len(self.chunks)} chunks")
        return self.chunks
    
    def create_embeddings(self, batch_size=32):
        """Create embeddings for all chunks"""
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")
        
        print("Preparing documents for embedding...")
        documents = []
        
        for chunk in self.chunks:
            # Use the chunk_text which already has title + content
            # Add e5's required "passage: " prefix
            doc_text = f"passage: {chunk['chunk_text']}"
            documents.append(doc_text)
        
        print(f"Creating embeddings for {len(documents)} documents...")
        print("This may take several minutes...")
        
        # Create embeddings in batches to avoid memory issues
        all_embeddings = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        print(f"Created embeddings: shape {embeddings.shape}")
        
        return embeddings
    
    def build_index(self, embeddings):
        """Build FAISS index from embeddings"""
        print("Building FAISS index...")
        
        dimension = embeddings.shape[1]
        print(f"Vector dimension: {dimension}")
        
        # Use IndexFlatIP for inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        print("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)
        
        # Add to index
        print("Adding embeddings to index...")
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
        return self.index
    
    def save_index(self, index_file='arch_wiki_index.faiss', metadata_file='arch_wiki_metadata.pkl'):
        """Save the index and metadata"""
        print(f"Saving index to {index_file}...")
        faiss.write_index(self.index, index_file)
        
        print(f"Saving metadata to {metadata_file}...")
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print("Index and metadata saved!")
    
    def load_index(self, index_file='arch_wiki_index.faiss', metadata_file='arch_wiki_metadata.pkl'):
        """Load previously saved index and metadata"""
        print(f"Loading index from {index_file}...")
        self.index = faiss.read_index(index_file)
        
        print(f"Loading metadata from {metadata_file}...")
        with open(metadata_file, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
    
    def search(self, query, top_k=5):
        """Search for similar documents"""
        if not self.index or not self.chunks:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        # Add e5's required "query: " prefix
        query_text = f"query: {query}"
        
        # Embed query
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Safety check
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'page_title': self.chunks[idx]['page_title'],
                    'section_path': self.chunks[idx]['section_path'],
                    'url': self.chunks[idx]['url'],
                    'content': self.chunks[idx]['content'],
                    'chunk_type': self.chunks[idx]['chunk_type'],
                    'full_chunk': self.chunks[idx]
                })
        
        return results
    
    def print_results(self, results):
        """Pretty print search results"""
        for result in results:
            print(f"\n{'='*60}")
            print(f"Rank {result['rank']} | Score: {result['score']:.4f}")
            print(f"Page: {result['page_title']}")
            print(f"Section: {result['section_path']}")
            print(f"Type: {result['chunk_type']}")
            print(f"URL: {result['url']}")
            print("-" * 60)
            # Truncate content for display
            content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            print(content)


def main():
    """Main function to embed your Arch Wiki chunks"""
    import sys
    
    # Get JSON file path from command line or use default
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'arch_chunks.json'
    
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found!")
        print("Usage: python arch_wiki_embedder.py [path_to_chunks.json]")
        return
    
    embedder = ArchWikiEmbedder()
    
    # Check if index already exists
    if os.path.exists('arch_wiki_index.faiss') and os.path.exists('arch_wiki_metadata.pkl'):
        print("Found existing index files.")
        choice = input("Load existing index? (y/n): ").lower().strip()
        if choice == 'y':
            embedder.load_index()
        else:
            # Create new index
            embedder.load_chunks(json_file)
            embeddings = embedder.create_embeddings()
            embedder.build_index(embeddings)
            embedder.save_index()
    else:
        # Create new index
        embedder.load_chunks(json_file)
        embeddings = embedder.create_embeddings()
        embedder.build_index(embeddings)
        embedder.save_index()
    
    # Interactive search
    print("\n" + "="*60)
    print("ARCH WIKI SEARCH - Ready!")
    print("Enter queries to search (or 'quit' to exit)")
    print("="*60)
    
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if query:
            print(f"\nSearching for: '{query}'")
            results = embedder.search(query, top_k=3)
            embedder.print_results(results)


if __name__ == "__main__":
    main()
