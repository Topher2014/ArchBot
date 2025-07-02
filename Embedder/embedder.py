#!/usr/bin/env python3
"""
Arch Wiki Document Embedder using E5-Large-V2
Creates embeddings for arch_chunks.json and saves searchable index
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import pickle
import argparse


class ArchWikiEmbedder:
    def __init__(self, model_name='intfloat/e5-large-v2'):
        print(f"Loading {model_name}...")
        # Force CPU-only to avoid GPU power draw
        self.model = SentenceTransformer(model_name, device='cpu')
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
    
    def save_index(self, output_dir='.', base_name='arch_wiki'):
        """Save the index and metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        index_file = os.path.join(output_dir, f'{base_name}_index.faiss')
        metadata_file = os.path.join(output_dir, f'{base_name}_metadata.pkl')
        
        print(f"Saving index to {index_file}...")
        faiss.write_index(self.index, index_file)
        
        print(f"Saving metadata to {metadata_file}...")
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print("Index and metadata saved!")
        print(f"Files created:")
        print(f"  - {index_file}")
        print(f"  - {metadata_file}")
        return index_file, metadata_file


def main():
    """Main function to embed your Arch Wiki chunks"""
    parser = argparse.ArgumentParser(description='Create embeddings for Arch Wiki chunks')
    parser.add_argument('json_file', 
                       help='Path to arch_chunks.json file')
    parser.add_argument('-o', '--output', default='.',
                       help='Output directory for index files (default: current directory)')
    parser.add_argument('-n', '--name', default='arch_wiki',
                       help='Base name for output files (default: arch_wiki)')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                       help='Batch size for embedding creation (default: 32)')
    parser.add_argument('-m', '--model', default='intfloat/e5-large-v2',
                       help='Sentence transformer model name (default: intfloat/e5-large-v2)')
    parser.add_argument('-f', '--force', action='store_true',
                       help='Overwrite existing index files without asking')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.json_file):
        print(f"Error: File '{args.json_file}' not found!")
        return 1
    
    # Check if output files already exist
    index_file = os.path.join(args.output, f'{args.name}_index.faiss')
    metadata_file = os.path.join(args.output, f'{args.name}_metadata.pkl')
    
    if not args.force and (os.path.exists(index_file) or os.path.exists(metadata_file)):
        print("Warning: Output files already exist:")
        if os.path.exists(index_file):
            print(f"  - {index_file}")
        if os.path.exists(metadata_file):
            print(f"  - {metadata_file}")
        
        choice = input("Overwrite existing files? (y/n): ").lower().strip()
        if choice != 'y':
            print("Aborted.")
            return 1
    
    # Create embedder and process
    try:
        embedder = ArchWikiEmbedder(model_name=args.model)
        embedder.load_chunks(args.json_file)
        embeddings = embedder.create_embeddings(batch_size=args.batch_size)
        embedder.build_index(embeddings)
        embedder.save_index(args.output, args.name)
        
        print("\n" + "="*60)
        print("SUCCESS! Embeddings created and saved.")
        print(f"To search, use: python retriever.py -i {args.output}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
