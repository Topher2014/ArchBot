#!/usr/bin/env python3
"""
Arch Wiki Document Retriever using E5-Large-V2
Searches through pre-built embeddings index
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import pickle
import argparse


class ArchWikiRetriever:
    def __init__(self, model_name='intfloat/e5-large-v2'):
        print(f"Loading {model_name}...")
        # Force CPU-only to avoid GPU power draw
        self.model = SentenceTransformer(model_name, device='cpu')
        self.chunks = None
        self.index = None
        
    def load_index(self, index_dir='.', base_name='arch_wiki'):
        """Load previously saved index and metadata"""
        index_file = os.path.join(index_dir, f'{base_name}_index.faiss')
        metadata_file = os.path.join(index_dir, f'{base_name}_metadata.pkl')
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        print(f"Loading index from {index_file}...")
        self.index = faiss.read_index(index_file)
        
        print(f"Loading metadata from {metadata_file}...")
        with open(metadata_file, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
    
    def search(self, query, top_k=3):
        """Search for similar documents"""
        if not self.index or not self.chunks:
            raise ValueError("Index not loaded. Call load_index() first.")
        
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
    
    def print_results(self, results, max_content_length=300):
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
            content = result['content']
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            print(content)
    
    def search_interactive(self, top_k=5):
        """Interactive search loop"""
        print("\n" + "="*60)
        print("ARCH WIKI SEARCH - Ready!")
        print("Enter queries to search (or 'quit' to exit)")
        print(f"Showing top {top_k} results per query")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the program")
        print("  'help' - Show this help")
        print("  'stats' - Show index statistics")
        print("="*60)
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif query.lower() == 'help':
                    print("\nCommands:")
                    print("  'quit' or 'exit' - Exit the program")
                    print("  'help' - Show this help")
                    print("  'stats' - Show index statistics")
                    continue
                elif query.lower() == 'stats':
                    print(f"\nIndex Statistics:")
                    print(f"  Total vectors: {self.index.ntotal}")
                    print(f"  Total chunks: {len(self.chunks)}")
                    print(f"  Vector dimension: {self.index.d}")
                    continue
                elif not query:
                    continue
                
                print(f"\nSearching for: '{query}'")
                results = self.search(query, top_k=top_k)
                
                if not results:
                    print("No results found.")
                else:
                    self.print_results(results)
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error during search: {e}")


def main():
    """Main function for the retriever"""
    parser = argparse.ArgumentParser(description='Search Arch Wiki embeddings')
    parser.add_argument('-i', '--index-dir', default='.',
                       help='Directory containing index files (default: current directory)')
    parser.add_argument('-n', '--name', default='arch_wiki',
                       help='Base name for index files (default: arch_wiki)')
    parser.add_argument('-k', '--top-k', type=int, default=5,
                       help='Number of results to return (default: 5)')
    parser.add_argument('-m', '--model', default='intfloat/e5-large-v2',
                       help='Sentence transformer model name (default: intfloat/e5-large-v2)')
    parser.add_argument('-q', '--query', 
                       help='Single query to search (non-interactive mode)')
    parser.add_argument('--max-content', type=int, default=300,
                       help='Maximum content length to display (default: 300)')
    
    args = parser.parse_args()
    
    try:
        # Initialize retriever
        retriever = ArchWikiRetriever(model_name=args.model)
        retriever.load_index(args.index_dir, args.name)
        
        if args.query:
            # Single query mode
            print(f"Searching for: '{args.query}'")
            results = retriever.search(args.query, top_k=args.top_k)
            
            if not results:
                print("No results found.")
                return 1
            else:
                retriever.print_results(results, max_content_length=args.max_content)
                return 0
        else:
            # Interactive mode
            retriever.search_interactive(top_k=args.top_k)
            return 0
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure you have run the embedder first:")
        print(f"python embedder.py arch_chunks.json -o {args.index_dir}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
