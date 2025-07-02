#!/usr/bin/env python3
"""
Basic usage examples for RDB.

This script demonstrates the fundamental operations of RDB:
- Scraping Arch Wiki documentation
- Processing and chunking documents
- Creating embeddings and building search index
- Performing semantic search queries
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdb import RDB
from rdb.config.settings import Config
from rdb.utils.logging import setup_logging


def main():
   """Demonstrate basic RDB usage."""
   
   # Setup logging
   setup_logging(log_level="INFO")
   
   print("RDB Basic Usage Example")
   print("=" * 50)
   
   # Initialize RDB with custom data directory
   data_dir = Path("./example_data")
   rdb = RDB(data_dir=str(data_dir))
   
   print(f"Initialized RDB with data directory: {data_dir}")
   print()
   
   # Example 1: Check current status
   print("1. Checking RDB Status")
   print("-" * 30)
   
   config = rdb.config
   print(f"Raw data directory: {config.raw_data_dir}")
   print(f"Chunks file: {config.chunks_file}")
   print(f"Index file: {config.index_file}")
   print(f"Embedding model: {config.embedding_model}")
   print()
   
   # Check if data exists
   if config.raw_data_dir.exists():
       json_files = list(config.raw_data_dir.glob("*.json"))
       print(f"Found {len(json_files)} scraped files")
   else:
       print("No scraped data found")
   
   if config.index_file.exists():
       print("Search index exists ✓")
   else:
       print("Search index not found ✗")
   
   print()
   
   # Example 2: Scraping (commented out as it takes time)
   print("2. Scraping Example (commented out)")
   print("-" * 40)
   print("# To scrape Arch Wiki documentation:")
   print("# success_count = rdb.scrape()")
   print("# print(f'Successfully scraped {success_count} pages')")
   print()
   
   # Example 3: Building index (if scraped data exists)
   print("3. Building Search Index")
   print("-" * 30)
   
   if config.raw_data_dir.exists() and list(config.raw_data_dir.glob("*.json")):
       print("Scraped data found. Building index...")
       try:
           chunk_count = rdb.build_index()
           print(f"Successfully built index with {chunk_count} chunks")
       except Exception as e:
           print(f"Error building index: {e}")
   else:
       print("No scraped data found. Skipping index building.")
       print("Run scraping first: rdb.scrape()")
   
   print()
   
   # Example 4: Searching (if index exists)
   print("4. Search Examples")
   print("-" * 20)
   
   if config.index_file.exists():
       print("Search index found. Running example searches...")
       
       # Example queries
       queries = [
           "wifi connection problems",
           "install arch linux",
           "graphics driver issues",
           "sound not working"
       ]
       
       for query in queries:
           print(f"\nQuery: '{query}'")
           try:
               results = rdb.search(query, top_k=3)
               
               if results:
                   print(f"Found {len(results)} results:")
                   for i, result in enumerate(results, 1):
                       print(f"  {i}. {result['page_title']} - {result['section_path']}")
                       print(f"     Score: {result['score']:.3f}")
                       print(f"     URL: {result['url']}")
                       # Show first 100 characters of content
                       content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                       print(f"     Preview: {content_preview}")
                       print()
               else:
                   print("  No results found")
                   
           except Exception as e:
               print(f"  Error searching: {e}")
       
   else:
       print("No search index found. Build index first:")
       print("rdb.build_index()")
   
   print()
   
   # Example 5: Advanced usage
   print("5. Advanced Usage Examples")
   print("-" * 30)
   
   print("# Search with query refinement:")
   print("# results = rdb.search('wifi broken', refine_query=True)")
   print()
   
   print("# Access individual components:")
   print("# scraper = rdb.get_scraper()")
   print("# chunker = rdb.get_chunker()") 
   print("# embedder = rdb.get_embedder()")
   print("# retriever = rdb.get_retriever()")
   print()
   
   print("# Custom configuration:")
   print("# config = Config(data_dir='./custom_data')")
   print("# config.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'")
   print("# rdb = RDB(config=config)")
   print()
   
   print("Example completed!")
   print()
   print("Next steps:")
   print("- Run 'python -m cli.main scrape' to scrape Arch Wiki")
   print("- Run 'python -m cli.main build' to build search index")
   print("- Run 'python -m cli.main search --interactive' for interactive search")


if __name__ == "__main__":
   main()
