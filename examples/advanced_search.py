#!/usr/bin/env python3
"""
Advanced search examples for RDB.

This script demonstrates advanced search capabilities:
- Query refinement with LLMs
- Different search strategies
- Result filtering and ranking
- Performance analysis
- Batch searching
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdb import RDB
from rdb.config.settings import Config
from rdb.utils.logging import setup_logging
from rdb.utils.helpers import Timer, format_duration


class AdvancedSearchDemo:
   """Demonstrates advanced search capabilities."""
   
   def __init__(self, data_dir: str = "./example_data"):
       """Initialize the demo with RDB instance."""
       self.rdb = RDB(data_dir=data_dir)
       self.retriever = None
       
       # Check if index is available
       if self.rdb.config.index_file.exists():
           self.retriever = self.rdb.get_retriever()
           self.retriever.load_index()
           print("✓ Search index loaded successfully")
       else:
           print("✗ No search index found. Please run 'rdb build' first.")
   
   def demo_basic_search(self):
       """Demonstrate basic search functionality."""
       print("\n1. Basic Search Examples")
       print("=" * 40)
       
       if not self.retriever:
           print("Search index not available. Skipping demo.")
           return
       
       queries = [
           "install packages with pacman",
           "wireless network configuration", 
           "boot loader grub setup",
           "desktop environment kde plasma"
       ]
       
       for query in queries:
           print(f"\nQuery: '{query}'")
           
           with Timer() as timer:
               results = self.retriever.search(query, top_k=3)
           
           print(f"Search time: {timer}")
           print(f"Results: {len(results)}")
           
           for i, result in enumerate(results, 1):
               print(f"  {i}. [{result['chunk_type']}] {result['page_title']}")
               print(f"     Section: {result['section_path']}")
               print(f"     Score: {result['score']:.4f}")
   
   def demo_query_refinement(self):
       """Demonstrate query refinement with LLM."""
       print("\n2. Query Refinement Examples")
       print("=" * 40)
       
       if not self.retriever or not self.retriever.query_refiner:
           print("Query refinement not available. Skipping demo.")
           return
       
       test_queries = [
           "wifi broken",
           "sound doesn't work",
           "can't install software",
           "computer won't boot",
           "graphics are slow"
       ]
       
       for query in test_queries:
           print(f"\nOriginal query: '{query}'")
           
           try:
               # Show refinement
               refined = self.retriever.query_refiner.refine_query(query)
               print(f"Refined query:  '{refined}'")
               
               # Compare search results
               print("\nComparing results:")
               
               # Search without refinement
               results_original = self.retriever.search(query, top_k=3, refine_query=False)
               print(f"Original query: {len(results_original)} results")
               
               # Search with refinement
               results_refined = self.retriever.search(query, top_k=3, refine_query=True)
               print(f"Refined query:  {len(results_refined)} results")
               
               # Show top result from each
               if results_original:
                   top_orig = results_original[0]
                   print(f"  Original top result: {top_orig['page_title']} (score: {top_orig['score']:.4f})")
               
               if results_refined:
                   top_ref = results_refined[0]
                   print(f"  Refined top result:  {top_ref['page_title']} (score: {top_ref['score']:.4f})")
                   
           except Exception as e:
               print(f"Error with query refinement: {e}")
   
   def demo_chunk_type_analysis(self):
       """Analyze results by chunk type."""
       print("\n3. Chunk Type Analysis")
       print("=" * 40)
       
       if not self.retriever:
           print("Search index not available. Skipping demo.")
           return
       
       query = "arch linux installation guide"
       results = self.retriever.search(query, top_k=15)
       
       if not results:
           print("No results found for analysis.")
           return
       
       # Group by chunk type
       by_type = {}
       for result in results:
           chunk_type = result['chunk_type']
           if chunk_type not in by_type:
               by_type[chunk_type] = []
           by_type[chunk_type].append(result)
       
       print(f"Query: '{query}'")
       print(f"Total results: {len(results)}")
       print("\nResults by chunk type:")
       
       for chunk_type, chunks in by_type.items():
           avg_score = sum(c['score'] for c in chunks) / len(chunks)
           print(f"  {chunk_type}: {len(chunks)} results (avg score: {avg_score:.4f})")
           
           # Show best result for this type
           best = max(chunks, key=lambda x: x['score'])
           print(f"    Best: {best['page_title']} - {best['section_path']}")
           print(f"          Score: {best['score']:.4f}")
   
   def demo_batch_search(self):
       """Demonstrate batch searching with performance analysis."""
       print("\n4. Batch Search Performance")
       print("=" * 40)
       
       if not self.retriever:
           print("Search index not available. Skipping demo.")
           return
       
       # Test queries covering different topics
       batch_queries = [
           "pacman package manager",
           "systemd services",
           "xorg display server", 
           "pulseaudio configuration",
           "networkmanager setup",
           "grub bootloader",
           "kde plasma desktop",
           "arch user repository aur",
           "makepkg build packages",
           "journalctl view logs"
       ]
       
       print(f"Running batch search with {len(batch_queries)} queries...")
       
       all_results = []
       total_time = 0
       
       for i, query in enumerate(batch_queries, 1):
           with Timer() as timer:
               results = self.retriever.search(query, top_k=5)
           
           search_time = timer.elapsed
           total_time += search_time
           all_results.extend(results)
           
           print(f"  {i:2d}. '{query[:30]}...' -> {len(results)} results ({search_time*1000:.1f}ms)")
       
       # Performance summary
       avg_time = total_time / len(batch_queries)
       total_results = len(all_results)
       
       print(f"\nBatch Performance Summary:")
       print(f"  Total queries: {len(batch_queries)}")
       print(f"  Total time: {format_duration(total_time)}")
       print(f"  Average time per query: {avg_time*1000:.1f}ms")
       print(f"  Total results: {total_results}")
       print(f"  Average results per query: {total_results/len(batch_queries):.1f}")
       
       # Analyze result diversity
       unique_pages = set(r['page_title'] for r in all_results)
       print(f"  Unique pages found: {len(unique_pages)}")
   
   def demo_result_filtering(self):
       """Demonstrate filtering and ranking results."""
       print("\n5. Result Filtering and Ranking")
       print("=" * 40)
       
       if not self.retriever:
           print("Search index not available. Skipping demo.")
           return
       
       query = "linux kernel modules"
       results = self.retriever.search(query, top_k=20)
       
       if not results:
           print("No results found for filtering demo.")
           return
       
       print(f"Query: '{query}'")
       print(f"Initial results: {len(results)}")
       
       # Filter 1: High confidence results only
       high_confidence = [r for r in results if r['score'] > 0.7]
       print(f"\nHigh confidence results (score > 0.7): {len(high_confidence)}")
       
       for result in high_confidence[:3]:
           print(f"  • {result['page_title']} - {result['section_path']}")
           print(f"    Score: {result['score']:.4f}, Type: {result['chunk_type']}")
       
       # Filter 2: Specific chunk types
       detailed_results = [r for r in results if r['chunk_type'] in ['medium', 'large']]
       print(f"\nDetailed results (medium/large chunks): {len(detailed_results)}")
       
       # Filter 3: Specific pages
       kernel_pages = [r for r in results if 'kernel' in r['page_title'].lower()]
       print(f"\nKernel-specific pages: {len(kernel_pages)}")
       
       # Custom ranking: boost official documentation
       def custom_score(result):
           base_score = result['score']
           # Boost if it's from main installation/configuration pages
           title_lower = result['page_title'].lower()
           if any(keyword in title_lower for keyword in ['kernel', 'modules', 'driver']):
               base_score *= 1.2
           return base_score
       
       # Re-rank with custom scoring
       custom_ranked = sorted(results, key=custom_score, reverse=True)
       
       print(f"\nTop 3 with custom ranking:")
       for i, result in enumerate(custom_ranked[:3], 1):
           custom_score_val = custom_score(result)
           print(f"  {i}. {result['page_title']}")
           print(f"     Original score: {result['score']:.4f}, Custom score: {custom_score_val:.4f}")
   
   def demo_search_analytics(self):
       """Demonstrate search analytics and insights."""
       print("\n6. Search Analytics")
       print("=" * 40)
       
       if not self.retriever:
           print("Search index not available. Skipping demo.")
           return
       
       # Get index statistics
       stats = self.retriever.index_manager.get_stats()
       
       print("Index Statistics:")
       print(f"  Total vectors: {stats.get('total_vectors', 'N/A')}")
       print(f"  Total chunks: {stats.get('total_chunks', 'N/A')}")
       print(f"  Vector dimension: {stats.get('vector_dimension', 'N/A')}")
       
       if 'chunk_types' in stats:
           print("  Chunk type distribution:")
           for chunk_type, count in stats['chunk_types'].items():
               percentage = (count / stats['total_chunks']) * 100
               print(f"    {chunk_type}: {count} ({percentage:.1f}%)")
       
       # Test different query types
       query_types = {
           "Installation": "install arch linux",
           "Configuration": "configure network settings", 
           "Troubleshooting": "fix boot problems",
           "Package Management": "pacman install packages",
           "Desktop Environment": "setup kde plasma"
       }
       
       print(f"\nQuery Type Analysis:")
       
       for category, query in query_types.items():
           results = self.retriever.search(query, top_k=10)
           
           if results:
               avg_score = sum(r['score'] for r in results) / len(results)
               chunk_types = {}
               for r in results:
                   ct = r['chunk_type']
                   chunk_types[ct] = chunk_types.get(ct, 0) + 1
               
               print(f"  {category}:")
               print(f"    Average score: {avg_score:.4f}")
               print(f"    Chunk types: {dict(chunk_types)}")
   
   def run_all_demos(self):
       """Run all demonstration functions."""
       print("RDB Advanced Search Demonstration")
       print("=" * 50)
       
       if not self.retriever:
           print("\n❌ Search index not available!")
           print("Please run the following commands first:")
           print("1. rdb scrape")
           print("2. rdb build")
           return
       
       try:
           self.demo_basic_search()
           self.demo_query_refinement()
           self.demo_chunk_type_analysis()
           self.demo_batch_search()
           self.demo_result_filtering()
           self.demo_search_analytics()
           
           print(f"\n🎉 All demos completed successfully!")
           
       except Exception as e:
           print(f"\n❌ Error during demo: {e}")
           import traceback
           traceback.print_exc()


def main():
   """Run the advanced search demonstration."""
   setup_logging(log_level="INFO")
   
   demo = AdvancedSearchDemo()
   demo.run_all_demos()
   
   print(f"\n" + "=" * 50)
   print("Advanced Search Demo Complete")
   print("=" * 50)
   print("\nTry these commands for more exploration:")
   print("- rdb search --interactive")
   print("- rdb search 'your query here' --refine")
   print("- rdb search --help")


if __name__ == "__main__":
   main()
