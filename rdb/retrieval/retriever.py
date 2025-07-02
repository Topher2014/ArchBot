"""
Document retriever for semantic search.
"""

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config.settings import Config
from ..utils.logging import get_logger
from ..embedding.models import EmbeddingModel
from .refiner import QueryRefiner
from .index_manager import IndexManager


class DocumentRetriever:
   """Retrieves documents using semantic search with optional query refinement."""
   
   def __init__(self, config: Config):
       """Initialize document retriever with configuration."""
       self.config = config
       self.logger = get_logger(__name__)
       self.embedding_model = EmbeddingModel(config.embedding_model, device=config.device)
       self.index_manager = IndexManager(config)
       self.query_refiner = None
       
       # Initialize query refiner if enabled
       if config.enable_query_refinement:
           try:
               self.query_refiner = QueryRefiner(config)
               self.logger.info("Query refinement enabled")
           except Exception as e:
               self.logger.warning(f"Could not load query refiner: {e}")
               self.logger.info("Continuing without query refinement")
   
   def load_index(self, index_dir: Optional[str] = None) -> bool:
       """Load FAISS index and metadata."""
       return self.index_manager.load_index(index_dir)
   
   def search(self, query: str, top_k: Optional[int] = None, 
              refine_query: bool = False, show_refinement: bool = False) -> List[Dict[str, Any]]:
       """Search for similar documents with optional query refinement."""
       if not self.index_manager.is_loaded():
           if not self.load_index():
               raise RuntimeError("Index not loaded and could not load from default location")
       
       if top_k is None:
           top_k = self.config.default_top_k
       
       original_query = query
       
       # Apply query refinement if requested and available
       if refine_query and self.query_refiner:
           try:
               refined_query = self.query_refiner.refine_query(query)
               if show_refinement:
                   self.logger.info(f"Original query: {original_query}")
                   self.logger.info(f"Refined query:  {refined_query}")
               query = refined_query
           except Exception as e:
               self.logger.warning(f"Query refinement failed: {e}")
               self.logger.info("Using original query")
       
       # Encode query
       query_embedding = self.embedding_model.encode_query(query)
       query_embedding = query_embedding.reshape(1, -1).astype('float32')
       
       # Normalize for cosine similarity
       faiss.normalize_L2(query_embedding)
       
       # Search
       scores, indices = self.index_manager.search(query_embedding, top_k)
       
       # Format results
       results = []
       for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
           if idx < len(self.index_manager.chunks):
               chunk = self.index_manager.chunks[idx]
               results.append({
                   'rank': i + 1,
                   'score': float(score),
                   'page_title': chunk['page_title'],
                   'section_path': chunk['section_path'],
                   'url': chunk['url'],
                   'content': chunk['content'],
                   'chunk_type': chunk['chunk_type'],
                   'section_level': chunk['section_level'],
                   'original_query': original_query,
                   'final_query': query,
                   'full_chunk': chunk
               })
       
       return results
   
   def search_interactive(self, top_k: Optional[int] = None, show_refinement: bool = True):
       """Interactive search loop."""
       if top_k is None:
           top_k = self.config.default_top_k
       
       print("\n" + "="*60)
       print("RDB INTERACTIVE SEARCH")
       print(f"Query refinement: {'ENABLED' if self.query_refiner else 'DISABLED'}")
       print("Enter queries to search (or 'quit' to exit)")
       print(f"Showing top {top_k} results per query")
       print("Commands:")
       print("  'quit' or 'exit' - Exit the program")
       print("  'help' - Show this help")
       print("  'stats' - Show index statistics")
       print("  'toggle' - Toggle query refinement display")
       if self.query_refiner:
           print("  'refine <query>' - Show refinement for a query without searching")
       print("="*60)
       
       while True:
           try:
               user_input = input("\nQuery: ").strip()
               
               if user_input.lower() in ['quit', 'exit', 'q']:
                   print("Goodbye!")
                   break
               elif user_input.lower() == 'help':
                   print("\nCommands:")
                   print("  'quit' or 'exit' - Exit the program")
                   print("  'help' - Show this help")
                   print("  'stats' - Show index statistics")
                   print("  'toggle' - Toggle query refinement display")
                   if self.query_refiner:
                       print("  'refine <query>' - Show refinement for a query without searching")
                   continue
               elif user_input.lower() == 'stats':
                   stats = self.index_manager.get_stats()
                   print(f"\nIndex Statistics:")
                   for key, value in stats.items():
                       print(f"  {key}: {value}")
                   continue
               elif user_input.lower() == 'toggle':
                   show_refinement = not show_refinement
                   print(f"Query refinement display: {'ON' if show_refinement else 'OFF'}")
                   continue
               elif user_input.lower().startswith('refine ') and self.query_refiner:
                   query = user_input[7:].strip()
                   if query:
                       refined = self.query_refiner.refine_query(query)
                       print(f"Original: {query}")
                       print(f"Refined:  {refined}")
                   continue
               elif not user_input:
                   continue
               
               print(f"\nSearching for: '{user_input}'")
               results = self.search(user_input, top_k=top_k, 
                                   refine_query=bool(self.query_refiner), 
                                   show_refinement=show_refinement)
               
               if not results:
                   print("No results found.")
               else:
                   self._print_results(results, show_queries=False)
                   
           except KeyboardInterrupt:
               print("\n\nGoodbye!")
               break
           except Exception as e:
               print(f"Error during search: {e}")
   
   def _print_results(self, results: List[Dict], max_content_length: int = 300, 
                     show_queries: bool = False):
       """Pretty print search results."""
       if show_queries and results:
           print(f"\nOriginal query: {results[0]['original_query']}")
           if results[0]['original_query'] != results[0]['final_query']:
               print(f"Refined query:  {results[0]['final_query']}")
       
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
