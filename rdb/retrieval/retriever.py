"""
Document retriever for semantic search with deduplication.
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
   """Retrieves documents using semantic search with optional query refinement and deduplication."""
   
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
              refine_query: bool = False, show_refinement: bool = False,
              enable_deduplication: bool = True) -> List[Dict[str, Any]]:
       """Search for similar documents with optional query refinement and deduplication."""
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
       
       # Search with higher top_k to account for deduplication
       search_k = top_k * 3 if enable_deduplication else top_k
       scores, indices = self.index_manager.search(query_embedding, search_k)
       
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
       
       # Apply boosting logic
       query_words = set(query.lower().split())
       for result in results:
           # Boost exact page title matches
           title_words = set(result['page_title'].lower().replace('_', ' ').split())
           if query_words.intersection(title_words):
               result['score'] *= 1.3  # Strong boost for page title match
           
           # Boost medium/large chunks over small intro chunks
           if result['chunk_type'] in ['medium', 'large']:
               result['score'] *= 1.1
           
           # Boost chunks with actual configuration content
           content_lower = result['content'].lower()
           if any(word in content_lower for word in ['connect', 'configure', 'setup', 'install']) and len(result['content']) > 200:
               result['score'] *= 1.1
       
       # Re-sort by boosted scores
       results.sort(key=lambda x: x['score'], reverse=True)
       
       # Apply deduplication if enabled
       if enable_deduplication:
           results = self._deduplicate_results(results)
           self.logger.debug(f"Deduplication reduced results from {len(results)} to {len(results)}")
       
       # Trim to requested top_k
       results = results[:top_k]
       
       # Update ranks after deduplication and trimming
       for i, result in enumerate(results):
           result['rank'] = i + 1
       
       return results
   
   def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       """Remove duplicate results based on content similarity and page titles."""
       if not results:
           return results
       
       seen_content = {}
       deduplicated = []
       original_count = len(results)
       
       for result in results:
           # Create content hash for exact content matching
           content_hash = hash(result['content'].strip())
           
           # Also check for very similar page titles (case-insensitive, punctuation-normalized)
           normalized_title = self._normalize_title(result['page_title'])
           
           # Create a composite key that considers both content and normalized title
           dedup_key = (content_hash, normalized_title)
           
           if dedup_key not in seen_content:
               # First occurrence of this content/title combination
               seen_content[dedup_key] = len(deduplicated)
               result['aliases'] = [result['page_title']]  # Track aliases
               deduplicated.append(result)
           else:
               # Duplicate found - decide which one to keep
               existing_idx = seen_content[dedup_key]
               existing_result = deduplicated[existing_idx]
               
               # Add current page title as an alias
               if result['page_title'] not in existing_result['aliases']:
                   existing_result['aliases'].append(result['page_title'])
               
               # Keep the result with higher score
               if result['score'] > existing_result['score']:
                   # Update with higher scoring version but keep accumulated aliases
                   aliases = existing_result['aliases']
                   result['aliases'] = aliases
                   deduplicated[existing_idx] = result
       
       # Log deduplication stats
       if original_count != len(deduplicated):
           self.logger.info(f"Deduplication: {original_count} -> {len(deduplicated)} results")
           
           # Log which titles were merged
           for result in deduplicated:
               if len(result['aliases']) > 1:
                   self.logger.debug(f"Merged aliases: {', '.join(result['aliases'])}")
       
       return deduplicated
   
   def _normalize_title(self, title: str) -> str:
       """Normalize page title for deduplication comparison."""
       # Convert to lowercase and replace common variations
       normalized = title.lower()
       
       # Remove/normalize common punctuation and spacing
       replacements = {
           '-': '',
           '_': '',
           ' ': '',
           'wi-fi': 'wifi',
           'wi_fi': 'wifi',
       }
       
       for old, new in replacements.items():
           normalized = normalized.replace(old, new)
       
       return normalized
   
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
       """Pretty print search results with alias information."""
       if show_queries and results:
           print(f"\nOriginal query: {results[0]['original_query']}")
           if results[0]['original_query'] != results[0]['final_query']:
               print(f"Refined query:  {results[0]['final_query']}")
       
       for result in results:
           print(f"\n{'='*60}")
           print(f"Rank {result['rank']} | Score: {result['score']:.4f}")
           
           # Show primary title and aliases if any
           if len(result.get('aliases', [])) > 1:
               primary_title = result['page_title']
               other_aliases = [alias for alias in result['aliases'] if alias != primary_title]
               print(f"Page: {primary_title}")
               print(f"Aliases: {', '.join(other_aliases)}")
           else:
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
