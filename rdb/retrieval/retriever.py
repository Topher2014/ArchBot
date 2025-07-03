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
        
        # Enhanced scoring logic
        query_words = set(query.lower().split())
        original_query_words = set(original_query.lower().split())
        
        for result in results:
            base_score = result['score']
            page_title = result['page_title'].lower()
            section_path = result['section_path'].lower()
            content = result['content'].lower()
            
            # 1. AUTHORITATIVE GUIDE DETECTION
            # Special boost for Network_configuration/* pages
            if 'network_configuration' in page_title or 'network configuration' in page_title:
                result['score'] *= 1.5  # Strong boost for main config pages
            
            # Detect other comprehensive configuration/setup guides
            elif any(pattern in page_title for pattern in [
                'installation_guide', 'installation guide', 'system administration',
                'getting started', 'beginners guide', 'general configuration'
            ]):
                result['score'] *= 1.3
            
            # 2. TOPIC RELEVANCE BOOSTING
            # Map common user queries to technical topics
            topic_mappings = {
                'wifi': ['wireless', 'wi-fi', 'network', 'iwctl', 'networkmanager'],
                'wireless': ['wi-fi', 'network', 'iwctl', 'networkmanager'],
                'internet': ['network', 'connection', 'dhcp'],
                'connect': ['connection', 'setup', 'configuration'],
                'setup': ['configuration', 'install', 'guide'],
                'install': ['installation', 'setup', 'guide'],
                'sound': ['audio', 'alsa', 'pulseaudio'],
                'audio': ['sound', 'alsa', 'pulseaudio'],
                'graphics': ['video', 'gpu', 'driver', 'xorg'],
                'boot': ['grub', 'systemd', 'bootloader'],
                'package': ['pacman', 'aur', 'makepkg'],
            }
            
            # Check if query concepts match page content
            for query_word in query_words.union(original_query_words):
                if query_word in topic_mappings:
                    related_terms = topic_mappings[query_word]
                    if any(term in page_title or term in content[:500] for term in related_terms):
                        result['score'] *= 1.2
                        break
            
            # 3. COMPREHENSIVENESS INDICATORS
            comprehensiveness_indicators = [
                'configuration', 'setup', 'installation', 'troubleshooting',
                'overview', 'guide', 'manual', 'documentation'
            ]
            
            comprehensiveness_score = sum(1 for indicator in comprehensiveness_indicators 
                                        if indicator in page_title or indicator in section_path)
            
            if comprehensiveness_score >= 2:
                result['score'] *= 1.25  # Boost comprehensive-looking pages
            
            # 4. CONTENT QUALITY INDICATORS
            # Longer content often indicates more comprehensive coverage
            if len(result['content']) > 1000:
                result['score'] *= 1.15
            elif len(result['content']) > 500:
                result['score'] *= 1.1
                
            # Content with multiple configuration methods/options
            config_indicators = ['install', 'configure', 'setup', 'enable', 'edit', 'create']
            config_count = sum(1 for indicator in config_indicators if indicator in content)
            if config_count >= 3:
                result['score'] *= 1.1
                
            # Content with code blocks/commands (practical guides)
            if '```' in result['content'] or any(cmd in content for cmd in ['sudo', 'systemctl', 'pacman']):
                result['score'] *= 1.05
            
            # 5. CHUNK TYPE OPTIMIZATION
            # Prefer medium/large chunks for comprehensive queries
            user_wants_overview = any(word in original_query.lower() 
                                    for word in ['how', 'setup', 'configure', 'install', 'guide'])
            
            if user_wants_overview:
                if result['chunk_type'] == 'large':
                    result['score'] *= 1.2
                elif result['chunk_type'] == 'medium':
                    result['score'] *= 1.1
                elif result['chunk_type'] == 'small':
                    result['score'] *= 0.95
            else:
                # For specific queries, prefer small chunks with direct answers
                if result['chunk_type'] == 'small':
                    result['score'] *= 1.1
            
            # 6. ENHANCED EXACT MATCH BOOSTING
            title_words = set(page_title.replace('_', ' ').replace('/', ' ').split())
            query_intersection = query_words.intersection(title_words)
            original_intersection = original_query_words.intersection(title_words)
            
            # Strong boost for exact title matches
            if query_intersection:
                match_ratio = len(query_intersection) / len(query_words) if query_words else 0
                result['score'] *= (1.0 + match_ratio * 0.5)  # Up to 50% boost
                
            if original_intersection:
                match_ratio = len(original_intersection) / len(original_query_words) if original_query_words else 0
                result['score'] *= (1.0 + match_ratio * 0.3)  # Up to 30% boost
        
        # Re-sort by enhanced scores
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Update ranks after re-sorting
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
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
