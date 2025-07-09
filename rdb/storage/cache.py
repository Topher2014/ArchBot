"""
Cache manager for temporary storage and performance optimization.
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

from ..config.settings import Config
from ..utils.logging import get_logger


class CacheManager:
   """Manages caching for performance optimization."""
   
   def __init__(self, config: Config):
       """Initialize cache manager with configuration."""
       self.config = config
       self.logger = get_logger(__name__)
       self.cache_dir = config.cache_dir
       
       # Ensure cache directories exist
       self.embeddings_cache = self.cache_dir / "embeddings"
       self.queries_cache = self.cache_dir / "queries"
       self.pages_cache = self.cache_dir / "pages"
       
       for cache_path in [self.embeddings_cache, self.queries_cache, self.pages_cache]:
           cache_path.mkdir(parents=True, exist_ok=True)
   
   def _get_cache_key(self, data: Any) -> str:
       """Generate cache key from data."""
       if isinstance(data, str):
           content = data
       else:
           content = json.dumps(data, sort_keys=True)
       
       return hashlib.md5(content.encode()).hexdigest()
   
   def _is_cache_valid(self, cache_file: Path, max_age_hours: int = 24) -> bool:
       """Check if cache file is still valid."""
       if not cache_file.exists():
           return False
       
       file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
       return file_age < timedelta(hours=max_age_hours)
   
   def cache_embedding(self, text: str, embedding: Any, model_name: str) -> None:
       """Cache an embedding for a text."""
       cache_key = self._get_cache_key(f"{model_name}:{text}")
       cache_file = self.embeddings_cache / f"{cache_key}.pkl"
       
       try:
           with open(cache_file, 'wb') as f:
               pickle.dump({
                   'text': text,
                   'embedding': embedding,
                   'model_name': model_name,
                   'timestamp': datetime.now()
               }, f)
       except Exception as e:
           self.logger.warning(f"Failed to cache embedding: {e}")
   
   def get_cached_embedding(self, text: str, model_name: str, max_age_hours: int = 168) -> Optional[Any]:
       """Get cached embedding for text (default 7 days)."""
       cache_key = self._get_cache_key(f"{model_name}:{text}")
       cache_file = self.embeddings_cache / f"{cache_key}.pkl"
       
       if not self._is_cache_valid(cache_file, max_age_hours):
           return None
       
       try:
           with open(cache_file, 'rb') as f:
               cached_data = pickle.load(f)
               return cached_data['embedding']
       except Exception as e:
           self.logger.warning(f"Failed to load cached embedding: {e}")
           return None
   
   def cache_query_refinement(self, original_query: str, refined_query: str, model_name: str) -> None:
       """Cache a query refinement."""
       cache_key = self._get_cache_key(f"{model_name}:{original_query}")
       cache_file = self.queries_cache / f"{cache_key}.json"
       
       try:
           with open(cache_file, 'w', encoding='utf-8') as f:
               json.dump({
                   'original_query': original_query,
                   'refined_query': refined_query,
                   'model_name': model_name,
                   'timestamp': datetime.now().isoformat()
               }, f, indent=2)
       except Exception as e:
           self.logger.warning(f"Failed to cache query refinement: {e}")
   
   def get_cached_query_refinement(self, original_query: str, model_name: str, 
                                  max_age_hours: int = 24) -> Optional[str]:
       """Get cached query refinement."""
       cache_key = self._get_cache_key(f"{model_name}:{original_query}")
       cache_file = self.queries_cache / f"{cache_key}.json"
       
       if not self._is_cache_valid(cache_file, max_age_hours):
           return None
       
       try:
           with open(cache_file, 'r', encoding='utf-8') as f:
               cached_data = json.load(f)
               return cached_data['refined_query']
       except Exception as e:
           self.logger.warning(f"Failed to load cached query refinement: {e}")
           return None
   
   def cache_page_content(self, url: str, content: Dict[str, Any]) -> None:
       """Cache scraped page content."""
       cache_key = self._get_cache_key(url)
       cache_file = self.pages_cache / f"{cache_key}.json"
       
       try:
           cache_data = {
               'url': url,
               'content': content,
               'timestamp': datetime.now().isoformat()
           }
           
           with open(cache_file, 'w', encoding='utf-8') as f:
               json.dump(cache_data, f, indent=2, ensure_ascii=False)
       except Exception as e:
           self.logger.warning(f"Failed to cache page content: {e}")
   
   def get_cached_page_content(self, url: str, max_age_hours: int = 168) -> Optional[Dict[str, Any]]:
       """Get cached page content (default 7 days)."""
       cache_key = self._get_cache_key(url)
       cache_file = self.pages_cache / f"{cache_key}.json"
       
       if not self._is_cache_valid(cache_file, max_age_hours):
           return None
       
       try:
           with open(cache_file, 'r', encoding='utf-8') as f:
               cached_data = json.load(f)
               return cached_data['content']
       except Exception as e:
           self.logger.warning(f"Failed to load cached page content: {e}")
           return None
   
   def clear_cache(self, cache_type: Optional[str] = None) -> int:
       """Clear cache files. If cache_type is None, clear all caches."""
       cleared_count = 0
       
       cache_dirs = []
       if cache_type is None:
           cache_dirs = [self.embeddings_cache, self.queries_cache, self.pages_cache]
       elif cache_type == "embeddings":
           cache_dirs = [self.embeddings_cache]
       elif cache_type == "queries":
           cache_dirs = [self.queries_cache]
       elif cache_type == "pages":
           cache_dirs = [self.pages_cache]
       else:
           self.logger.warning(f"Unknown cache type: {cache_type}")
           return 0
       
       for cache_dir in cache_dirs:
           for cache_file in cache_dir.iterdir():
               if cache_file.is_file():
                   try:
                       cache_file.unlink()
                       cleared_count += 1
                   except Exception as e:
                       self.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
       
       self.logger.info(f"Cleared {cleared_count} cache files")
       return cleared_count
   
   def get_cache_stats(self) -> Dict[str, Any]:
       """Get cache statistics."""
       stats = {}
       
       for cache_name, cache_dir in [
           ("embeddings", self.embeddings_cache),
           ("queries", self.queries_cache),
           ("pages", self.pages_cache)
       ]:
           cache_files = list(cache_dir.iterdir())
           total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
           
           stats[cache_name] = {
               "file_count": len(cache_files),
               "total_size_mb": round(total_size / (1024 * 1024), 2),
               "directory": str(cache_dir)
           }
       
       return stats
   
   def cleanup_expired_cache(self, max_age_hours: int = 168) -> int:
       """Remove expired cache files (default 7 days)."""
       cleaned_count = 0
       
       for cache_dir in [self.embeddings_cache, self.queries_cache, self.pages_cache]:
           for cache_file in cache_dir.iterdir():
               if cache_file.is_file() and not self._is_cache_valid(cache_file, max_age_hours):
                   try:
                       cache_file.unlink()
                       cleaned_count += 1
                   except Exception as e:
                       self.logger.warning(f"Failed to delete expired cache file {cache_file}: {e}")
       
       self.logger.info(f"Cleaned up {cleaned_count} expired cache files")
       return cleaned_count
