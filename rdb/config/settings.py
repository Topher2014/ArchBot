"""
Configuration management for RDB.
"""

import os
import torch
from pathlib import Path
from typing import Optional


class Config:
   """Configuration settings for RDB."""
   
   def __init__(self, data_dir: Optional[str] = None):
       """Initialize configuration with optional data directory."""
       # Base directories
       self.project_root = Path(__file__).parent.parent.parent
       self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
       
       # Ensure data directories exist
       self.raw_data_dir = self.data_dir / "raw"
       self.chunks_dir = self.data_dir / "chunks" 
       self.index_dir = self.data_dir / "index"
       self.cache_dir = self.data_dir / "cache"
       
       # Create directories
       for dir_path in [self.raw_data_dir, self.chunks_dir, self.index_dir, self.cache_dir]:
           dir_path.mkdir(parents=True, exist_ok=True)
       
       # Scraping settings
       self.scrape_delay_min = float(os.getenv("RDB_SCRAPE_DELAY_MIN", "1.0"))
       self.scrape_delay_max = float(os.getenv("RDB_SCRAPE_DELAY_MAX", "3.0"))
       self.scrape_max_retries = int(os.getenv("RDB_SCRAPE_MAX_RETRIES", "3"))
       
       # Chunking settings
       self.chunk_size_small = int(os.getenv("RDB_CHUNK_SIZE_SMALL", "300"))
       self.chunk_size_medium = int(os.getenv("RDB_CHUNK_SIZE_MEDIUM", "800"))
       self.chunk_size_large = int(os.getenv("RDB_CHUNK_SIZE_LARGE", "2000"))
       self.chunk_overlap = int(os.getenv("RDB_CHUNK_OVERLAP", "50"))
       
       # Embedding settings
       self.embedding_model = os.getenv("RDB_EMBEDDING_MODEL", "intfloat/e5-large-v2")
       self.embedding_batch_size = int(os.getenv("RDB_EMBEDDING_BATCH_SIZE", "32"))
       gpu_available = torch.cuda.is_available()
       self.use_gpu = os.getenv("RDB_USE_GPU", str(gpu_available)).lower() == "true"
       self.device = "cuda" if self.use_gpu else "cpu"
       
       # Retrieval settings
       self.default_top_k = int(os.getenv("RDB_DEFAULT_TOP_K", "5"))
       self.enable_query_refinement = os.getenv("RDB_ENABLE_QUERY_REFINEMENT", "true").lower() == "true"
       
       # Query refinement settings
       self.refiner_model = os.getenv("RDB_REFINER_MODEL", None)
       self.refiner_max_tokens = int(os.getenv("RDB_REFINER_MAX_TOKENS", "30"))
       self.refiner_temperature = float(os.getenv("RDB_REFINER_TEMPERATURE", "0.7"))
       
       # File paths
       self.chunks_file = self.chunks_dir / "chunks.json"
       self.index_file = self.index_dir / "index.faiss"
       self.metadata_file = self.index_dir / "metadata.pkl"
       
       # Logging
       self.log_level = os.getenv("RDB_LOG_LEVEL", "INFO")
       self.log_file = self.data_dir / "rdb.log"
       
       # Web scraping
       self.user_agent = os.getenv("RDB_USER_AGENT", 
           "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
       
   def get_cache_path(self, cache_type: str, identifier: str) -> Path:
       """Get cache file path for a specific cache type and identifier."""
       cache_subdir = self.cache_dir / cache_type
       cache_subdir.mkdir(exist_ok=True)
       return cache_subdir / f"{identifier}.cache"
   
   def __repr__(self):
       """String representation of config."""
       return f"Config(data_dir={self.data_dir}, embedding_model={self.embedding_model})"
