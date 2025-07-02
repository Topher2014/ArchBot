"""
FAISS index management for RDB.
"""

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..config.settings import Config
from ..utils.logging import get_logger


class IndexManager:
   """Manages FAISS index loading, saving, and searching."""
   
   def __init__(self, config: Config):
       """Initialize index manager with configuration."""
       self.config = config
       self.logger = get_logger(__name__)
       self.index: Optional[faiss.Index] = None
       self.chunks: Optional[List[Dict[str, Any]]] = None
   
   def load_index(self, index_dir: Optional[str] = None) -> bool:
       """Load FAISS index and metadata from files."""
       if index_dir is None:
           index_dir = self.config.index_dir
       else:
           index_dir = Path(index_dir)
       
       index_file = index_dir / "index.faiss"
       metadata_file = index_dir / "metadata.pkl"
       
       if not index_file.exists():
           self.logger.error(f"Index file not found: {index_file}")
           return False
       
       if not metadata_file.exists():
           self.logger.error(f"Metadata file not found: {metadata_file}")
           return False
       
       try:
           self.logger.info(f"Loading index from {index_file}...")
           self.index = faiss.read_index(str(index_file))
           
           self.logger.info(f"Loading metadata from {metadata_file}...")
           with open(metadata_file, 'rb') as f:
               self.chunks = pickle.load(f)
           
           self.logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
           return True
           
       except Exception as e:
           self.logger.error(f"Error loading index: {e}")
           return False
   
   def save_index(self, index: faiss.Index, chunks: List[Dict[str, Any]], 
                  output_dir: Optional[str] = None) -> Tuple[str, str]:
       """Save FAISS index and metadata to files."""
       if output_dir is None:
           output_dir = self.config.index_dir
       else:
           output_dir = Path(output_dir)
       
       output_dir.mkdir(parents=True, exist_ok=True)
       
       index_file = output_dir / "index.faiss"
       metadata_file = output_dir / "metadata.pkl"
       
       try:
           self.logger.info(f"Saving index to {index_file}...")
           faiss.write_index(index, str(index_file))
           
           self.logger.info(f"Saving metadata to {metadata_file}...")
           with open(metadata_file, 'wb') as f:
               pickle.dump(chunks, f)
           
           self.logger.info("Index and metadata saved successfully!")
           return str(index_file), str(metadata_file)
           
       except Exception as e:
           self.logger.error(f"Error saving index: {e}")
           raise
   
   def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
       """Search the index for similar vectors."""
       if not self.is_loaded():
           raise RuntimeError("Index not loaded")
       
       scores, indices = self.index.search(query_embedding, top_k)
       return scores, indices
   
   def is_loaded(self) -> bool:
       """Check if index and metadata are loaded."""
       return self.index is not None and self.chunks is not None
   
   def get_stats(self) -> Dict[str, Any]:
       """Get index statistics."""
       if not self.is_loaded():
           return {"status": "not_loaded"}
       
       chunk_types = {}
       for chunk in self.chunks:
           chunk_type = chunk.get('chunk_type', 'unknown')
           chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
       
       return {
           "status": "loaded",
           "total_vectors": self.index.ntotal,
           "total_chunks": len(self.chunks),
           "vector_dimension": self.index.d,
           "index_type": type(self.index).__name__,
           "chunk_types": chunk_types
       }
   
   def get_chunk(self, index: int) -> Optional[Dict[str, Any]]:
       """Get chunk by index."""
       if not self.is_loaded():
           return None
       
       if 0 <= index < len(self.chunks):
           return self.chunks[index]
       
       return None
   
   def rebuild_index(self, new_chunks: List[Dict[str, Any]], 
                    embeddings: np.ndarray) -> bool:
       """Rebuild index with new chunks and embeddings."""
       try:
           dimension = embeddings.shape[1]
           
           # Create new index
           new_index = faiss.IndexFlatIP(dimension)
           
           # Normalize embeddings
           faiss.normalize_L2(embeddings)
           
           # Add embeddings
           new_index.add(embeddings.astype('float32'))
           
           # Update stored data
           self.index = new_index
           self.chunks = new_chunks
           
           self.logger.info(f"Rebuilt index with {self.index.ntotal} vectors")
           return True
           
       except Exception as e:
           self.logger.error(f"Error rebuilding index: {e}")
           return False
