"""
Embedding model management for RDB.
"""

from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import numpy as np

from ..utils.logging import get_logger


class EmbeddingModel:
   """Wrapper for sentence transformer models."""
   
   def __init__(self, model_name: str = 'intfloat/e5-large-v2', device: str = 'cpu'):
       """Initialize embedding model."""
       self.model_name = model_name
       self.device = device
       self.logger = get_logger(__name__)
       
       self.logger.info(f"Loading embedding model: {model_name}")
       self.logger.info(f"Device: {device}")
       
       # Load model
       self.model = SentenceTransformer(model_name, device=device)
       
       # Get model info
       self.dimension = self.model.get_sentence_embedding_dimension()
       self.max_seq_length = self.model.max_seq_length
       
       self.logger.info(f"Model loaded successfully!")
       self.logger.info(f"Embedding dimension: {self.dimension}")
       self.logger.info(f"Max sequence length: {self.max_seq_length}")
   
   def encode(self, texts: Union[str, List[str]], batch_size: int = 32, 
              show_progress_bar: bool = False, normalize_embeddings: bool = False) -> np.ndarray:
       """Encode texts into embeddings."""
       if isinstance(texts, str):
           texts = [texts]
       
       embeddings = self.model.encode(
           texts,
           batch_size=batch_size,
           show_progress_bar=show_progress_bar,
           normalize_embeddings=normalize_embeddings,
           convert_to_numpy=True
       )
       
       return embeddings
   
   def encode_query(self, query: str) -> np.ndarray:
       """Encode a single query with proper prefix for e5 models."""
       if self.model_name.startswith('intfloat/e5'):
           # E5 models require "query: " prefix for queries
           query_text = f"query: {query}"
       else:
           query_text = query
       
       return self.encode([query_text])[0]
   
   def encode_passage(self, passage: str) -> np.ndarray:
       """Encode a single passage with proper prefix for e5 models."""
       if self.model_name.startswith('intfloat/e5'):
           # E5 models require "passage: " prefix for passages
           passage_text = f"passage: {passage}"
       else:
           passage_text = passage
       
       return self.encode([passage_text])[0]
   
   def get_info(self) -> dict:
       """Get model information."""
       return {
           'model_name': self.model_name,
           'device': self.device,
           'dimension': self.dimension,
           'max_seq_length': self.max_seq_length,
           'is_cuda_available': torch.cuda.is_available(),
           'current_device': str(self.model.device)
       }
