"""
Document embedder for creating vector representations.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from ..config.settings import Config
from ..utils.logging import get_logger
from ..chunking.chunker import Chunk
from .models import EmbeddingModel


class DocumentEmbedder:
   """Creates embeddings for document chunks and builds search index."""
   
   def __init__(self, config: Config):
       """Initialize document embedder with configuration."""
       self.config = config
       self.logger = get_logger(__name__)
       self.model = EmbeddingModel(config.embedding_model, device=config.device)
       self.chunks: Optional[List[dict]] = None
       self.index: Optional[faiss.Index] = None
   
   def load_chunks(self, chunks_file: Optional[str] = None) -> List[dict]:
       """Load chunks from JSON file."""
       if chunks_file is None:
           chunks_file = self.config.chunks_file
       else:
           chunks_file = Path(chunks_file)
       
       self.logger.info(f"Loading chunks from {chunks_file}...")
       
       with open(chunks_file, 'r', encoding='utf-8') as f:
           self.chunks = json.load(f)
       
       self.logger.info(f"Loaded {len(self.chunks)} chunks")
       return self.chunks
   
   def create_embeddings(self, chunks: Optional[List] = None, batch_size: Optional[int] = None) -> np.ndarray:
       """Create embeddings for all chunks."""
       if chunks is not None:
           # Convert Chunk objects to dict if needed
           if chunks and hasattr(chunks[0], 'chunk_text'):
               self.chunks = [
                   {
                       'page_title': chunk.page_title,
                       'section_path': chunk.section_path,
                       'content': chunk.content,
                       'chunk_text': chunk.chunk_text,
                       'url': chunk.url,
                       'chunk_type': chunk.chunk_type,
                       'section_level': chunk.section_level
                   }
                   for chunk in chunks
               ]
           else:
               self.chunks = chunks
       
       if not self.chunks:
           raise ValueError("No chunks loaded. Call load_chunks() first or provide chunks.")
       
       if batch_size is None:
           batch_size = self.config.embedding_batch_size
       
       self.logger.info("Preparing documents for embedding...")
       
       documents = []
       for chunk in self.chunks:
           # Use chunk_text which has context, add e5's required prefix
           doc_text = f"passage: {chunk['chunk_text']}"
           documents.append(doc_text)
       
       self.logger.info(f"Creating embeddings for {len(documents)} documents...")
       
       # Create embeddings in batches
       all_embeddings = []
       for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
           batch = documents[i:i + batch_size]
           batch_embeddings = self.model.encode(batch)
           all_embeddings.append(batch_embeddings)
       
       # Combine all embeddings
       embeddings = np.vstack(all_embeddings)
       self.logger.info(f"Created embeddings: shape {embeddings.shape}")
       
       return embeddings
   
   def build_index(self, embeddings: np.ndarray) -> faiss.Index:
       """Build FAISS index from embeddings."""
       self.logger.info("Building FAISS index...")
       
       dimension = embeddings.shape[1]
       self.logger.info(f"Vector dimension: {dimension}")
       
       # Use IndexFlatIP for inner product (cosine similarity after normalization)
       self.index = faiss.IndexFlatIP(dimension)
       
       # Normalize embeddings for cosine similarity
       self.logger.info("Normalizing embeddings...")
       faiss.normalize_L2(embeddings)
       
       # Add to index
       self.logger.info("Adding embeddings to index...")
       self.index.add(embeddings.astype('float32'))
       
       self.logger.info(f"Index built with {self.index.ntotal} vectors")
       return self.index
   
   def save_index(self, output_dir: Optional[str] = None) -> tuple:
       """Save index and metadata to files."""
       if output_dir is None:
           output_dir = self.config.index_dir
       else:
           output_dir = Path(output_dir)
       
       output_dir.mkdir(parents=True, exist_ok=True)
       
       index_file = output_dir / "index.faiss"
       metadata_file = output_dir / "metadata.pkl"
       
       self.logger.info(f"Saving index to {index_file}...")
       faiss.write_index(self.index, str(index_file))
       
       self.logger.info(f"Saving metadata to {metadata_file}...")
       with open(metadata_file, 'wb') as f:
           pickle.dump(self.chunks, f)
       
       self.logger.info("Index and metadata saved!")
       return str(index_file), str(metadata_file)
   
   def create_and_save_index(self, chunks_file: Optional[str] = None, 
                            output_dir: Optional[str] = None) -> tuple:
       """Complete pipeline: load chunks, create embeddings, build index, save."""
       # Load chunks
       self.load_chunks(chunks_file)
       
       # Create embeddings
       embeddings = self.create_embeddings()
       
       # Build index
       self.build_index(embeddings)
       
       # Save index
       return self.save_index(output_dir)
