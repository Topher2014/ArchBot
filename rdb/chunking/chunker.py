"""
Document chunker for RDB.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config.settings import Config
from ..utils.logging import get_logger
from .models import Chunk
from .strategies import SmallChunkStrategy, MediumChunkStrategy, LargeChunkStrategy


class DocumentChunker:
    """Creates multi-level chunks from scraped documents."""
    
    def __init__(self, config: Config):
        """Initialize document chunker with configuration."""
        self.config = config
        self.logger = get_logger(__name__)
        self.chunks: List[Chunk] = []
        
        # Initialize chunking strategies
        self.small_strategy = SmallChunkStrategy(config)
        self.medium_strategy = MediumChunkStrategy(config)
        self.large_strategy = LargeChunkStrategy(config)
    
    def process_directory(self, input_dir: Optional[str] = None) -> List[Chunk]:
        """Process all JSON files in a directory."""
        if input_dir is None:
            input_dir = self.config.raw_data_dir
        else:
            input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        json_files = list(input_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {input_dir}")
        
        self.logger.info(f"Processing {len(json_files)} JSON files...")
        
        self.chunks = []
        for json_file in json_files:
            if json_file.name == "page_list.json":
                continue  # Skip the page list file
                
            self.logger.debug(f"Processing: {json_file.name}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                self._process_document(doc)
            except Exception as e:
                self.logger.error(f"Error processing {json_file}: {e}")
                continue
        
        self.logger.info(f"Created {len(self.chunks)} total chunks")
        return self.chunks
    
    def _process_document(self, doc: Dict[str, Any]) -> None:
        """Process a single document and create all chunk levels."""
        page_title = doc.get('title', 'Unknown')
        url = doc.get('url', '')
        sections = doc.get('sections', [])
        
        if not sections:
            self.logger.warning(f"No sections found in document: {page_title}")
            return
        
        # Create chunks using different strategies
        try:
            large_chunks = self.large_strategy.create_chunks(page_title, url, sections)
            medium_chunks = self.medium_strategy.create_chunks(page_title, url, sections)
            small_chunks = self.small_strategy.create_chunks(page_title, url, sections)
            
            # Add all chunks to the collection
            self.chunks.extend(large_chunks)
            self.chunks.extend(medium_chunks)
            self.chunks.extend(small_chunks)
            
        except Exception as e:
            self.logger.error(f"Error chunking document {page_title}: {e}")
    
    def save_chunks(self, output_file: Optional[str] = None) -> None:
        """Save chunks to JSON file."""
        if output_file is None:
            output_file = self.config.chunks_file
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                'page_title': chunk.page_title,
                'section_path': chunk.section_path,
                'content': chunk.content,
                'chunk_text': chunk.chunk_text,
                'url': chunk.url,
                'chunk_type': chunk.chunk_type,
                'section_level': chunk.section_level
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(chunks_data)} chunks to {output_file}")
    
    def load_chunks(self, input_file: Optional[str] = None) -> List[Chunk]:
        """Load chunks from JSON file."""
        if input_file is None:
            input_file = self.config.chunks_file
        else:
            input_file = Path(input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = []
        for chunk_data in chunks_data:
            chunk = Chunk(
                page_title=chunk_data['page_title'],
                section_path=chunk_data['section_path'],
                content=chunk_data['content'],
                chunk_text=chunk_data['chunk_text'],
                url=chunk_data['url'],
                chunk_type=chunk_data['chunk_type'],
                section_level=chunk_data['section_level']
            )
            self.chunks.append(chunk)
        
        self.logger.info(f"Loaded {len(self.chunks)} chunks from {input_file}")
        return self.chunks
    
    def get_stats(self) -> Dict[str, int]:
        """Get chunking statistics."""
        stats = {
            'small': len([c for c in self.chunks if c.chunk_type == "small"]),
            'medium': len([c for c in self.chunks if c.chunk_type == "medium"]),
            'large': len([c for c in self.chunks if c.chunk_type == "large"]),
            'total': len(self.chunks)
        }
        return stats
    
    def print_stats(self) -> None:
        """Print chunking statistics."""
        stats = self.get_stats()
        
        self.logger.info("Chunking Statistics:")
        self.logger.info(f"  Small chunks:  {stats['small']}")
        self.logger.info(f"  Medium chunks: {stats['medium']}")
        self.logger.info(f"  Large chunks:  {stats['large']}")
        self.logger.info(f"  Total chunks:  {stats['total']}")
