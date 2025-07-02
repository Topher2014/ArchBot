"""
RDB - Retrieval Database for Arch Wiki Documentation

A powerful retrieval-augmented generation (RAG) system for semantic search
through Arch Linux documentation.
"""

__version__ = "0.1.0"
__author__ = "Topher Ludlow"
__email__ = "topherludlow@protonmail.com"

from .config.settings import Config
from .scraper.wiki_scraper import WikiScraper
from .chunking.chunker import DocumentChunker
from .embedding.embedder import DocumentEmbedder
from .retrieval.retriever import DocumentRetriever

class RDB:
   """Main RDB interface for document retrieval operations."""
   
   def __init__(self, data_dir=None, config=None):
       """Initialize RDB with optional data directory and config."""
       self.config = config or Config(data_dir=data_dir)
       self.scraper = None
       self.chunker = None
       self.embedder = None
       self.retriever = None
   
   def get_scraper(self):
       """Get or create scraper instance."""
       if self.scraper is None:
           self.scraper = WikiScraper(self.config)
       return self.scraper
   
   def get_chunker(self):
       """Get or create chunker instance."""
       if self.chunker is None:
           self.chunker = DocumentChunker(self.config)
       return self.chunker
   
   def get_embedder(self):
       """Get or create embedder instance."""
       if self.embedder is None:
           self.embedder = DocumentEmbedder(self.config)
       return self.embedder
   
   def get_retriever(self):
       """Get or create retriever instance."""
       if self.retriever is None:
           self.retriever = DocumentRetriever(self.config)
       return self.retriever
   
   def scrape(self, output_dir=None):
       """Scrape Arch Wiki documentation."""
       scraper = self.get_scraper()
       return scraper.scrape_all(output_dir)
   
   def build_index(self, input_dir=None, output_dir=None):
       """Build search index from scraped documents."""
       chunker = self.get_chunker()
       embedder = self.get_embedder()
       
       # Process documents into chunks
       chunks = chunker.process_directory(input_dir)
       
       # Create embeddings and build index
       embedder.create_embeddings(chunks)
       embedder.save_index(output_dir)
       
       return len(chunks)
   
   def search(self, query, top_k=5, refine_query=False):
       """Search documents with optional query refinement."""
       retriever = self.get_retriever()
       return retriever.search(query, top_k=top_k, refine_query=refine_query)

__all__ = [
   "RDB",
   "Config",
   "WikiScraper", 
   "DocumentChunker",
   "DocumentEmbedder",
   "DocumentRetriever",
   "__version__",
   "__author__",
   "__email__",
]
