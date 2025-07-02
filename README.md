# RDB - Retrieval Database

A powerful retrieval-augmented generation (RAG) system for Arch Linux documentation. RDB scrapes, processes, embeds, and enables semantic search through the Arch Wiki with optional LLM-powered query refinement.

## Features

- **Multi-source scraping**: Extract content from Arch Wiki with structured parsing
- **Smart chunking**: Multi-level chunking (small/medium/large) for optimal retrieval
- **Semantic search**: Vector embeddings using E5-Large-V2 for accurate content matching
- **Query refinement**: LLM-powered query expansion for better search results
- **Fast retrieval**: FAISS-based vector indexing for millisecond search times
- **CLI interface**: Easy-to-use command-line tools for all operations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Topher2014/rdb.git
cd rdb

# Install in development mode
pip install -e .

# Or install with development tools
pip install -e ".[dev]"
```

### Basic Usage

```bash
# 1. Scrape Arch Wiki
rdb scrape --output data/raw/

# 2. Process and chunk content
rdb build --input data/raw/ --output data/index/

# 3. Search documentation
rdb search "wifi connection issues"

# 4. Interactive search mode
rdb search --interactive
```

## Architecture

```
rdb/
├── rdb/                    # Core package
│   ├── scraper/           # Web scraping and content extraction
│   ├── chunking/          # Document processing and chunking
│   ├── embedding/         # Vector embeddings and indexing
│   ├── retrieval/         # Search and query refinement
│   ├── storage/           # Data persistence and caching
│   ├── config/            # Configuration management
│   └── utils/             # Common utilities
├── cli/                   # Command-line interface
├── tests/                 # Test suite
├── data/                  # Default data directory
└── examples/              # Usage examples
```

## Configuration

RDB uses environment variables and configuration files for customization:

```bash
# Set custom data directory
export RDB_DATA_DIR="/path/to/data"

# Use GPU for embeddings (if available)
export RDB_USE_GPU=true

# Custom embedding model
export RDB_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Lint code
flake8
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rdb

# Run specific test file
pytest tests/test_scraper.py
```

## API Usage

```python
from rdb import RDB

# Initialize RDB
rdb = RDB(data_dir="./data")

# Build index from scraped data
rdb.build_index("./data/raw")

# Search
results = rdb.search("network configuration", top_k=5)

# Search with query refinement
results = rdb.search("wifi broken", refine_query=True, top_k=5)
```

## Advanced Features

### Custom Chunking Strategies

```python
from rdb.chunking import ChunkingStrategy

class CustomStrategy(ChunkingStrategy):
    def chunk(self, document):
        # Custom chunking logic
        pass
```

### Custom Embedding Models

```python
from rdb.embedding import EmbeddingModel

# Use custom model
embedder = EmbeddingModel("your-custom-model")
```

## Performance

- **Indexing**: ~1000 documents per minute
- **Search**: <100ms for typical queries
- **Memory usage**: ~2GB for full Arch Wiki index
- **Storage**: ~500MB for embeddings + metadata

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Arch Linux](https://archlinux.org/) for the excellent documentation
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search