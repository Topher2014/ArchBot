# Arch Wiki Retriever

Fast semantic search through your Arch Wiki embeddings.

## Quick Start

```bash
# Interactive search mode
python retriever.py

# Single query
python retriever.py -q "install graphics drivers"
```

## Requirements

```bash
pip install sentence-transformers faiss-cpu numpy
```

## Usage

```bash
python retriever.py [options]
```

### Options

- `-i, --index-dir DIR` - Directory with index files (default: current directory)
- `-n, --name NAME` - Base name for index files (default: arch_wiki)
- `-k, --top-k NUM` - Number of results to show (default: 5)
- `-q, --query TEXT` - Single query (non-interactive mode)
- `-m, --model MODEL` - Sentence transformer model (default: intfloat/e5-large-v2)
- `--max-content NUM` - Max content length to display (default: 300)

### Examples

```bash
# Interactive mode (recommended)
python retriever.py

# Search specific index directory
python retriever.py -i ./my_embeddings

# Single query with more results
python retriever.py -q "bluetooth setup" -k 10

# Custom index name
python retriever.py -i ./custom -n my_wiki

# Show more content per result
python retriever.py --max-content 500
```

## Interactive Mode

In interactive mode, you can:

- **Search**: Type any query and press Enter
- **Help**: Type `help` for commands
- **Stats**: Type `stats` to see index information
- **Quit**: Type `quit`, `exit`, or `q` to exit
- **Ctrl+C**: Also exits the program

### Sample Session

```
Query: install arch linux
# Shows top 5 results with installation guides

Query: wifi not working
# Shows networking troubleshooting docs

Query: stats
Index Statistics:
  Total vectors: 1,247
  Total chunks: 1,247
  Vector dimension: 1024

Query: quit
Goodbye!
```

## Search Tips

- **Natural language**: "How do I set up WiFi?" works great
- **Keywords**: "bluetooth audio" or "graphics drivers"
- **Specific issues**: "black screen after update"
- **Commands**: "pacman install" or "systemctl enable"

## Result Format

Each result shows:
- **Rank & Score**: Relevance ranking and similarity score
- **Page**: Arch Wiki page title
- **Section**: Specific section within the page
- **Type**: Content type (section, intro, etc.)
- **URL**: Direct link to the wiki page
- **Content**: Relevant text excerpt

## Prerequisites

You must first create embeddings using the embedder:

```bash
python embedder.py arch_chunks.json
```

This creates the required index files that the retriever searches through.
