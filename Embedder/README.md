# Arch Wiki Retrieval

Semantic search for Arch Wiki documentation using E5-Large-V2 embeddings.

## Setup

```bash
pip install sentence-transformers faiss-cpu numpy tqdm
```

## Usage

### Basic Usage
```bash
python arch_wiki_embedder.py arch_chunks.json
```

### Specify Output Directory
```bash
python arch_wiki_embedder.py arch_chunks.json -o ./output/
```

### Command Format
```bash
python arch_wiki_embedder.py [json_file] [-o output_dir]
```

- `json_file`: Path to your arch_chunks.json (default: arch_chunks.json)
- `-o, --output`: Output directory for index files (default: current directory)

## What It Does

1. **First Run**: Creates embeddings and builds search index
2. **Subsequent Runs**: Detects existing index files and asks if you want to reuse them
3. **Interactive Search**: Enter queries to search the Arch Wiki

## Files Created

- `arch_wiki_index.faiss` - FAISS search index
- `arch_wiki_metadata.pkl` - Document chunks and metadata

## Examples

```bash
# Basic usage (saves to current directory)
python arch_wiki_embedder.py ./arch_chunks.json

# Save to specific directory
python arch_wiki_embedder.py ./data/arch_chunks.json -o ./indexes/

# Using default json file with custom output
python arch_wiki_embedder.py -o ./models/
```

## Search

After the index is built:

```
Query: how to configure wifi
Query: systemd service not starting
Query: quit
```
