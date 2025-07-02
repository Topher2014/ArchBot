# Arch Wiki Retrieval

Semantic search for Arch Wiki documentation using E5-Large-V2 embeddings.

## Setup

```bash
pip install sentence-transformers faiss-cpu numpy tqdm
```

## Usage

### Create New Index
```bash
# Basic usage (saves to current directory)
python arch_wiki_embedder.py arch_chunks.json

# Save to specific directory
python arch_wiki_embedder.py arch_chunks.json -o ./indexes/
```

### Load Existing Index
```bash
# Load from specific directory
python arch_wiki_embedder.py -l ./indexes/

# Load from different location
python arch_wiki_embedder.py -l ./models/arch_vectors/
```

### Command Format
```bash
python arch_wiki_embedder.py [json_file] [-o output_dir] [-l load_dir]
```

- `json_file`: Path to your arch_chunks.json (only needed when creating)
- `-o, --output`: Directory to save NEW index files
- `-l, --load`: Directory to load EXISTING index files from

## Multiple Vector Sets

```bash
# Create different vector sets
python arch_wiki_embedder.py arch_chunks.json -o ./vectors/v1/
python arch_wiki_embedder.py arch_updated.json -o ./vectors/v2/

# Switch between them
python arch_wiki_embedder.py -l ./vectors/v1/    # Use version 1
python arch_wiki_embedder.py -l ./vectors/v2/    # Use version 2
```

## Files Created

- `arch_wiki_index.faiss` - FAISS search index
- `arch_wiki_metadata.pkl` - Document chunks and metadata

## Examples

```bash
# Create new index
python arch_wiki_embedder.py ./arch_chunks.json -o ./models/

# Later, load that index for searching
python arch_wiki_embedder.py -l ./models/

# Or load from a different location
python arch_wiki_embedder.py -l ./backup_vectors/
```
