# Arch Wiki Retrieval

Semantic search for Arch Wiki documentation using E5-Large-V2 embeddings.

## Setup

```bash
pip install sentence-transformers faiss-cpu numpy tqdm
```

## Usage

### First Run (Create Index)
```bash
python arch_wiki_embedder.py path/to/arch_chunks.json
```

This will:
- Load your Arch Wiki chunks
- Create embeddings using E5-Large-V2
- Build a FAISS search index
- Save everything for future use

### Subsequent Runs (Use Existing Index)
```bash
python arch_wiki_embedder.py
```

The script will detect existing index files and ask if you want to reuse them.

## Search

After the index is built, you can search interactively:

```
Query: how to configure wifi
Query: systemd service not starting
Query: quit
```

## Files Created

- `arch_wiki_index.faiss` - FAISS search index
- `arch_wiki_metadata.pkl` - Document chunks and metadata

## Example

```bash
# Initial setup
python arch_wiki_embedder.py ./arch_chunks.json

# Search
Query: pacman package conflicts
```
