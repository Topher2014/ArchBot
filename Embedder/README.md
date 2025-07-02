# Arch Wiki Embedder

Creates searchable embeddings from your Arch Wiki chunks using E5-Large-V2.

## Quick Start

```bash
# Basic usage - creates index in current directory
python embedder.py arch_chunks.json

# Specify output directory
python embedder.py arch_chunks.json -o ./my_index
```

## Requirements

```bash
pip install sentence-transformers faiss-cpu numpy tqdm
```

## Usage

```bash
python embedder.py <json_file> [options]
```

### Options

- `-o, --output DIR` - Output directory for index files (default: current directory)
- `-n, --name NAME` - Base name for output files (default: arch_wiki)
- `-b, --batch-size SIZE` - Batch size for embedding creation (default: 32)
- `-m, --model MODEL` - Sentence transformer model (default: intfloat/e5-large-v2)
- `-f, --force` - Overwrite existing files without asking

### Examples

```bash
# Basic embedding
python embedder.py arch_chunks.json

# Custom output location
python embedder.py arch_chunks.json -o ./embeddings

# Larger batch size for faster processing (uses more RAM)
python embedder.py arch_chunks.json -b 64

# Force overwrite existing files
python embedder.py arch_chunks.json -f

# Different model
python embedder.py arch_chunks.json -m sentence-transformers/all-MiniLM-L6-v2
```

## Output Files

The embedder creates two files:
- `{name}_index.faiss` - The searchable vector index
- `{name}_metadata.pkl` - Document metadata and text content

## Performance Notes

- **Time**: Embedding creation takes several minutes depending on document count
- **CPU-only**: Uses CPU to avoid GPU power draw (configurable in code)
- **Memory**: Adjust batch size based on available RAM
- **Storage**: Index files are typically 100-500MB depending on document count

## Next Steps

After creating embeddings, use the retriever to search:

```bash
python retriever.py -i ./my_index
```
