# Arch Wiki Retrieval with Query Refinement

Semantic search for Arch Wiki documentation using E5-Large-V2 embeddings with optional LLM query enhancement.

## Setup

```bash
# For embeddings
pip install sentence-transformers faiss-cpu numpy tqdm

# For query refinement (optional)
pip install torch transformers accelerate
```

## Components

- **`embedder.py`** - Creates vector embeddings from Arch Wiki chunks
- **`query_refiner.py`** - Expands queries using local LLMs
- **`retriever.py`** - Searches with optional query refinement

## Quick Start

### 1. Create Index
```bash
python embedder.py arch_chunks.json -o ./indexes/
```

### 2. Search with Query Refinement
```bash
python retriever.py -i ./indexes/
```

### 3. Search without Refinement
```bash
python retriever.py -i ./indexes/ --no-refinement
```

## Usage Examples

### Basic Search
```bash
# Interactive mode with refinement
python retriever.py -i ./indexes/

# Single query
python retriever.py -i ./indexes/ -q "wifi broken" --show-refinement
```

### Custom Models
```bash
# Use specific refiner model
python retriever.py -i ./indexes/ -r "../Models/Phi-3-mini-4k-instruct"

# Use different embedding model
python retriever.py -i ./indexes/ -m "intfloat/e5-base-v2"
```

## Interactive Commands

- `wifi broken` - Search for wireless troubleshooting
- `toggle` - Toggle refinement display on/off
- `refine <query>` - Show query refinement without searching
- `stats` - Show index statistics
- `help` - Show available commands
- `quit` - Exit

## Query Refinement Examples

**Without refinement:** "wifi broken" → searches literally  
**With refinement:** "wifi broken" → "wireless network configuration NetworkManager iwctl troubleshooting"

**Without refinement:** "sound not working" → basic search  
**With refinement:** "sound not working" → "audio configuration ALSA PulseAudio sound card driver"

## Options

### Retriever Options
- `-i, --index-dir` - Directory containing index files
- `-k, --top-k` - Number of results to return (default: 5)
- `-r, --refiner-model` - Path to refiner model
- `--no-refinement` - Disable query refinement
- `--show-refinement` - Show query refinement in output

### Files Created by Embedder
- `arch_wiki_index.faiss` - FAISS search index
- `arch_wiki_metadata.pkl` - Document chunks and metadata
