# Arch Wiki Chunker

Creates 3-level chunks from Arch Wiki JSON files.

## Usage

```bash
python chunker.py <json_dir> [options]
```

## Arguments

- `json_dir` - Directory containing JSON files (required)

## Options

- `--output`, `-o` - Output file (default: `chunks.json`)

## Examples

```bash
# Basic usage
python chunker.py ./arch_wiki_jsons

# Custom output
python chunker.py ./arch_wiki_jsons -o my_chunks.json
```
