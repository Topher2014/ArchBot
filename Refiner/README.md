# Query Refiner

Expands user queries into technical search terms using local LLMs for better document retrieval.

## Setup

```bash
pip install torch transformers accelerate
```

## Usage

### Test with Sample Queries
```bash
python query_refiner.py -t
```

### Refine a Single Query
```bash
python query_refiner.py -q "wifi broken"
```

### Interactive Mode
```bash
python query_refiner.py
```

### Specify Model
```bash
# Use specific local model
python query_refiner.py -m "../Models/Phi-3-mini-4k-instruct" -q "sound not working"

# Use HuggingFace model (downloads automatically)
python query_refiner.py -m "microsoft/Phi-3-mini-4k-instruct" -t
```

## Model Auto-Detection

The refiner automatically searches for models in this order:
1. `../Models/Llama/Meta-Llama-3.1-8B`
2. `../Models/Phi-3-mini-4k-instruct`
3. `microsoft/Phi-3-mini-4k-instruct` (downloads from HuggingFace)
4. `meta-llama/Llama-3.1-8B-Instruct` (downloads from HuggingFace)

## Examples

**Input:** "wifi broken"  
**Output:** "wireless network configuration NetworkManager iwctl troubleshooting connection issues"

**Input:** "I just installed arch, how do I connect to internet?"  
**Output:** "Arch Linux post-installation network configuration NetworkManager iwctl wireless setup"

**Input:** "sound not working"  
**Output:** "audio configuration ALSA PulseAudio sound card driver troubleshooting"

## Options

- `-m, --model` - Model path or HuggingFace model name
- `-q, --query` - Single query to refine
- `-t, --test` - Run test with sample queries
- `-d, --device` - Device to use (auto, cpu, cuda)

## Changing Models

Edit the `default_models` list in `query_refiner.py` to change model priority or add new models.
