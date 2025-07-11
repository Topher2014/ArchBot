"""
Query refiner using local LLMs for better search terms.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional

from ..config.settings import Config
from ..utils.logging import get_logger


class QueryRefiner:
    """Refines user queries into technical search terms using local LLMs."""

    def __init__(self, config: Config):
       """Initialize query refiner with configuration."""
       self.config = config
       self.logger = get_logger(__name__)
       
       # Determine model path
       model_path = config.refiner_model
       if model_path is None:
           model_path = self._find_default_model()
       
       if model_path is None:
           raise ValueError("No refiner model found. Set RDB_REFINER_MODEL environment variable.")
       
       self.model_path = model_path
       self.device = self._get_device()
       
       self.logger.info(f"Loading refiner model: {model_path}")
       self.logger.info(f"Device: {self.device}")
       
       # Load tokenizer and model
       self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
       self.model = AutoModelForCausalLM.from_pretrained(
           model_path,
           torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
           device_map=self.device if self.device != 'cpu' else None,
           trust_remote_code=True
       )
       
       # Add pad token if needed
       if self.tokenizer.pad_token is None:
           self.tokenizer.pad_token = self.tokenizer.eos_token
       
       self.logger.info("Refiner model loaded successfully!")
       
    def _find_default_model(self) -> Optional[str]:
        """Find default model from local directory or fall back to remote."""
        # Check project-local models directory first
        models_dir = Path("local/models")
        
        if models_dir.exists():
            valid_models = []
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and self._is_valid_model_dir(model_dir):
                    valid_models.append(model_dir)
            
            if len(valid_models) > 1:
                # Sort for deterministic behavior
                valid_models.sort(key=lambda x: x.name)
                model_names = [m.name for m in valid_models]
                self.logger.warning(f"Multiple models found: {model_names}")
                self.logger.warning(f"Using: {valid_models[0].name}")
                self.logger.info("Set RDB_REFINER_MODEL to specify which model to use")
            
            if valid_models:
                self.logger.info(f"Found local model: {valid_models[0].name}")
                return str(valid_models[0])
        
        # Fall back to remote models
        remote_models = [
            "Qwen/Qwen2.5-1.5B-Instruct"
        ]
        
        self.logger.info(f"No local models found, will download: {remote_models[0]}")
        return remote_models[0]

    def _is_valid_model_dir(self, path: Path) -> bool:
        """Check if directory contains a valid model."""
        # Check for essential model files
        required_files = ["config.json"]
        optional_files = ["tokenizer.json", "tokenizer_config.json", "pytorch_model.bin", "model.safetensors"]
        
        # Must have config.json
        if not (path / "config.json").exists():
            return False
        
        # Must have at least one model file
        has_model_file = any((path / file).exists() for file in optional_files)
        return has_model_file

    def _get_device(self) -> str:
       """Determine the best device to use."""
       if self.config.use_gpu and torch.cuda.is_available():
           return 'cuda'
       return 'cpu'

    def refine_query(self, user_query: str) -> str:
       """Refine a user query into technical search terms."""
       prompt = self._create_refinement_prompt(user_query)
       
       # Tokenize input
       inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
       
       if self.device != 'cpu':
           inputs = {k: v.to(self.device) for k, v in inputs.items()}
       
       # Generate response
       with torch.no_grad():
           outputs = self.model.generate(
               **inputs,
               max_new_tokens=self.config.refiner_max_tokens,
               temperature=self.config.refiner_temperature,
               do_sample=True,
               top_p=0.9,
               repetition_penalty=1.1,
               pad_token_id=self.tokenizer.eos_token_id,
               eos_token_id=self.tokenizer.eos_token_id
           )
       
       # Decode response
       full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
       
       # Extract refined query
       refined_query = full_response[len(prompt):].strip()
       
       # Clean up response
       refined_query = self._clean_response(refined_query)
       
       return refined_query

    def _create_refinement_prompt(self, user_query: str) -> str:
        """Create a more specific prompt for Arch Linux documentation search."""
        prompt = f"""You are an expert Arch Linux system administrator. Convert user questions into specific technical search terms that match Arch Wiki page titles and content.

        IMPORTANT: Include both specific commands AND general page titles in your search terms.

        Examples:
        User: "How do I connect to wifi?"
        Search: "Wireless network configuration iwctl station connect NetworkManager wifi setup"

        User: "wifi broken"  
        Search: "Wireless network configuration troubleshooting iwctl connection NetworkManager"

        User: "sound not working"
        Search: "ALSA sound configuration PulseAudio audio troubleshooting"

        User: "install packages"
        Search: "Pacman package manager installation AUR"

        User: "{user_query}"
        Search:"""
        return prompt

    def _clean_response(self, response: str) -> str:
       """Clean up model response."""
       response = response.strip()
       
       # Remove quotes
       if response.startswith('"') and response.endswith('"'):
           response = response[1:-1]
       
       # Remove trailing explanations
       if '\n' in response:
           response = response.split('\n')[0]
       
       # Remove common prefixes
       prefixes_to_remove = [
           "Technical search query:",
           "Search terms:",
           "Refined query:",
           "Query:"
       ]
       
       for prefix in prefixes_to_remove:
           if response.lower().startswith(prefix.lower()):
               response = response[len(prefix):].strip()
       
       # Remove excessive repetition
       words = response.split()
       seen = set()
       filtered_words = []
       for word in words:
           clean_word = word.strip('",.')
           if clean_word.lower() not in seen:
               seen.add(clean_word.lower())
               filtered_words.append(word)
       
       response = ' '.join(filtered_words)
       
       # Limit length
       if len(response) > 200:
           response = response[:200].strip()
       
       return response
