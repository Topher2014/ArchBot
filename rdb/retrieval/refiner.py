"""
Query refiner using local LLMs for better search terms.
"""

import os
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
       """Find default model from common locations."""
       default_models = [
           "../../Models/Llama/Meta-Llama-3.1-8B",
           "../../Models/Phi-3-mini-4k-instruct", 
           "microsoft/Phi-3-mini-4k-instruct",
           "meta-llama/Llama-3.1-8B-Instruct"
       ]
       
       for model in default_models:
           if os.path.exists(model) or not model.startswith("../"):
               return model
       
       return None
   
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
       """Create prompt for query refinement."""
       prompt = f"""You are an expert system for Arch Linux documentation search. Expand user queries into technical terms.

Examples:
User query: wifi broken
Technical search query: wireless network configuration NetworkManager iwctl troubleshooting connection issues

User query: sound not working
Technical search query: audio configuration ALSA PulseAudio sound card driver troubleshooting

User query: package won't install
Technical search query: pacman package manager installation dependencies conflicts

User query: {user_query}
Technical search query:"""
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
