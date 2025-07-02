#!/usr/bin/env python3
"""
Query Refiner using local LLMs
Expands user queries into technical search terms for better retrieval
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class QueryRefiner:
    def __init__(self, model_path=None, device='auto'):
        """
        Initialize the query refiner
        
        Args:
            model_path: Path to local model or HuggingFace model name
            device: 'auto', 'cpu', 'cuda', or specific device
        """
        if model_path is None:
            # Default models in order of preference
            default_models = [
                "../../Models/Llama/Meta-Llama-3.1-8B",
                "../../Models/Phi-3-mini-4k-instruct", 
                "microsoft/Phi-3-mini-4k-instruct",
                "meta-llama/Llama-3.1-8B-Instruct"
            ]
            
            for model in default_models:
                if os.path.exists(model) or not model.startswith("../"):
                    model_path = model
                    break
            
            if model_path is None:
                raise ValueError("No model found. Please specify model_path or download a model.")
        
        self.model_path = model_path
        self.device = self._get_device(device)
        
        print(f"Loading refiner model: {model_path}")
        print(f"Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
            device_map=self.device if self.device != 'cpu' else None,
            trust_remote_code=True
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Refiner model loaded successfully!")
    
    def _get_device(self, device):
        """Determine the best device to use"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def refine_query(self, user_query, max_new_tokens=30, temperature=0.7):
        """
        Refine a user query into technical search terms
        
        Args:
            user_query: Original user query
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more focused)
        
        Returns:
            str: Refined query with technical terms
        """
        # Create prompt for query refinement
        prompt = self._create_refinement_prompt(user_query)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        if self.device != 'cpu':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the refined query (remove the prompt)
        refined_query = full_response[len(prompt):].strip()
        
        # Clean up the response
        refined_query = self._clean_response(refined_query)
        
        return refined_query
    
    def _create_refinement_prompt(self, user_query):
        """Create a prompt for query refinement"""
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
    
    def _clean_response(self, response):
        """Clean up the model response"""
        # Remove common unwanted prefixes/suffixes
        response = response.strip()
        
        # Remove quotes if the model added them
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Remove any trailing explanations after newlines
        if '\n' in response:
            response = response.split('\n')[0]
        
        # Remove common prefixes the model might add
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
        
        # Limit length to reasonable query size
        if len(response) > 200:
            response = response[:200].strip()
        
        return response
    
    def test_refinement(self, test_queries=None):
        """Test the refiner with sample queries"""
        if test_queries is None:
            test_queries = [
                "wifi broken",
                "I just installed arch, how do I connect to internet?",
                "sound not working",
                "my graphics are slow",
                "package manager won't install anything",
                "boot hangs on black screen"
            ]
        
        print("\n" + "="*60)
        print("QUERY REFINEMENT TEST")
        print("="*60)
        
        for query in test_queries:
            print(f"\nOriginal: {query}")
            refined = self.refine_query(query)
            print(f"Refined:  {refined}")
            print("-" * 40)


def main():
    """Test the query refiner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test query refinement')
    parser.add_argument('-m', '--model', 
                       help='Model path or HuggingFace model name')
    parser.add_argument('-q', '--query',
                       help='Single query to refine')
    parser.add_argument('-t', '--test', action='store_true',
                       help='Run test with sample queries')
    parser.add_argument('-d', '--device', default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    try:
        refiner = QueryRefiner(model_path=args.model, device=args.device)
        
        if args.query:
            print(f"Original: {args.query}")
            refined = refiner.refine_query(args.query)
            print(f"Refined:  {refined}")
        elif args.test:
            refiner.test_refinement()
        else:
            # Interactive mode
            print("\nInteractive Query Refinement")
            print("Enter queries to refine (or 'quit' to exit)")
            
            while True:
                query = input("\nQuery: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    refined = refiner.refine_query(query)
                    print(f"Refined: {refined}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
