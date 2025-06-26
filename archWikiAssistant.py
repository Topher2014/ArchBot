import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse

# Configuration
VECTOR_DIR = "arch_wiki_vectors"  # Where your vectors are stored
LLM_MODEL = "microsoft/phi-2"     # Default LLM model
TOP_K = 8                        # Number of relevant chunks to retrieve

class ArchLinuxAssistant:
    def __init__(self, vector_dir=VECTOR_DIR, llm_model=LLM_MODEL):
        print("Initializing Arch Linux Assistant...")
        self.vector_dir = vector_dir
        
        # Load chunks
        print("Loading knowledge chunks...")
        with open(os.path.join(vector_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded {len(self.chunks)} knowledge chunks")
        
        # Load FAISS index
        print("Loading vector index...")
        self.index = faiss.read_index(os.path.join(vector_dir, "wiki.index"))
        
        # Load embedding model
        print("Loading embedding model...")
        with open(os.path.join(vector_dir, "model_info.json"), "r") as f:
            import json
            model_info = json.load(f)
            embedding_model = model_info.get("model", "all-MiniLM-L6-v2")
        
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load language model
        print(f"Loading language model: {llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to be the same as eos token             
        
        # Configure hardware acceleration
        device_map = "auto"
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            dtype = torch.float16  # Use half precision with GPU
        else:
            print("CUDA is not available, using CPU")
            dtype = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        print("Arch Linux Assistant is ready!")
    
    def retrieve_relevant_content(self, query, top_k=TOP_K):
        """Find the most relevant wiki content for a query"""
        # Create query embedding
        start_time = time.time()
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search in the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get the corresponding chunks
        relevant_chunks = [self.chunks[idx] for idx in indices[0]]
        
        # Format context for the model
        context_parts = []
        seen_sections = set()  # To avoid duplicates
        
        for chunk in relevant_chunks:
            # Create a unique identifier for this section
            section_id = f"{chunk['page_title']}#{chunk['title']}"
            
            # Skip if we've already included this section
            if section_id in seen_sections:
                continue
                
            seen_sections.add(section_id)
            
            # Format the chunk
            section_url = f"{chunk['url']}#{chunk['title'].replace(' ', '_')}"
            context_parts.append(
                f"Source: {section_url}\n"
                f"Title: {chunk['page_title']}\n"
                f"Section: {chunk['title']}\n"
                f"Content: {chunk['text']}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        retrieval_time = time.time() - start_time
        print(f"Retrieved {len(context_parts)} relevant sections in {retrieval_time:.2f} seconds")
        
        return context, retrieval_time
    
    def generate_response(self, query, max_new_tokens=512, temperature=0.8):
        """Generate a response using the LLM with retrieved context"""
        # Retrieve relevant content
        context, retrieval_time = self.retrieve_relevant_content(query)
        
        # Create prompt for the model
        prompt = f"""You are an Arch Linux assistant that helps users by providing accurate information from the Arch Wiki.


Below is information from the Arch Wiki relevant to the user's question:

{context}

Based on the above information from the Arch Wiki, answer the following question as helpfully as possible.
If the information provided doesn't contain a complete answer, provide whatever relevant information is available.
Be specific and include commands when they are mentioned in the context.
Only use commands and instructions that are specifically mentioned in the Arch Wiki content provided.

User question: {query}

Answer:"""             
        
        # Generate response
        print("Generating response...")
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#        inputs = self.tokenizer(prompt, return_tensors="pt")
#        inputs.attention_mask = torch.ones_like(inputs.input_ids)  # Set all tokens as attention-worthy
#        inputs = inputs.to(self.model.device)             

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )              
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response (after the prompt)
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif "User question:" in response and len(response) > len(prompt):
            response = response.split("User question:")[1].split("\n", 1)[1].strip()
        else:
            # Fallback approach
            response = response[len(prompt):].strip()
            
        # Make sure we have a response
        if not response or len(response) < 10:
            response = "I couldn't generate a proper response. Please try asking again or rephrase your question."                                  

        generation_time = time.time() - start_time
        print(f"Generated response in {generation_time:.2f} seconds")
        
        return response, context, retrieval_time, generation_time

def main():
    parser = argparse.ArgumentParser(description="Arch Linux Assistant")
    parser.add_argument("--model", default=LLM_MODEL, help="LLM model to use")
    parser.add_argument("--vector-dir", default=VECTOR_DIR, help="Directory containing vector data")
    args = parser.parse_args()
    
    # Initialize the assistant
    assistant = ArchLinuxAssistant(vector_dir=args.vector_dir, llm_model=args.model)
    
    # Simple command line interface
    print("\n===== Arch Linux Assistant =====")
    print("Ask questions about Arch Linux or type 'exit' to quit")
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        if not query.strip():
            continue
            
        # Generate and display response
        print("\nSearching Arch Wiki...")
        response, context, retrieval_time, generation_time = assistant.generate_response(query)
        
        print("\n" + "="*50)
        print(response)
        print("="*50)
        print(f"\nResponse generated in {retrieval_time+generation_time:.2f} seconds total")
        
        # Optionally show sources
        show_sources = input("\nShow sources? (y/n): ").lower().startswith("y")
        if show_sources:
            sources = []
            for section in context.split("---"):
                if "Source:" in section:
                    source_line = section.split("\n")[0].strip()
                    sources.append(source_line)
            
            print("\nSources:")
            for source in sorted(set(sources)):
                print(f"- {source}")

if __name__ == "__main__":
    main()             
