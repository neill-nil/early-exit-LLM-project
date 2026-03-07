import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
from abc import ABC, abstractmethod

class LLMWrapper(ABC):
    """
    Abstract base class for Large Language Model wrappers.
    This allows us to switch between local HF models and APIs easily.
    """

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        Generate text from the model.
        """
        pass

    @abstractmethod
    def generate_step(self, prompt: str, current_generation: str) -> Dict[str, Any]:
        """
        Generate a single step or a small chunk of tokens.
        Should return the generated text and potentially other info like logits/entropy.
        """
        pass


class HuggingFaceLLMWrapper(LLMWrapper):
    """
    Wrapper for loading open-source models (like Qwen2.5-Math-7B) via Hugging Face Transformers.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading tokenizer {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading model {model_name} on {self.device}...")
        # For Kaggle (15GB VRAM), bfloat16 is usually best for 7B models.
        # If it OOMs, we can switch to load_in_8bit=True
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, 
            device_map="auto" # Automatically places layers on GPU/CPU to fit memory
        )
        self.model.eval()
        
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Standard generation (runs until max_new_tokens or EOS token).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode only the newly generated tokens (skip the prompt)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def generate_step(self, prompt: str, current_generation: str = "", step_tokens_limit: int = 15) -> Dict[str, Any]:
        """
        Generates a small chunk of tokens (a 'step') to allow early-exit controllers to evaluate.
        In a real scenario, returning logits/entropy here is useful for the Learning-Based controller.
        """
        full_context = prompt + current_generation
        inputs = self.tokenizer(full_context, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Generate a small number of tokens (e.g., 15-20)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=step_tokens_limit, # Small step size
                do_sample=False, # Usually greedy decoding for reasoning chunks helps consistency
                output_scores=True, # Need this for entropy calculation later
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[0][input_length:]
        step_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate some basic features (like token count) for the controllers
        step_dict = {
            "step_text": step_text,
            "num_tokens": len(generated_tokens),
            "is_eos": self.tokenizer.eos_token_id in generated_tokens
        }
        
        return step_dict
