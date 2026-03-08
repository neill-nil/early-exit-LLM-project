import torch
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from models.llm_wrapper import HuggingFaceLLMWrapper
from train_controller import EarlyExitMLP

class EarlyExitPipeline:
    def __init__(self, llm_wrapper, controller_path="models/early_exit_controller.pt", scaler_path="models/scaler.pt", threshold=0.8):
        self.llm = llm_wrapper
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load embedding model
        print("Loading SentenceTransformer...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        
        # Load scaler
        print("Loading scaler...")
        scaler = torch.load(scaler_path, map_location=self.device)
        self.mean_scalars = scaler["mean"].to(self.device).float()
        self.std_scalars = scaler["std"].to(self.device).float()
        
        # Load MLP
        print("Loading Early-Exit Controller...")
        self.controller = EarlyExitMLP(input_dim=386).to(self.device)
        self.controller.load_state_dict(torch.load(controller_path, map_location=self.device))
        self.controller.eval()
        
    def extract_final_answer(self, text, prompt):
        """
        Gracefully extracts the FINAL answer from a truncated reasoning trace.
        It uses the original prompt to determine what kind of answer to look for.
        """
        text_lower = text.lower()
        prompt_lower = prompt.lower()
        
        # Determine question type from prompt instructions
        # Multiple Choice (MathQA style)
        if 'multiple-choice' in prompt_lower or 'options:' in prompt_lower:
            # Often it says "option is b", "option b", "answer is b"
            matches = re.findall(r'\b(?:option|answer)(?:\s+is)?\s+([a-e])\b', text_lower)
            if matches:
                return matches[-1]
                
        # Boolean QA (StrategyQA / HotpotQA style)
        elif "'yes' or 'no'" in prompt_lower or 'logical question' in prompt_lower:
            matches = re.findall(r'\b(yes|no|true|false)\b', text_lower)
            if matches:
                return matches[-1] # Grabs the last logical boolean derived
            return ""
                
        # Standard Math / Default Fallback
        # We grab the *last* number generated, as our few-shot prompt concludes with "The final answer is X"
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
            
        return ""

    def generate_with_early_exit(self, prompt, max_steps=60, step_tokens=20):
        current_generation = ""
        early_exit_triggered = False
        stopped_at_step = -1
        total_tokens = 0
        
        for step_idx in range(max_steps):
            step_info = self.llm.generate_step(
                prompt=prompt,
                current_generation=current_generation,
                step_tokens_limit=step_tokens
            )
            
            step_text = step_info["step_text"]
            current_generation += step_text
            total_tokens += step_info["num_tokens"]
            
            # --- Early Exit Check ---
            with torch.no_grad():
                # 1. Embed text
                emb = self.embedder.encode(current_generation, convert_to_tensor=True, device=self.device)
                
                # 2. Normalize scalars
                scalars = torch.tensor([step_idx, total_tokens], dtype=torch.float32, device=self.device)
                scalars_norm = (scalars - self.mean_scalars) / self.std_scalars
                
                # 3. Concatenate and predict
                x = torch.cat([emb, scalars_norm]).unsqueeze(0) # Batch size 1
                prob = self.controller(x).item()
                
            print(f"  [Step {step_idx} | Tokens: {total_tokens}] Controller Confidence: {prob*100:.2f}%")
            
            # If our MLP is X% confident the answer is correct/present, we STOP the LLM.
            if prob >= self.threshold:
                print(f"  >>> EARLY EXIT TRIGGERED (Confidence {prob*100:.2f}% >= Threshold {self.threshold*100:.2f}%)")
                early_exit_triggered = True
                stopped_at_step = step_idx
                break
                
            if step_info["is_eos"]:
                break
                
        final_answer = self.extract_final_answer(current_generation, prompt)
        
        return {
            "generation": current_generation,
            "extracted_answer": final_answer,
            "early_exit_triggered": early_exit_triggered,
            "stopped_at_step": stopped_at_step,
            "total_tokens": total_tokens
        }

if __name__ == "__main__":
    print("Successfully loaded Inference Pipeline. Import this in Kaggle to test!")
