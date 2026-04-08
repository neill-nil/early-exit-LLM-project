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
        
        # Calculate dynamic input dimension (384 embedding dims + scaled features)
        dynamic_input_dim = 384 + len(self.mean_scalars)
        
        self.controller = EarlyExitMLP(input_dim=dynamic_input_dim).to(self.device)
        self.controller.load_state_dict(torch.load(controller_path, map_location=self.device))
        self.controller.eval()
        
    def has_clear_answer(self, text, prompt):
        """Checks if the generation already contains a clearly formatted final answer."""
        text_lower = text.lower()
        prompt_lower = prompt.lower()
        
        if 'multiple-choice' in prompt_lower or 'options:' in prompt_lower:
            if re.search(r'\b(?:option|answer)(?:\s+is)?\s+([a-e])\b', text_lower):
                return True
        elif "'yes' or 'no'" in prompt_lower or 'logical question' in prompt_lower:
            if re.search(r'\b(yes|no|true|false)\b', text_lower):
                return True
                
        # Universal checks
        if re.search(r'\\boxed\{([^\}]+)\}', text) or ("the final answer is" in text_lower):
            return True
            
        return False
        
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
    def extract_final_answer(self, text, prompt=None):
        """Extracts the final answer from traces."""
        text_lower = text.lower()
        
        # --- StrategyQA Support ---
        if prompt and ("'yes' or 'no'" in prompt.lower() or "yes/no question" in prompt.lower()):
            # If the text explicitly says "the answer is yes" or "the answer is no"
            if "the answer is yes" in text_lower or "'yes'" in text_lower:
                return "yes"
            elif "the answer is no" in text_lower or "'no'" in text_lower:
                return "no"
            
            # Fallback to the absolute last occurrence of yes or no
            matches = re.findall(r'\b(yes|no)\b', text_lower)
            if matches:
                return matches[-1]
            return ""
            
        # --- Standard Math Support ---
        # Priority 1: Qwen often natively outputs \boxed{number}
        boxed_matches = re.findall(r'\\boxed\{([^\}]+)\}', text)
        if boxed_matches:
            boxed_val = boxed_matches[-1]
            nums_in_box = re.findall(r'-?\d+(?:\.\d+)?', boxed_val)
            if nums_in_box: return nums_in_box[-1]
            return boxed_val

        # Priority 2: Look for our explicit prompt ending string
        if "the final answer is" in text_lower:
            ans_part = text_lower.split("the final answer is")[-1]
            nums = re.findall(r'-?\d+(?:\.\d+)?', ans_part)
            if nums: return nums[0]
            
        # Priority 3: Extract the last number BEFORE the "double-check" phase starts
        if "double-check:" in text_lower:
            main_text = text_lower.split("double-check:")[0]
            numbers = re.findall(r'-?\d+(?:\.\d+)?', main_text)
            if numbers: return numbers[-1]

        # Priority 4: Default Fallback to the absolute last number generated
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
            
        return ""

    def generate_with_early_exit(self, prompt, max_steps=60, step_tokens=20, min_steps_before_check=4):
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
            
            # Skip early exit check for the initial steps (unless eos ends generation)
            if step_idx <= min_steps_before_check and not step_info["is_eos"]:
                print(f"  [Step {step_idx} | Tokens: {total_tokens}] Warmup — skipping early exit check")
                continue
            
            # --- Early Exit Check ---
            with torch.no_grad():
                # 1. Embed text (ensure it's 2D: [1, 384])
                emb = self.embedder.encode(current_generation, convert_to_tensor=True, device=self.device)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                
                # 2. Normalize scalars
                # We use the token count *for this step*, NOT the accumulating total, because that is what the MLP was trained on in prepare_features.py
                current_step_tokens = step_info["num_tokens"]
                
                scalar_features = [step_idx, current_step_tokens]
                
                # If trained with advanced features, the scaler mean vector will have length > 2
                if len(self.mean_scalars) > 2:
                    from strategies.advanced_features import extract_advanced_features
                    step_dict = {
                        "text_added": step_info["step_text"],
                        "cumulative_text": current_generation,
                        "num_tokens": current_step_tokens,
                        "step_index": step_idx
                    }
                    scalar_features.extend(extract_advanced_features(step_dict))
                
                scalars = torch.tensor([scalar_features], dtype=torch.float32, device=self.device)
                scalars_norm = (scalars - self.mean_scalars) / self.std_scalars
                
                # 3. Concatenate (Result length automatically adjusts)
                x = torch.cat([emb, scalars_norm], dim=1)
                
                # Predict
                prob = self.controller(x).item()
                
            print(f"  [Step {step_idx} | Tokens: {total_tokens}] Controller Confidence: {prob*100:.2f}%")
            
            # If our MLP is X% confident the answer is correct/present, we STOP the LLM.
            if prob >= self.threshold:
                if self.has_clear_answer(current_generation, prompt):
                    print(f"  >>> EARLY EXIT TRIGGERED (Confidence {prob*100:.2f}% >= Threshold {self.threshold*100:.2f}%)")
                    early_exit_triggered = True
                    stopped_at_step = step_idx
                    break
                else:
                    print(f"  >>> Controller confident ({prob*100:.2f}%), but answer not fully printed. Running another step...")
                
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
    import argparse
    parser = argparse.ArgumentParser(description="Test Early Exit Pipeline")
    parser.add_argument("--controller", type=str, default="models/early_exit_controller.pt", help="Path to early_exit_controller.pt")
    parser.add_argument("--scaler", type=str, default="models/scaler.pt", help="Path to scaler.pt")
    args = parser.parse_args()
    
    print("Successfully loaded Inference Pipeline. Import this in Kaggle to test!")
    print(f"When initializing EarlyExitPipeline, use controller_path='{args.controller}' and scaler_path='{args.scaler}'")
