import torch
from models.llm_wrapper import HuggingFaceLLMWrapper
from early_exit_inference import EarlyExitPipeline
from strategies.static_budget import StaticBudgetController

class StaticBudgetPipeline(EarlyExitPipeline):
    def __init__(self, llm_wrapper, controller_path="models/static_classifier.pkl"):
        # We don't call super() fully because we don't need the dynamic scaler/MLP
        self.llm = llm_wrapper
        print(f"Loading Static Budget Controller from {controller_path}...")
        self.controller = StaticBudgetController(budget_model_path=controller_path)
        
    def generate_with_early_exit(self, prompt, max_steps=60, step_tokens=20, min_steps_before_check=None, question=None):
        current_generation = ""
        early_exit_triggered = False
        stopped_at_step = -1
        total_tokens = 0
        
        # 1. Predict difficulty once before generation starts (use bare question to match training distribution)
        text_to_embed = question if question else prompt
        budget = self.controller.predict_budget(text_to_embed)
        print(f"  [Static Budget] Assigned limit: {budget} tokens based on prompt difficulty.")
        
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
            info_dict = {"total_tokens": total_tokens, "budget": budget}
            
            if self.controller.should_exit(prompt, current_generation, info_dict):
                print(f"  >>> STATIC EXIT TRIGGERED (Tokens {total_tokens} >= Budget Limit {budget})")
                print("  >>> Mercilessly chopping off model generation limit.")
                early_exit_triggered = True
                stopped_at_step = step_idx
                break
                
            if step_info["is_eos"]:
                break
                
        final_answer = self.extract_final_answer(current_generation, prompt)
        found_answer = self.has_clear_answer(current_generation, prompt) or (final_answer != "")
        
        return {
            "generation": current_generation,
            "extracted_answer": final_answer,
            "found_answer": found_answer,
            "early_exit_triggered": early_exit_triggered,
            "stopped_at_step": stopped_at_step,
            "total_tokens": total_tokens
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Static Budget Pipeline")
    parser.add_argument("--controller", type=str, default="models/static_classifier.pkl", help="Path to static_classifier.pkl")
    args = parser.parse_args()
    
    print("Successfully loaded Static Budget Inference Pipeline. Import this in Kaggle to test!")
    print(f"When initializing, use controller_path='{args.controller}'")
