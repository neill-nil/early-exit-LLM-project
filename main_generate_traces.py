import json
import os
import re
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from models.llm_wrapper import HuggingFaceLLMWrapper

def extract_answer_gsm8k(text: str) -> str:
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
        numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_part)
        if numbers:
            return numbers[0]
        return answer_part
    return ""

def get_question(item) -> str:
    if "Problem" in item: # MathQA
        return item["Problem"]
    if "Body" in item and "Question" in item: # SVAMP
        return f"{item['Body']} {item['Question']}"
    if "question" in item: # GSM8K, StrategyQA
        return item["question"]
    return str(item)

def get_true_answer(item, dataset_name: str) -> str:
    dataset_name = dataset_name.lower()
    if 'gsm8k' in dataset_name:
        return extract_answer_gsm8k(item.get('answer', ''))
        
    if 'math_qa' in dataset_name:
        correct_letter = item.get('correct', '').strip().lower()
        options = item.get('options', '')
        # Matches "a ) 123" or "a) 123" 
        match = re.search(fr"{correct_letter}\s*\)\s*([^,]+)", options, re.IGNORECASE)
        if match:
             # Often has trailing quotes or spaces, clean it up
             ans = match.group(1).strip().strip("'\"").strip()
             return ans
        return correct_letter
        
    if 'svamp' in dataset_name:
        return str(item.get('Answer', ''))
        
    if 'strategy' in dataset_name:
        ans = item.get('answer', '')
        if isinstance(ans, bool):
            return "true" if ans else "false"
        return str(ans).strip().lower()
        
    if 'hotpot' in dataset_name:
        return str(item.get('answer', '')).strip().lower()
        
    # Fallback
    for col in ['answer', 'Answer', 'correct', 'target']:
        if col in item:
            return str(item[col])
    return ""

def check_intermediate_correctness(text: str, true_answer: str, dataset_name: str) -> bool:
    """
    Checks if the true answer appears anywhere in the text.
    Uses regex to ensure it matches the token distinctly.
    """
    true_answer = true_answer.strip().lower()
    text_lower = text.lower()
    
    # QA datasets usually mean looking for "yes" / "no" / "true" / "false" or an exact match phrase
    if 'strategy' in dataset_name.lower() or 'hotpot' in dataset_name.lower():
        if true_answer == "true" or true_answer == "yes":
            return bool(re.search(r'\b(true|yes)\b', text_lower))
        elif true_answer == "false" or true_answer == "no":
            return bool(re.search(r'\b(false|no)\b', text_lower))
        return bool(re.search(fr"\b{re.escape(true_answer)}\b", text_lower))
        
    # Math datasets: true_answer might be messy (e.g., "30 % .")
    # So we should extract the core number from the true_answer first
    numbers = re.findall(r'-?\d+(?:\.\d+)?', true_answer)
    if numbers:
        search_num = numbers[0]
        # Look for exact number surrounded by non-digits
        pattern = fr"(?<!\d){re.escape(search_num)}(?!\d)"
        if re.search(pattern, text_lower):
            return True
    else:
        # Fallback if there are no numbers in the true answer but it's a math dataset
        if re.search(fr"\b{re.escape(true_answer)}\b", text_lower):
            return True
        
    return False

def generate_traces_for_dataset(
    model_wrapper: HuggingFaceLLMWrapper,
    dataset_name: str,
    output_path: str,
    split: str = "train",
    start_idx: int = 0,
    end_idx: int = 20
):
    print(f"Loading {dataset_name} {split} split...")
    try:
        # Check if local data exists for this dataset 
        local_path = f"data/{dataset_name.split('/')[-1]}"
        if os.path.exists(local_path):
            dataset = load_from_disk(local_path)[split]
        else:
            print(f"Local {dataset_name} data not found. Downloading from Hugging Face...")
            if dataset_name == "gsm8k":
                dataset = load_dataset(dataset_name, "main")[split]
            else:
                dataset = load_dataset(dataset_name)[split]
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # Take the specific slice requested
    if start_idx is not None and end_idx is not None:
        # Ensure we don't go out of bounds of the dataset
        end_idx = min(end_idx, len(dataset))
        print(f"Slicing dataset from index {start_idx} to {end_idx}...")
        dataset = dataset.select(range(start_idx, end_idx))
        
    results = []
    
    # Adjust prompt depending on dataset
    if 'strategy' in dataset_name.lower() or 'hotpot' in dataset_name.lower():
        prompt_template = (
            "Read the following question and answer it step-by-step.\n"
            "State your final answer clearly.\n"
            "After stating the answer, briefly double-check your logic to ensure it is correct.\n\n"
            "Question: {question}\n\nSolution:\n"
        )
    else:
        prompt_template = (
            "Solve the following math problem step-by-step.\n"
            "State your final numerical answer clearly.\n"
            "After stating the answer, briefly double-check your work to ensure it is correct.\n\n"
            "Problem: {question}\n\nSolution:\n"
        )
    
    print(f"Starting trace generation for {len(dataset)} samples...")
    for idx, item in enumerate(tqdm(dataset)):
        real_idx = start_idx + idx 
        
        question = get_question(item)
        true_answer_str = get_true_answer(item, dataset_name)
        
        prompt = prompt_template.format(question=question)
        
        # We need to generate step-by-step to save the "trace"
        # Each step will be ~15 tokens. We allow up to 40 steps (600 tokens max)
        current_generation = ""
        steps_data = [] # Stores what was generated at each step
        
        max_steps = 40
        already_solved = False
        solved_at_step = -1
        
        for step_idx in range(max_steps):
            step_info = model_wrapper.generate_step(
                prompt=prompt,
                current_generation=current_generation,
                step_tokens_limit=15
            )
            
            step_text = step_info["step_text"]
            current_generation += step_text
            
            # Check if this latest addition contains the correct answer
            is_step_correct = check_intermediate_correctness(current_generation, true_answer_str, dataset_name)
            
            if is_step_correct and not already_solved:
                already_solved = True
                solved_at_step = step_idx
            
            # Record the trace step
            steps_data.append({
                "step_index": step_idx,
                "text_added": step_text,
                "cumulative_text": current_generation,
                "num_tokens": step_info["num_tokens"],
                "has_correct_answer": is_step_correct  # New: label each step!
            })
            
            # Stop if the model produced an EOS token
            if step_info["is_eos"]:
                break
                
        # Evaluate if the final answer was correct (fallback)
        model_final_answer = "" # Difficult to extract uniformally across all datasets without complex logic
        is_correct = already_solved 
        
        results.append({
            "id": real_idx,
            "question": question,
            "true_answer": true_answer_str,
            "model_answer": model_final_answer,
            "is_correct": is_correct,
            "solved_at_step": solved_at_step, # New metadata 
            "full_generation": current_generation,
            "steps": steps_data
        })
        
        # Periodically save to avoid losing data if Kaggle notebook crashes
        if (idx + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
                
    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Finished! Traces saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate reasoning traces")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="HuggingFace dataset name (e.g. gsm8k, math_qa, hotpot_qa)")
    parser.add_argument("--start", type=int, default=0, help="Starting index of the dataset")
    parser.add_argument("--end", type=int, default=20, help="Ending index of the dataset")
    args = parser.parse_args()

    # Ensure output directory exists (can remove if outputting directly to root)
    os.makedirs("data/traces", exist_ok=True)
    
    # Initialize the LLM Wrapper (This will download Qwen2.5-Math-7B)
    print("Initializing Qwen2.5-Math-7B-Instruct...")
    wrapper = HuggingFaceLLMWrapper(model_name="Qwen/Qwen2.5-Math-7B-Instruct")
    
    # Create a dynamic filename based on dataset and batch
    dataset_clean = args.dataset.split('/')[-1]
    filename = f"{dataset_clean}_train_traces_{args.start}_to_{args.end}.json"
    
    generate_traces_for_dataset(
        model_wrapper=wrapper,
        dataset_name=args.dataset,
        output_path=filename, # Saves to working directory
        split="train",
        start_idx=args.start,
        end_idx=args.end 
    )
