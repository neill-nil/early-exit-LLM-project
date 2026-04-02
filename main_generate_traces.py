import json
import os
import re
import google.generativeai as genai
import time
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

def get_question(item, dataset_name: str) -> str:
    dataset_name = dataset_name.lower()
    if "Problem" in item and "math_qa" in dataset_name: 
        options = item.get("options", "")
        return f"{item['Problem']}\nOptions: {options}"
    if "math_qa" in dataset_name and "Problem" in item:
        return item["Problem"]
    if "Body" in item and "Question" in item: # SVAMP
        return f"{item['Body']} {item['Question']}"
    if "question" in item: # GSM8K, StrategyQA, HotpotQA
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



def check_intermediate_correctness_llm(text: str, true_answer: str, question: str, judge_wrapper: HuggingFaceLLMWrapper) -> bool:
    """
    Uses a locally hosted base LLM to carefully read the reasoning trace and decide 
    if the student has actually stated the final answer yet.
    """
    # Fast path: if the exact answer string isn't even in the text, 
    # there is zero percent chance they have stated it as the final conclusion.
    if str(true_answer).strip() not in text:
        return False

    prompt = f"""You are an incredibly strict math teacher grading a student's partial scratchpad.

    Problem: {question}
    Correct Final Answer: {true_answer}

    Student's current scratchpad:
    \"\"\"{text}\"\"\"

    Task: Has the student explicitly arrived at and written down the final answer "{true_answer}" as their FINAL conclusion to the problem?
    - If the student's scratchpad cuts off before stating the final answer, output NO.
    - If the student wrote "{true_answer}" but as part of an intermediate calculation (e.g. adding numbers), output NO.
    - If the student clearly concludes their work with the final answer "{true_answer}", output YES.

    Respond with exactly and ONLY the word "YES" or "NO". Do not explain.
    """
    
    # We MUST apply the Instruct chat template so the model behaves like an assistant, not an autocomplete engine
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = judge_wrapper.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    # We pass the prompt to the local judge model and ask for a very short generation
    # Temperature 0.01 forces effectively deterministic answers without crashing the sampler
    response_text = judge_wrapper.generate(formatted_prompt, max_new_tokens=10, temperature=0.01)
    decision = response_text.strip().upper()
    
    return decision.startswith("YES")

def generate_traces_for_dataset(
    model_wrapper: HuggingFaceLLMWrapper,
    judge_wrapper: HuggingFaceLLMWrapper,
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
    
    dataset_lower = dataset_name.lower()
    
    # Few-shot prompt templates tailored to each dataset
    if 'math_qa' in dataset_lower:
        prompt_template = (
            "Solve the following multiple-choice math problem step-by-step.\n"
            "State your final numerical answer clearly, and explicitly mention the correct option letter.\n"
            "After stating the answer, briefly double-check your work to ensure it is correct.\n\n"
            "Example:\n"
            "Problem: a shopkeeper sells an article at a loss of 10 % . if he had sold it for rs . 45 more , he would have gained 5 % . find the cost price of the article ?\n"
            "Options: a ) 250 , b ) 300 , c ) 350 , d ) 400 , e ) 450\n"
            "Solution: Let the cost price be x.\n"
            "Loss = 10% of x = 0.1x. Selling price = x - 0.1x = 0.9x.\n"
            "If sold for 45 more, new selling price = 0.9x + 45.\n"
            "New gain = 5% of x = 0.05x. New selling price = x + 0.05x = 1.05x.\n"
            "Therefore, 0.9x + 45 = 1.05x.\n"
            "0.15x = 45 => x = 45 / 0.15 = 300.\n"
            "The cost price is 300. The correct option is b.\n"
            "Double-check: 10% loss on 300 is 270. 270 + 45 = 315. 315 is 105% of 300. Thus, 5% gain. Correct.\n\n"
            "Problem: {question}\n\nSolution:\n"
        )
    elif 'strategy' in dataset_lower or 'hotpot' in dataset_lower:
        prompt_template = (
            "Read the following logical question and answer it step-by-step.\n"
            "State your final answer clearly as either 'Yes' or 'No', or the exact target entity.\n"
            "After stating the answer, briefly double-check your logic to ensure it is correct.\n\n"
            "Example:\n"
            "Question: Do hamsters provide food for any animals?\n"
            "Solution: Hamsters are prey animals. Many predators, such as hawks, owls, and snakes, hunt and eat hamsters in the wild.\n"
            "Therefore, hamsters provide food for other animals. The answer is Yes.\n"
            "Double-check: Prey animals provide food for predators. Hamsters are prey animals. Thus, Yes is correct.\n\n"
            "Question: {question}\n\nSolution:\n"
        )
    else: # GSM8K, SVAMP
        prompt_template = (
            "Solve the following math problem step-by-step.\n"
            "State your final numerical answer clearly.\n"
            "After stating the answer, briefly double-check your work to ensure it is correct.\n\n"
            "Example:\n"
            "Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
            "Solution: Natalia sold 48 clips in April. In May, she sold half as many, which is 48 / 2 = 24 clips.\n"
            "Altogether, she sold 48 + 24 = 72 clips.\n"
            "The final answer is 72.\n"
            "Double-check: 48 (April) + 24 (May) = 72. Correct.\n\n"
            "Problem: {question}\n\nSolution:\n"
        )
    
    print(f"Starting trace generation for {len(dataset)} samples...")
    for idx, item in enumerate(tqdm(dataset)):
        real_idx = start_idx + idx 
        
        question = get_question(item, dataset_name)
        true_answer_str = get_true_answer(item, dataset_name)
        
        prompt = prompt_template.format(question=question)
        
        # We need to generate step-by-step to save the "trace"
        # Each step will be ~15 tokens. We allow up to 40 steps (600 tokens max)
        current_generation = ""
        steps_data = [] # Stores what was generated at each step
        
        max_steps = 60
        already_solved = False
        solved_at_step = -1
        
        for step_idx in range(max_steps):
            step_info = model_wrapper.generate_step(
                prompt=prompt,
                current_generation=current_generation,
                step_tokens_limit=20
            )
            
            step_text = step_info["step_text"]
            current_generation += step_text
            
            # Check if this latest addition contains the correct answer
            is_step_correct = check_intermediate_correctness_llm(current_generation, true_answer_str, question, judge_wrapper)
            
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
    
    print("Initializing Qwen2.5-3B-Instruct (Local Judge)...")
    judge_wrapper = HuggingFaceLLMWrapper(model_name="Qwen/Qwen2.5-3B-Instruct")
    
    # Create a dynamic filename based on dataset and batch
    dataset_clean = args.dataset.split('/')[-1]
    filename = f"{dataset_clean}_train_traces_{args.start}_to_{args.end}.json"
    
    generate_traces_for_dataset(
        model_wrapper=wrapper,
        judge_wrapper=judge_wrapper,
        dataset_name=args.dataset,
        output_path=filename, # Saves to working directory
        split="train",
        start_idx=args.start,
        end_idx=args.end 
    )
