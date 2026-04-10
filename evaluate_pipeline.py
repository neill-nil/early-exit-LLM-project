import json
import os
import re as _re
from tqdm import tqdm
from datasets import load_dataset
from models.llm_wrapper import HuggingFaceLLMWrapper
from early_exit_inference import EarlyExitPipeline
<<<<<<< HEAD
from main_generate_traces import get_question, get_true_answer

def compute_dynamic_baseline(dataset_identifiers: list, default_baseline: float) -> float:
    """Calculates the average unconstrained tokens used from the existing trace files."""
    traces_dir = "data/traces"
    total_tokens = 0
    count = 0
    
    for identifier in dataset_identifiers:
        for file_path in glob.glob(f"{traces_dir}/*{identifier}*.json"):
            if "fewshot" in file_path.lower() or "eval" in file_path.lower(): 
                continue
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        if "steps" in item and len(item["steps"]) > 0:
                            last_step = item["steps"][-1]
                            total_tokens += last_step.get("num_tokens", 0)
                            count += 1
            except Exception:
                pass
                
    if count > 0:
        return float(total_tokens) / count
    return default_baseline

def _regex_check(extracted: str, true_ans: str) -> bool:
    """Word-boundary regex check to avoid '60' matching '600'."""
    try:
        pattern = r'(?<![\d.])' + re.escape(true_ans) + r'(?![\d.])'
        return bool(re.search(pattern, extracted, re.IGNORECASE))
    except re.error:
        return extracted == true_ans


def _check_math_qa_correctness(extracted_ans: str, true_ans: str) -> bool:
    """Robust extraction matching for MathQA."""
    extracted_ans = str(extracted_ans).strip().lower()
    true_ans = true_ans.strip().lower()
    
    match = re.search(r"option\s+([a-e])\s*\(value:\s*(.*?)\)", true_ans)
    if match:
        letter = match.group(1).strip()
        value = match.group(2).strip()
        
        # Clean up value if formatted like "c ) 2"
        val_match = re.search(r"[a-e]\s*\)\s*(.*)", value)
        if val_match:
            value = val_match.group(1).strip()
            
        if extracted_ans == letter or extracted_ans == value:
            return True
            
        # Fallback to regex word boundary check against value
        return _regex_check(extracted_ans, value)
        
    return _regex_check(extracted_ans, true_ans)
=======
from main_generate_traces import get_question, get_true_answer, check_intermediate_correctness_llm as check_intermediate_correctness
>>>>>>> origin/dev-consistency-work

def get_few_shot_prompt(dataset_name):
    dataset_lower = dataset_name.lower()
    if 'math_qa' in dataset_lower:
        return (
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
        return (
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
        return (
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

def evaluate_on_dataset(pipeline, dataset_name, split="test", num_samples=20, start_idx=0):
    print(f"\nEvaluating pipeline on {dataset_name} ({split} set)...")
    
    try:
        # Check if local data exists first, exactly like traces generation
        local_path = f"data/{dataset_name.split('/')[-1]}"
        if os.path.exists(local_path):
            from datasets import load_from_disk
            dataset = load_from_disk(local_path)[split]
        else:
            if dataset_name == "gsm8k":
                dataset = load_dataset(dataset_name, "main", trust_remote_code=True)[split]
            else:
                dataset = load_dataset(dataset_name, trust_remote_code=True)[split]
                
        if dataset_name == "math_qa":
            # Taking samples 300 to 300+num_samples from the train split to avoid training overlap
            dataset = dataset.select(range(300, 300 + num_samples))
        else:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
    except Exception as e:
        print(f"Hugging Face fetch failed ({e}). Attempting offline load from data/{dataset_name.split('/')[-1]}...")
        from datasets import load_from_disk
        dataset = load_from_disk(f"data/{dataset_name.split('/')[-1]}")[split]
        
        if dataset_name == "math_qa":
            dataset = dataset.select(range(300, 300 + num_samples))
        else:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

<<<<<<< HEAD
    if dataset_name == "math_qa":
        # Taking samples from start_idx to avoid training overlap
        dataset = dataset.select(range(start_idx, start_idx + num_samples))
    else:
        dataset = dataset.select(range(start_idx, min(start_idx + num_samples, len(dataset))))
=======
>>>>>>> origin/dev-consistency-work
    prompt_template = get_few_shot_prompt(dataset_name)
    
    results = []
    total_baseline_tokens = 0
    total_exit_tokens = 0
    correct_extractions = 0
    
    for idx, item in enumerate(tqdm(dataset)):
        question = get_question(item, dataset_name)
        true_ans = get_true_answer(item, dataset_name).strip().lower()
        prompt = prompt_template.format(question=question)
        
        MAX_STEPS = 60
        STEP_TOKENS = 40
        
        # We assume a dataset-specific average baseline derived from the unconstrained training traces
        if "gsm8k" in dataset_name.lower():
<<<<<<< HEAD
            baseline_tokens = compute_dynamic_baseline(["gsm8k"], 387.4)
        elif "math_qa" in dataset_name.lower():
            baseline_tokens = compute_dynamic_baseline(["math_qa"], 523.9)
=======
            baseline_tokens = 386.0  # Empirical average from GSM8K traces
        elif "math_qa" in dataset_name.lower():
            baseline_tokens = 500.0  # Empirical average from MathQA traces
>>>>>>> origin/dev-consistency-work
        else:
            baseline_tokens = 600.0  # Fallback
            
        exit_result = pipeline.generate_with_early_exit(prompt, max_steps=MAX_STEPS, step_tokens=STEP_TOKENS)
        
        actual_tokens_used = exit_result["total_tokens"]
        extracted_ans = exit_result["extracted_answer"].strip().lower()
        
        # ── Check correctness (dataset-aware) ─────────────────────────────
        if true_ans == "":
            is_correct = False
<<<<<<< HEAD
        elif "math_qa" in dataset_name.lower():
            # Extract both letter and value to verify correctness rigorously
            is_correct = _check_math_qa_correctness(extracted_ans, true_ans)
        elif extracted_ans == true_ans:
            is_correct = True
        else:
            # Word-boundary regex check to avoid "60" matching "600"
            is_correct = _regex_check(extracted_ans, true_ans)
=======
>>>>>>> origin/dev-consistency-work

        elif "math_qa" in dataset_name.lower():
            # get_true_answer returns "option c (value: 24)" for MathQA.
            # The pipeline's extract_final_answer returns just the letter (e.g. "c").
            # We must compare letter-to-letter to avoid a guaranteed 0% accuracy.
            letter_match = _re.search(r'option\s+([a-e])', true_ans, _re.IGNORECASE)
            true_letter = letter_match.group(1).lower() if letter_match else true_ans.strip()
            is_correct = (extracted_ans == true_letter)

        elif extracted_ans == true_ans:
            is_correct = True  # Exact match is always correct

        else:
            # Fallback: substring/numeric check to avoid "60" matching "600"
            is_correct = bool(_re.search(rf"(?<![\d.])" + _re.escape(true_ans) + r"(?![\d.])", extracted_ans))
            
        if is_correct:
            correct_extractions += 1
            
        total_baseline_tokens += baseline_tokens
        total_exit_tokens += actual_tokens_used
        
        results.append({
            "question": question,
            "true_ans": true_ans,
            "extracted_ans": extracted_ans,
            "is_correct": is_correct,
            "early_exit": exit_result["early_exit_triggered"],
            "tokens_used": actual_tokens_used,
            "true_baseline_tokens": baseline_tokens,
            "tokens_saved": baseline_tokens - actual_tokens_used,
            "full_generation": exit_result["generation"]
        })
        
    avg_accuracy = correct_extractions / num_samples
    token_savings_pct = ((total_baseline_tokens - total_exit_tokens) / total_baseline_tokens) * 100
    
    # Save detailed evaluation results to disk
    os.makedirs("results", exist_ok=True)
    output_filename = f"results/eval_results_{dataset_name.split('/')[-1]}_{split}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
        
    print("-" * 50)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Final Accuracy: {avg_accuracy*100:.2f}%")
    print(f"Total Tokens Saved vs Historical Trace Average ({total_baseline_tokens / num_samples:.1f}): {token_savings_pct:.2f}%")
    print(f"Average Tokens Generated Before Exit: {total_exit_tokens / num_samples:.1f}")
    print(f"Detailed logs saved to: {output_filename}")
    print("-" * 50)
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Early Exit Pipeline")
    parser.add_argument(
        "--strategy", type=str, default="mlp", choices=["mlp", "consistency"],
        help="Which early-exit strategy to use: 'mlp' (Approach 1) or 'consistency' (Approach 3)."
    )
    # MLP-specific arguments
    parser.add_argument("--controller", type=str, default="models/early_exit_controller.pt", help="Path to early_exit_controller.pt (mlp only)")
    parser.add_argument("--scaler",     type=str, default="models/scaler.pt",                help="Path to scaler.pt (mlp only)")
    parser.add_argument("--threshold",  type=float, default=0.85,                            help="MLP confidence threshold (mlp only)")
    # Consistency-specific arguments
    parser.add_argument("--consistency-threshold", type=int, default=2, help="Consecutive stable answers before exit (consistency only)")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-3B-Instruct",      help="HF model name for the judge (consistency only)")
    # Shared arguments
    parser.add_argument("--dataset",    type=str, default="all", choices=["all", "gsm8k", "math_qa"], help="Dataset to evaluate")
    parser.add_argument("--num_samples", type=int, default=25,                               help="Number of test samples")
    parser.add_argument("--start", type=int, default=0, help="Start index (use 300 for math_qa if you want to skip trace overlap blindly).")
    args = parser.parse_args()

    print("Initializing Qwen2.5-Math-7B-Instruct (reasoner)...")
    wrapper = HuggingFaceLLMWrapper(model_name="Qwen/Qwen2.5-Math-7B-Instruct")

    # ── Build the selected strategy ──────────────────────────────────────────
    if args.strategy == "mlp":
        from strategies.learning_based import LearningBasedController
        print(f"Strategy: MLP (threshold={args.threshold})")
        strategy = LearningBasedController(
            controller_path=args.controller,
            scaler_path=args.scaler,
            threshold=args.threshold,
        )
    else:  # consistency
        from strategies.consistency import ConsistencyController
        print(f"Strategy: Consistency (threshold={args.consistency_threshold}, judge={args.judge_model})")
        judge_wrapper = HuggingFaceLLMWrapper(model_name=args.judge_model)
        strategy = ConsistencyController(
            judge_wrapper=judge_wrapper,
            consistency_threshold=args.consistency_threshold,
            dataset_name=args.dataset if args.dataset != "all" else "gsm8k",
        )

    pipeline = EarlyExitPipeline(wrapper, strategy=strategy)

    if args.dataset in ["all", "gsm8k"]:
<<<<<<< HEAD
        evaluate_on_dataset(pipeline, "gsm8k", split="test", num_samples=args.num_samples, start_idx=args.start)
    
    if args.dataset in ["all", "math_qa"]:
        # User only has math_qa train split uploaded to Kaggle local storage, so we evaluate strictly on unseen "train" slice
        # Default start array fallback applied to 600 if math_qa is used and start is left at 0 to avoid training overlap
        actual_start = args.start if args.start > 0 else 600
        evaluate_on_dataset(pipeline, "math_qa", split="train", num_samples=args.num_samples, start_idx=actual_start)
=======
        evaluate_on_dataset(pipeline, "gsm8k", split="test", num_samples=args.num_samples)

    if args.dataset in ["all", "math_qa"]:
        # math_qa: evaluate on an unseen train slice to avoid training overlap
        evaluate_on_dataset(pipeline, "math_qa", split="train", num_samples=args.num_samples)
>>>>>>> origin/dev-consistency-work
