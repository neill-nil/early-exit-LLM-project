import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os
import re
from tqdm import tqdm
from datasets import load_dataset
from models.llm_wrapper import HuggingFaceLLMWrapper
from early_exit_inference import EarlyExitPipeline
from main_generate_traces import get_question, get_true_answer

def _regex_check(extracted: str, true_ans: str) -> bool:
    """Word-boundary regex check to avoid '60' matching '600'."""
    try:
        pattern = r'(?<![\d.])' + re.escape(true_ans) + r'(?![\d.])'
        return bool(re.search(pattern, extracted, re.IGNORECASE))
    except re.error:
        return extracted == true_ans


def _extract_option_letter(true_ans: str) -> str:
    """Extract just the letter from 'option d (value: 45)' → 'd'."""
    m = re.match(r'option\s+([a-e])', true_ans.strip(), re.IGNORECASE)
    return m.group(1).lower() if m else true_ans.strip().lower()

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

def evaluate_on_dataset(pipeline, dataset_name, split="test", num_samples=20):
    print(f"\nEvaluating pipeline on {dataset_name} ({split} set)...")
    
    # 1. Try local disk first (for Kaggle where datasets are pre-loaded)
    local_path = f"data/{dataset_name.split('/')[-1]}"
    dataset = None

    if os.path.exists(local_path):
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(local_path)[split]
            print(f"Loaded from local disk: {local_path}")
        except Exception as e:
            print(f"Local load failed ({e}), trying HuggingFace...")

    # 2. Fall back to HuggingFace download
    if dataset is None:
        try:
            if dataset_name == "gsm8k":
                dataset = load_dataset(dataset_name, "main")[split]
            elif dataset_name == "math_qa":
                dataset = load_dataset("allenai/math_qa")[split]
            else:
                dataset = load_dataset(dataset_name)[split]
            print(f"Loaded from HuggingFace: {dataset_name}")
        except Exception as e:
            raise RuntimeError(
                f"Could not load '{dataset_name}' from disk or HuggingFace. Error: {e}"
            )

    if dataset_name == "math_qa":
        # Taking samples 300 to 300+num_samples from the train split to avoid training overlap
        dataset = dataset.select(range(300, 300 + num_samples))
    else:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
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
        STEP_TOKENS = 20
        
        # We assume a dataset-specific average baseline derived from the unconstrained training traces
        if "gsm8k" in dataset_name.lower():
            baseline_tokens = 386.0  # Empirical average from GSM8K traces
        elif "math_qa" in dataset_name.lower():
            baseline_tokens = 500.0  # Empirical average from MathQA traces
        else:
            baseline_tokens = 600.0  # Fallback
            
        exit_result = pipeline.generate_with_early_exit(prompt, max_steps=MAX_STEPS, step_tokens=STEP_TOKENS, question=question)
        
        actual_tokens_used = exit_result["total_tokens"]
        extracted_ans = exit_result["extracted_answer"].strip().lower()
        found_ans = exit_result.get("found_answer", True) # Default true for older runs
        
        # Check correctness
        if true_ans == "":
            is_correct = False
        elif "math_qa" in dataset_name.lower():
            # true_ans is "option d (value: 45)" — extract just the letter
            true_letter = _extract_option_letter(true_ans)
            is_correct = extracted_ans.strip().lower() == true_letter
        elif extracted_ans == true_ans:
            is_correct = True
        else:
            # Word-boundary regex check to avoid "60" matching "600"
            is_correct = _regex_check(extracted_ans, true_ans)

        if is_correct: correct_extractions += 1
            
        total_baseline_tokens += baseline_tokens
        total_exit_tokens += actual_tokens_used
        
        results.append({
            "question": question,
            "true_ans": true_ans,
            "extracted_ans": extracted_ans,
            "found_answer": found_ans,
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
        
    report_lines = [
        "=" * 55,
        f"  EVALUATION METRICS: {dataset_name.upper()} ({split})",
        "=" * 55,
        f"  Total Samples        : {num_samples}",
        f"  Correct Answers      : {correct_extractions}",
        f"  Accuracy             : {avg_accuracy*100:.2f}%",
        f"  Avg Baseline Tokens  : {total_baseline_tokens / num_samples:.1f}",
        f"  Avg Tokens Used      : {total_exit_tokens / num_samples:.1f}",
        f"  Token Savings        : {token_savings_pct:.2f}%",
        "=" * 55,
    ]
    report = "\n".join(report_lines)
    print(report)
    print(f"  Detailed logs  → {output_filename}")

    # Save metrics to txt
    txt_filename = output_filename.replace(".json", "_metrics.txt")
    with open(txt_filename, "w") as f:
        f.write(report + "\n")
    print(f"  Metrics report → {txt_filename}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Static Budget Pipeline")
    parser.add_argument("--controller", type=str, default="models/static_classifier.pkl", help="Path to static_classifier.pkl")
    parser.add_argument("--dataset", type=str, default="math_qa", choices=["all", "gsm8k", "math_qa"], help="Which dataset to evaluate. Options: all, gsm8k, math_qa")
    args = parser.parse_args()

    print("Initializing Qwen Model and Static Pipeline...")
    wrapper = HuggingFaceLLMWrapper(model_name="Qwen/Qwen2.5-Math-7B-Instruct")
    
    from static_budget_inference import StaticBudgetPipeline
    pipeline = StaticBudgetPipeline(wrapper, controller_path=args.controller)
        
    if args.dataset in ["all", "gsm8k"]:
        evaluate_on_dataset(pipeline, "gsm8k", split="test", num_samples=25)
    
    if args.dataset in ["all", "math_qa"]:
        # User only has math_qa train split uploaded to Kaggle local storage, so we evaluate strictly on unseen "train" slice
        evaluate_on_dataset(pipeline, "math_qa", split="train", num_samples=20)
