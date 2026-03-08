import json
import os
from tqdm import tqdm
from datasets import load_dataset
from models.llm_wrapper import HuggingFaceLLMWrapper
from early_exit_inference import EarlyExitPipeline
from main_generate_traces import get_question, get_true_answer, check_intermediate_correctness

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
                dataset = load_dataset(dataset_name)[split]
                
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
            
        exit_result = pipeline.generate_with_early_exit(prompt, max_steps=MAX_STEPS, step_tokens=STEP_TOKENS)
        
        actual_tokens_used = exit_result["total_tokens"]
        extracted_ans = exit_result["extracted_answer"].strip().lower()
        
        # Check correctness
        if true_ans == "": 
            is_correct = False
        elif extracted_ans == true_ans:
            is_correct = True # Exact match is always true 
        else:
            # Use the robust regex boundary matcher from the generation script to avoid "60" matching "600"
            is_correct = check_intermediate_correctness(extracted_ans, true_ans, dataset_name)
            
        if is_correct: correct_extractions += 1
            
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
    parser.add_argument("--controller", type=str, default="models/early_exit_controller.pt", help="Path to early_exit_controller.pt")
    parser.add_argument("--scaler", type=str, default="models/scaler.pt", help="Path to scaler.pt")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "gsm8k", "math_qa"], help="Which dataset to evaluate. Options: all, gsm8k, math_qa")
    args = parser.parse_args()

    print("Initializing Qwen Model and Pipeline...")
    wrapper = HuggingFaceLLMWrapper(model_name="Qwen/Qwen2.5-Math-7B-Instruct")
    # You can tweak the confidence threshold. Higher = safer but less token savings.
    pipeline = EarlyExitPipeline(wrapper, controller_path=args.controller, scaler_path=args.scaler, threshold=0.85)
    
    if args.dataset in ["all", "gsm8k"]:
        evaluate_on_dataset(pipeline, "gsm8k", split="test", num_samples=25)
    
    if args.dataset in ["all", "math_qa"]:
        # User only has math_qa train split uploaded to Kaggle local storage, so we evaluate strictly on unseen "train" slice
        evaluate_on_dataset(pipeline, "math_qa", split="train", num_samples=25)
