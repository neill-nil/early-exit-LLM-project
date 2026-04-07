"""
StrategyQA Evaluation Pipeline using allenai/OLMo-3-7B-Think

Evaluates the Early Exit Pipeline on StrategyQA using OLMo-3-7B-Think as the
reasoning model. The controller/MLP/embedder are model-agnostic and work
identically to the math pipeline.

Usage:
    python evaluate_pipeline_strategyqa.py --controller models/early_exit_controller.pt --scaler models/scaler.pt
"""

import json
import os
import re
from tqdm import tqdm
from datasets import load_dataset
from models.llm_wrapper import HuggingFaceLLMWrapper
from early_exit_inference import EarlyExitPipeline
from main_generate_traces_strategyqa import (
    get_question_strategyqa,
    get_true_answer_strategyqa,
)
from evaluate_pipeline import compute_dynamic_baseline


def get_few_shot_prompt_strategyqa(question: str, model_wrapper: HuggingFaceLLMWrapper) -> str:
    """
    Build the evaluation prompt for OLMo-3-7B-Think using its chat template.
    Same format as in trace generation for consistency.
    """
    system_message = (
        "You are a helpful reasoning assistant. Think through the question step by step, "
        "then provide your final answer as either 'Yes' or 'No'."
    )

    user_message = (
        "Answer the following yes/no question by reasoning step-by-step.\n"
        "After your reasoning, clearly state your final answer as 'Yes' or 'No'.\n\n"
        "Example:\n"
        "Question: Do hamsters provide food for any animals?\n"
        "Answer: Hamsters are prey animals. Many predators, such as hawks, owls, and snakes, "
        "hunt and eat hamsters in the wild. Therefore, hamsters do provide food for other animals. "
        "The answer is Yes.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    formatted = model_wrapper.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return formatted


def check_boolean_answer(extracted_ans: str, true_ans: str) -> bool:
    """Check if extracted answer matches the true boolean answer."""
    extracted = extracted_ans.strip().lower()
    true = true_ans.strip().lower()

    # Direct match
    if extracted == true:
        return True

    # Cross-match yes/true and no/false
    yes_set = {"yes", "true"}
    no_set = {"no", "false"}

    if extracted in yes_set and true in yes_set:
        return True
    if extracted in no_set and true in no_set:
        return True

    return False


def evaluate_on_strategyqa(
    pipeline: EarlyExitPipeline,
    model_wrapper: HuggingFaceLLMWrapper,
    split: str = "train",
    num_samples: int = 20,
    start_idx: int = 0,
):
    dataset_name = "ChilleD/StrategyQA"
    print(f"\nEvaluating pipeline on StrategyQA ({split} set)...")

    try:
        local_path = "data/StrategyQA"
        if os.path.exists(local_path):
            from datasets import load_from_disk
            dataset = load_from_disk(local_path)[split]
        else:
            dataset = load_dataset(dataset_name)[split]

        # Use a different slice than training to avoid overlap
        eval_start = start_idx
        eval_end = min(eval_start + num_samples, len(dataset))
        dataset = dataset.select(range(eval_start, eval_end))

    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return

    results = []
    total_baseline_tokens = 0
    total_exit_tokens = 0
    correct_extractions = 0

    # Empirical baseline for StrategyQA traces computed dynamically from traces:
    baseline_tokens = compute_dynamic_baseline(["strategy-qa"], 850.0)

    for idx, item in enumerate(tqdm(dataset)):
        question = get_question_strategyqa(item)
        true_ans = get_true_answer_strategyqa(item).strip().lower()

        # Build prompt using OLMo chat template
        prompt = get_few_shot_prompt_strategyqa(question, model_wrapper)

        MAX_STEPS = 30  # 30 steps * 40 tokens = max 1200 tokens
        STEP_TOKENS = 40  # 40 tokens/step: fewer MLP calls, faster evaluation

        exit_result = pipeline.generate_with_early_exit(
            prompt, max_steps=MAX_STEPS, step_tokens=STEP_TOKENS, min_steps_before_check=1
        )

        actual_tokens_used = exit_result["total_tokens"]
        extracted_ans = exit_result["extracted_answer"].strip().lower()

        # Check correctness using boolean matching
        is_correct = check_boolean_answer(extracted_ans, true_ans)

        if is_correct:
            correct_extractions += 1

        total_baseline_tokens += baseline_tokens
        total_exit_tokens += actual_tokens_used

        results.append(
            {
                "question": question,
                "true_ans": true_ans,
                "extracted_ans": extracted_ans,
                "is_correct": is_correct,
                "early_exit": exit_result["early_exit_triggered"],
                "tokens_used": actual_tokens_used,
                "true_baseline_tokens": baseline_tokens,
                "tokens_saved": baseline_tokens - actual_tokens_used,
                "full_generation": exit_result["generation"],
            }
        )

    actual_num = len(dataset)
    avg_accuracy = correct_extractions / max(1, actual_num)
    token_savings_pct = (
        ((total_baseline_tokens - total_exit_tokens) / max(1, total_baseline_tokens)) * 100
    )

    # Save detailed evaluation results
    os.makedirs("results", exist_ok=True)
    output_filename = f"results/eval_results_strategy_qa_{split}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)

    report_lines = [
        "=" * 55,
        f"  EVALUATION METRICS: STRATEGY_QA ({split})",
        "=" * 55,
        f"  Total Samples        : {actual_num}",
        f"  Correct Answers      : {correct_extractions}",
        f"  Accuracy             : {avg_accuracy*100:.2f}%",
        f"  Avg Baseline Tokens  : {baseline_tokens:.1f}",
        f"  Avg Tokens Used      : {total_exit_tokens / max(1, actual_num):.1f}",
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

    parser = argparse.ArgumentParser(
        description="Evaluate Early Exit Pipeline on StrategyQA"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="models/early_exit_controller.pt",
        help="Path to early_exit_controller.pt",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="models/scaler.pt",
        help="Path to scaler.pt",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=25,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=300,
        help="Start index in the dataset (to avoid training overlap)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Early exit confidence threshold",
    )
    args = parser.parse_args()

    print("Initializing OLMo-3-7B-Think for StrategyQA evaluation...")
    wrapper = HuggingFaceLLMWrapper(model_name="allenai/OLMo-3-7B-Think")

    pipeline = EarlyExitPipeline(
        wrapper,
        controller_path=args.controller,
        scaler_path=args.scaler,
        threshold=args.threshold,
    )

    evaluate_on_strategyqa(
        pipeline,
        model_wrapper=wrapper,
        split="train",
        num_samples=args.num_samples,
        start_idx=args.start,
    )