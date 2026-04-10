"""
compute_metrics.py
------------------
Recalculates accuracy and token-saving metrics from saved evaluation JSON files.
Fixes the MathQA comparison bug where true_ans = "option d (value: 45)" but
extracted_ans = "d" — now correctly extracts the option letter before comparing.

Usage:
    python compute_metrics.py --results results/eval_results_math_qa_train.json --dataset math_qa
    python compute_metrics.py --results results/eval_results_gsm8k_test.json --dataset gsm8k
    python compute_metrics.py --results_dir results/  # processes all JSONs
"""

import json
import re
import os
import argparse
from datetime import datetime


def extract_option_letter(true_ans: str) -> str:
    """Extract just the letter from 'option d (value: 45)' → 'd'."""
    m = re.match(r'option\s+([a-e])', true_ans.strip(), re.IGNORECASE)
    return m.group(1).lower() if m else true_ans.strip().lower()


def is_correct(extracted_ans: str, true_ans: str, dataset_name: str) -> bool:
    """Correct comparison logic for each dataset type."""
    extracted = extracted_ans.strip().lower()
    true = true_ans.strip().lower()

    if not extracted or not true:
        return False

    if "math_qa" in dataset_name.lower():
        # Compare just the option letter
        true_letter = extract_option_letter(true)
        return extracted == true_letter or extracted.startswith(true_letter)

    else:  # GSM8K and others — word-boundary regex
        if extracted == true:
            return True
        try:
            pattern = r'(?<![\d.])' + re.escape(true) + r'(?![\d.])'
            return bool(re.search(pattern, extracted, re.IGNORECASE))
        except re.error:
            return extracted == true


def compute_metrics(results: list, dataset_name: str) -> dict:
    total = len(results)
    correct = 0
    total_baseline = 0
    total_exit_tokens = 0
    early_exits = 0

    corrected_results = []
    for item in results:
        is_c = is_correct(item["extracted_ans"], item["true_ans"], dataset_name)
        if is_c:
            correct += 1
        if item.get("early_exit", False):
            early_exits += 1

        total_baseline += item.get("true_baseline_tokens", 0)
        total_exit_tokens += item.get("tokens_used", 0)

        corrected_results.append({**item, "is_correct": is_c})

    accuracy = correct / total if total > 0 else 0
    token_savings_pct = ((total_baseline - total_exit_tokens) / total_baseline * 100) if total_baseline > 0 else 0
    avg_exit_tokens = total_exit_tokens / total if total > 0 else 0
    avg_baseline_tokens = total_baseline / total if total > 0 else 0
    early_exit_rate = early_exits / total if total > 0 else 0

    return {
        "dataset": dataset_name,
        "total_samples": total,
        "correct": correct,
        "accuracy_pct": accuracy * 100,
        "early_exit_rate_pct": early_exit_rate * 100,
        "avg_baseline_tokens": avg_baseline_tokens,
        "avg_exit_tokens": avg_exit_tokens,
        "token_savings_pct": token_savings_pct,
        "corrected_results": corrected_results,
    }


def format_report(metrics: dict) -> str:
    lines = [
        "=" * 55,
        f"  EVALUATION METRICS: {metrics['dataset'].upper()}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 55,
        f"  Total Samples          : {metrics['total_samples']}",
        f"  Correct Answers        : {metrics['correct']}",
        f"  Accuracy               : {metrics['accuracy_pct']:.2f}%",
        f"  Early Exit Rate        : {metrics['early_exit_rate_pct']:.2f}%",
        f"  Avg Baseline Tokens    : {metrics['avg_baseline_tokens']:.1f}",
        f"  Avg Tokens Used        : {metrics['avg_exit_tokens']:.1f}",
        f"  Token Savings          : {metrics['token_savings_pct']:.2f}%",
        "=" * 55,
    ]
    return "\n".join(lines)


def process_file(results_path: str, dataset_name: str):
    with open(results_path) as f:
        results = json.load(f)

    metrics = compute_metrics(results, dataset_name)
    report = format_report(metrics)
    print(report)

    # Save corrected JSON
    corrected_path = results_path.replace(".json", "_corrected.json")
    with open(corrected_path, "w") as f:
        json.dump(metrics["corrected_results"], f, indent=4)
    print(f"\n  Corrected JSON  → {corrected_path}")

    # Save metrics txt
    txt_path = results_path.replace(".json", "_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print(f"  Metrics TXT     → {txt_path}\n")

    return metrics


def infer_dataset(path: str) -> str:
    name = os.path.basename(path).lower()
    if "math_qa" in name:
        return "math_qa"
    elif "gsm8k" in name:
        return "gsm8k"
    return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, help="Path to single results JSON")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (gsm8k or math_qa). Auto-detected if not set.")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to process all eval JSON files")
    args = parser.parse_args()

    if args.results_dir:
        for fname in sorted(os.listdir(args.results_dir)):
            if fname.startswith("eval_results") and fname.endswith(".json") and "_corrected" not in fname:
                fpath = os.path.join(args.results_dir, fname)
                dataset = args.dataset or infer_dataset(fpath)
                process_file(fpath, dataset)
    elif args.results:
        dataset = args.dataset or infer_dataset(args.results)
        process_file(args.results, dataset)
    else:
        print("Provide --results <file> or --results_dir <dir>")
