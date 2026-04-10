import json
import re

def _regex_check(extracted: str, true_ans: str) -> bool:
    try:
        pattern = r'(?<![\d.])' + re.escape(true_ans) + r'(?![\d.])'
        return bool(re.search(pattern, extracted, re.IGNORECASE))
    except re.error:
        return extracted == true_ans

def _check_math_qa_correctness(extracted_ans: str, true_ans: str) -> bool:
    extracted_ans = str(extracted_ans).strip().lower()
    true_ans = true_ans.strip().lower()
    
    match = re.search(r"option\s+([a-e])\s*\(value:\s*(.*?)\)", true_ans)
    if match:
        letter = match.group(1).strip()
        value = match.group(2).strip()
        val_match = re.search(r"[a-e]\s*\)\s*(.*)", value)
        if val_match:
            value = val_match.group(1).strip()
        if extracted_ans == letter or extracted_ans == value:
            return True
        return _regex_check(extracted_ans, value)
    return _regex_check(extracted_ans, true_ans)

with open("data/eval/eval_results_math_qa_train_baseline.json") as f:
    baseline = json.load(f)
with open("data/eval/eval_results_math_qa_train_newnew.json") as f:
    newnew = json.load(f)

baseline_correct = 0
for b in baseline:
    if _check_math_qa_correctness(b.get("extracted_ans", ""), b["true_ans"]):
        b["is_correct"] = True
        baseline_correct += 1
    else:
        b["is_correct"] = False
        
newnew_correct = 0
for n in newnew:
    if _check_math_qa_correctness(n.get("extracted_ans", ""), n["true_ans"]):
        n["is_correct"] = True
        newnew_correct += 1
    else:
        n["is_correct"] = False

print(f"Regraded Baseline Accuracy: {baseline_correct}/{len(baseline)}")
print(f"Regraded Newnew Accuracy  : {newnew_correct}/{len(newnew)}")

lost_accuracy = []
for i in range(min(len(baseline), len(newnew))):
    if baseline[i]["question"] != newnew[i]["question"]:
        print(f"Mismatch at {i}")
        break  # We assume strictly parallel indices
    if baseline[i]["is_correct"] and not newnew[i]["is_correct"]:
        lost_accuracy.append(i)

print(f"\nTotal questions Baseline was right but Newnew was wrong: {len(lost_accuracy)}")

print("\n--- Sample Analysis of Lost Accuracy ---")
for idx in lost_accuracy[:5]:
    b = baseline[idx]
    n = newnew[idx]
    print(f"\nQuestion: {b['question'].splitlines()[0][:100]}...")
    print(f"True Ans: {b['true_ans']}")
    print(f"Baseline Extracted: {b.get('extracted_ans', '')}")
    print(f"Newnew Extracted  : {n.get('extracted_ans', '')}")
    print(f"Newnew Exited Early? {n['early_exit']}")
    print(f"Newnew Tokens used : {n['tokens_used']}")
    print(f"Baseline Tokens    : {b['tokens_used']}")
    print(f"Newnew Full Gen    : {n.get('full_generation', '')[-200:]}")
