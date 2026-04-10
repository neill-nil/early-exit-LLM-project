import json
import re

file_path = "data/eval/eval_results_math_qa_train_newnew.json"
with open(file_path, "r") as f:
    data = json.load(f)

correct = 0
total = len(data)

for item in data:
    true_ans_str = item["true_ans"].strip().lower()
    extracted_ans = str(item.get("extracted_ans", "")).strip().lower()
    
    # Extract letter and value from true_ans
    # format: "option d (value: 120)" or sometimes "option c (value: c ) 2)" -> wait, look at item 14.
    
    match = re.search(r"option\s+([a-e])\s*\(value:\s*(.*?)\)", true_ans_str)
    
    is_correct = False
    
    if match:
        letter = match.group(1).strip()
        value = match.group(2).strip()
        
        # Clean up value (sometimes it might have "a ) " prefix inside the value string)
        val_match = re.search(r"[a-e]\s*\)\s*(.*)", value)
        if val_match:
            value = val_match.group(1).strip()
            
        if extracted_ans == letter or extracted_ans == value:
            is_correct = True
    else:
        # fallback
        if extracted_ans in true_ans_str:
            is_correct = True
            
    # Also handle some edge cases
    if not is_correct and extracted_ans and true_ans_str.startswith("option " + extracted_ans):
        is_correct = True
        
    item["is_correct"] = is_correct
    if is_correct:
        correct += 1

with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"New Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
