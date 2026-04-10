import os
import json
from datasets import load_dataset
from main_generate_traces import get_question, get_true_answer

os.makedirs("data", exist_ok=True)

# Export GSM8K
print("Exporting GSM8K...")
gsm = load_dataset("gsm8k", "main", split="test")
gsm_data = []
for item in gsm.select(range(min(200, len(gsm)))):
    gsm_data.append({
        "question": get_question(item, "gsm8k"),
        "true_answer": get_true_answer(item, "gsm8k")
    })
with open("data/gsm8k_test.json", "w") as f:
    json.dump(gsm_data, f, indent=4)

# Export MathQA
print("Exporting MathQA...")
mathqa = load_dataset("math_qa", split="test", trust_remote_code=True)
mathqa_data = []
for item in mathqa.select(range(min(200, len(mathqa)))):
    mathqa_data.append({
        "question": get_question(item, "math_qa"),
        "true_answer": get_true_answer(item, "math_qa")
    })
with open("data/math_qa_test.json", "w") as f:
    json.dump(mathqa_data, f, indent=4)
print("Done!")
