import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os

def load_traces(file_paths: list, dataset_label: str = ""):
    """Load and merge traces from an explicit list of files.
    
    - Skips files whose name contains 'fewshot'.
    - Deduplicates by question 'id' (first occurrence wins).
    - Skips samples where is_correct is False.
    - Does NOT apply the old solved_at_step <= 2 guard (LLM judge handles this).
    """
    all_traces = []
    seen_ids = set()
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        
        # Skip fewshot files
        if "fewshot" in file_name.lower():
            print(f"Skipping fewshot file: {file_name}")
            continue
        
        print(f"Loading traces from {file_name}...")
        with open(file_path, "r") as f:
            data = json.load(f)
        
        kept, skipped_incorrect, skipped_dup = 0, 0, 0
        for item in data:
            # Skip if model was wrong
            if not item.get("is_correct", False):
                skipped_incorrect += 1
                continue
            
            # Dedup by question id
            qid = item.get("id")
            if qid is not None and qid in seen_ids:
                skipped_dup += 1
                continue
            if qid is not None:
                seen_ids.add(qid)
            
            all_traces.append(item)
            kept += 1
        
        print(f"  -> Kept {kept} | Skipped (wrong): {skipped_incorrect} | Skipped (duplicate id): {skipped_dup}")
    
    print(f"\n[{dataset_label}] Total valid samples loaded: {len(all_traces)}\n")
    return all_traces


def extract_features(traces, model_name="all-MiniLM-L6-v2"):
    print(f"Loading embedding model: {model_name}...")
    embedder = SentenceTransformer(model_name)
    
    features = []
    labels = []
    
    print("Extracting features from traces...")
    for trace in tqdm(traces):
        steps = trace.get("steps", [])
        for step in steps:
            # Feature 1: The semantics of what has been generated so far
            text = step.get("cumulative_text", "")
            
            # Feature 2 & 3: Progression features
            step_idx = step.get("step_index", 0)
            num_tokens = step.get("num_tokens", 0)
            
            # Label
            is_correct = 1.0 if step.get("has_correct_answer", False) else 0.0
            
            features.append({
                "text": text,
                "step_idx": step_idx,
                "num_tokens": num_tokens
            })
            labels.append(is_correct)
            
    print("Computing embeddings in batches (this may take a minute)...")
    texts = [f["text"] for f in features]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    embeddings = embeddings.cpu().numpy()
    
    # Combine embeddings with scalar features
    X = []
    for i, f in enumerate(features):
        row = np.concatenate([
            embeddings[i],
            [f["step_idx"]], 
            [f["num_tokens"]]
        ])
        X.append(row)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    return X, y

if __name__ == "__main__":
    TRACES_DIR = "data/traces"
    
    # --- File lists (fewshot files are auto-skipped too, but we exclude them explicitly here) ---
    gsm8k_files = [
        f"{TRACES_DIR}/gsm8k_train_traces_110_to_310.json",
        f"{TRACES_DIR}/gsm8k_train_traces_100_to_110 (1).json",
        f"{TRACES_DIR}/gsm8k_train_traces_0_to_200-api.json",
    ]
    mathqa_files = [
        f"{TRACES_DIR}/math_qa_train_traces_5_to_150.json",
        f"{TRACES_DIR}/math_qa_train_traces_0_to_5.json",
    ]
    strategyqa_files = [
        f"{TRACES_DIR}/strategy-qa_train_traces_0_to_50.json",
        f"{TRACES_DIR}/strategy-qa_train_traces_50_to_100.json",
        f"{TRACES_DIR}/strategy-qa_train_traces_100_to_150.json",
    ]
    
    gsm8k_traces = load_traces(gsm8k_files, dataset_label="GSM8K")
    mathqa_traces = load_traces(mathqa_files, dataset_label="MathQA")
    strategyqa_traces = load_traces(strategyqa_files, dataset_label="StrategyQA")

    all_traces = gsm8k_traces + mathqa_traces + strategyqa_traces
    
    if not all_traces:
        print("No valid traces found. Please generate traces first.")
        exit()
    
    print(f"Combined dataset: {len(all_traces)} total samples ({len(gsm8k_traces)} GSM8K + {len(mathqa_traces)} MathQA + {len(strategyqa_traces)} StrategyQA)")
    
    X, y = extract_features(all_traces)
    
    print(f"\nFeature matrix X shape: {X.shape}")
    print(f"Labels y shape: {y.shape}")
    print(f"Positive class (Correct) ratio: {y.mean():.4f}")
    
    os.makedirs("data/features", exist_ok=True)
    
    print("Saving features and labels to data/features/...")
    np.save("data/features/X.npy", X)
    np.save("data/features/y.npy", y)
    print("Done! Features are ready for MLP training.")

