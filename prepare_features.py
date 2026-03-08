import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os

def load_traces(traces_dir="data/traces"):
    all_traces = []
    # Combine all json files from the traces directory
    for file_path in glob.glob(f"{traces_dir}/*.json"):
        print(f"Loading traces from {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
            
            # Filter out traces based on user conditions
            valid_traces = []
            for item in data:
                # 1. Skip if model wasn't able to answer it
                if not item.get("is_correct", False):
                    continue
                    
                # 2. Skip if it answered way too early (Step 0, 1, or 2)
                # This usually means coincidental number matching rather than actual solved math
                if item.get("solved_at_step", -1) <= 2:
                    continue
                    
                valid_traces.append(item)
                
            all_traces.extend(valid_traces)
            print(f"  -> Kept {len(valid_traces)} valid samples out of {len(data)} total.")
            
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
    traces = load_traces("data/traces") 
    if not traces:
        print("No traces found in data/traces/. Please generate traces first.")
        exit()
        
    X, y = extract_features(traces)
    
    print(f"\nFeature matrix X shape: {X.shape}")
    print(f"Labels y shape: {y.shape}")
    print(f"Positive class (Correct) ratio: {y.mean():.4f}")
    
    os.makedirs("data/features", exist_ok=True)
    
    print("Saving features and labels to data/features/...")
    np.save("data/features/X.npy", X)
    np.save("data/features/y.npy", y)
    print("Done! Features are ready for MLP training.")
