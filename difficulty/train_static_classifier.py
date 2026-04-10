import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np

def extract_features_and_labels(traces_dir):
    print(f"Searching for trace files in '{traces_dir}'...")
    trace_files = glob.glob(os.path.join(traces_dir, "*_traces_*.json"))
    
    if not trace_files:
        print("No trace files found. Please ensure generation scripts were run and data is mounted.")
        return [], []
        
    questions = []
    labels = []
    
    for file_path in trace_files:
        print(f"Processing {file_path}...")
        try:
            with open(file_path, 'r') as f:
                traces = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"  [WARNING] Skipping {file_path} because it is corrupted or incomplete: {e}")
            continue
        except Exception as e:
            print(f"  [WARNING] Skipping {file_path} due to error: {e}")
            continue
            
        for ix, item in enumerate(traces):
            if not item.get("is_correct", False):
                continue
                
            steps = item.get("steps", [])
            solved_at_step = item.get("solved_at_step", -1)
            
            if solved_at_step == -1 and steps: # Fallback if solved_at_step is missing but is_correct is True
                solved_at_step = len(steps) - 1
            
            total_tokens = 0
            for i, step in enumerate(steps):
                total_tokens += step.get("num_tokens", 0)
                if i == solved_at_step:
                    break
                    
            question = item.get("question", "")
            
            # Binning logic
            if total_tokens <= 200:
                label = 0
            elif total_tokens <= 400:
                label = 1
            else:
                label = 2
                
            questions.append(question)
            labels.append(label)
            
    return questions, labels

import joblib

def train(traces_dir):
    questions, labels = extract_features_and_labels(traces_dir)
    
    if not questions:
        print("CRITICAL ERROR: No traces found. The model cannot be trained. Did you pass the correct --traces_dir?")
        return
        
    print(f"Found {len(questions)} successful traces for training.")
    
    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))
    
    print("Loading SentenceTransformer for text embedding...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Embedding questions...")
    X = embedder.encode(questions, show_progress_bar=True)
    y = np.array(labels)
    
    print(f"Training features shape: {X.shape}, labels shape: {y.shape}")
    
    from sklearn.linear_model import LogisticRegression
    # Logistic Regression is the most mathematically sound classifier for dense continuous embeddings
    # max_iter=1000 ensures it converges, class_weight='balanced' handles imbalanced Easy questions
    print("Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, multi_class='multinomial')
    
    model.fit(X, y)
    
    train_acc = model.score(X, y)
    print(f"\nTraining complete! Accuracy on full trace set: {train_acc:.4f}")
    
    os.makedirs("models", exist_ok=True)
    out_path = "models/static_classifier.pkl"
    joblib.dump(model, out_path)
    print(f"Model successfully saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_dir", type=str, default=".", help="Directory where *_traces_*.json are stored")
    args = parser.parse_args()
    
    import sys
    sys.argv = [sys.argv[0]] # clean for train()
    train(args.traces_dir)
