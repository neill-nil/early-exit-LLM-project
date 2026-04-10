import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import copy

class EarlyExitMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Input: sentence embedding (e.g. 384) + step_idx + num_tokens = 386
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid() # We want a probability between 0 and 1
        )
        
    def forward(self, x):
        return self.network(x).squeeze()

def train_and_evaluate(use_advanced=False):
    print("Loading extracted features...")
    try:
        suffix = "_adv" if use_advanced else ""
        X = np.load(f"data/features/X{suffix}.npy")
        y = np.load(f"data/features/y{suffix}.npy")
    except FileNotFoundError:
        print("Features not found! Please run prepare_features.py first.")
        return
        
    # Split into train/test (80/20)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = torch.FloatTensor(X[train_idx]), torch.FloatTensor(y[train_idx])
    X_test, y_test = torch.FloatTensor(X[test_idx]), torch.FloatTensor(y[test_idx])
    
    # Standard sentence embeddings are the first 384 dimensions.
    # Everything after that are scalar features that need normalization.
    scalar_start_idx = 384
    mean_scalars = X_train[:, scalar_start_idx:].mean(dim=0)
    std_scalars = X_train[:, scalar_start_idx:].std(dim=0) + 1e-8
    
    X_train[:, scalar_start_idx:] = (X_train[:, scalar_start_idx:] - mean_scalars) / std_scalars
    X_test[:, scalar_start_idx:] = (X_test[:, scalar_start_idx:] - mean_scalars) / std_scalars
    
    os.makedirs("models", exist_ok=True)
    suffix = "_adv" if use_advanced else ""
    torch.save({"mean": mean_scalars, "std": std_scalars}, f"models/scaler{suffix}.pt")
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Scalar feature dimensions scaled: {X.shape[1] - 384}")
    
    # Dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    input_dim = X.shape[1]
    model = EarlyExitMLP(input_dim)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epochs = 20
    best_loss = float('inf')
    best_model_weights = None
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        test_loss = 0.0
        correct_preds = total_preds = 0
        true_pos = false_pos = false_neg = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                test_loss += loss.item()
                
                # Binarize predictions
                binary_preds = (preds >= 0.5).float()
                correct_preds += (binary_preds == batch_y).sum().item()
                total_preds += len(batch_y)
                
                true_pos += ((binary_preds == 1) & (batch_y == 1)).sum().item()
                false_pos += ((binary_preds == 1) & (batch_y == 0)).sum().item()
                false_neg += ((binary_preds == 0) & (batch_y == 1)).sum().item()
                
        train_loss /= max(1, len(train_loader))
        test_loss /= max(1, len(test_loader))
        accuracy = correct_preds / max(1, total_preds)
        
        precision = true_pos / max(1, (true_pos + false_pos))
        recall = true_pos / max(1, (true_pos + false_neg))
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            
    print("\nTraining complete! Saving best model...")
    suffix = "_adv" if use_advanced else ""
    torch.save(best_model_weights, f"models/early_exit_controller{suffix}.pt")
    print(f"Model saved to models/early_exit_controller{suffix}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_advanced_features", action="store_true", help="Train model using the advanced feature set (_adv.npy files).")
    args = parser.parse_args()
    
    train_and_evaluate(use_advanced=args.use_advanced_features)
