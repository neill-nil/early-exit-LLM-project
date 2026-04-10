from utils.base_strategy import EarlyExitStrategy
from typing import Dict, Any
import joblib
from sentence_transformers import SentenceTransformer

class StaticBudgetController(EarlyExitStrategy):
    """
    Approach 2: Difficulty-Aware Static Budgeting.
    Predicts expected difficulty before generation and assigns a fixed budget.
    """

    def __init__(self, budget_model_path: str = "models/static_classifier.pkl"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = joblib.load(budget_model_path)
        
        # Token budgets mapped to class indices 0: Easy, 1: Medium, 2: Hard
        self.class_to_budget = {
            0: 200,
            1: 400,
            2: 600
        }

    def predict_budget(self, context: str) -> int:
        """
        Predicts the number of tokens allowed for this input.
        """
        # Scikit-learn expects 2D array input, so we use string array
        emb = self.embedder.encode([context])
        pred_class = self.model.predict(emb)[0]
            
        return self.class_to_budget.get(pred_class, 600)

    def should_exit(self, context: str, current_output: str, step_info: Dict[str, Any] = None) -> bool:
        """
        Checks if the current generation length exceeds the predicted budget.
        """
        if step_info is None or "total_tokens" not in step_info or "budget" not in step_info:
            return False
            
        return step_info["total_tokens"] >= step_info["budget"]
