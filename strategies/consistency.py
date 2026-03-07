from .base import EarlyExitStrategy
from typing import Dict, Any

class ConsistencyController(EarlyExitStrategy):
    """
    Approach 3: Consistency-Based Early Exit.
    Stops if the answer remains unchanged across consecutive steps.
    """

    def __init__(self, consistency_threshold: int = 2):
        self.consistency_threshold = consistency_threshold
        self.history = []

    def should_exit(self, context: str, current_output: str, step_info: Dict[str, Any] = None) -> bool:
        """
        Checks if the extracted answer has been stable for `consistency_threshold` steps.
        """
        # TODO: Extract answer from current_output
        # TODO: Compare with previous answers in self.history
        return False # Placeholder
