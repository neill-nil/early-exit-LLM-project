from abc import ABC, abstractmethod
from typing import Any, Dict

class EarlyExitStrategy(ABC):
    """
    Abstract base class for early-exit strategies.
    """

    @abstractmethod
    def should_exit(self, context: str, current_output: str, step_info: Dict[str, Any] = None) -> bool:
        """
        Determine whether to stop generation.
        
        Args:
            context: The original input prompt/question.
            current_output: The text generated so far.
            step_info: Additional info from the model (e.g., logits, entropy), if available.
            
        Returns:
            True if generation should stop, False otherwise.
        """
        pass
