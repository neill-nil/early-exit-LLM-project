import torch
from sentence_transformers import SentenceTransformer
from .base import EarlyExitStrategy
from typing import Dict, Any
import re


class LearningBasedController(EarlyExitStrategy):
    """
    Approach 1: Learning-Based Early-Exit Controller.

    Migrated from EarlyExitPipeline (early_exit_inference.py) so the pipeline
    is strategy-agnostic.  Wraps the trained EarlyExitMLP + SentenceTransformer
    + StandardScaler state.

    Args:
        controller_path: Path to the saved MLP state dict (.pt).
        scaler_path: Path to the saved scaler dict (.pt) with 'mean'/'std'.
        threshold: Confidence threshold above which we trigger an early exit.
    """

    def __init__(
        self,
        controller_path: str = "models/early_exit_controller.pt",
        scaler_path: str = "models/scaler.pt",
        threshold: float = 0.85,
    ):
        from train_controller import EarlyExitMLP   # local import to avoid circular deps

        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading SentenceTransformer...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        print("Loading scaler...")
        scaler = torch.load(scaler_path, map_location=self.device)
        self.mean_scalars = scaler["mean"].to(self.device).float()
        self.std_scalars  = scaler["std"].to(self.device).float()

        print("Loading Early-Exit MLP Controller...")
        self.controller = EarlyExitMLP(input_dim=386).to(self.device)
        self.controller.load_state_dict(
            torch.load(controller_path, map_location=self.device)
        )
        self.controller.eval()

    # ------------------------------------------------------------------
    # EarlyExitStrategy interface
    # ------------------------------------------------------------------

    def should_exit(
        self,
        context: str,
        current_output: str,
        step_info: Dict[str, Any] = None,
    ) -> bool:
        """
        Returns True if the MLP confidence exceeds `threshold` AND the
        output already contains a clearly formatted answer.
        """
        step_idx    = step_info.get("step_idx", 0)   if step_info else 0
        num_tokens  = step_info.get("num_tokens", 20) if step_info else 20

        with torch.no_grad():
            # 1. Sentence embedding  →  [1, 384]
            emb = self.embedder.encode(
                current_output, convert_to_tensor=True, device=self.device
            )
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)

            # 2. Positional scalars  →  [1, 2]  (normalised)
            scalars = torch.tensor(
                [[step_idx, num_tokens]], dtype=torch.float32, device=self.device
            )
            scalars_norm = (scalars - self.mean_scalars) / self.std_scalars

            # 3. Concatenate  →  [1, 386]
            x = torch.cat([emb, scalars_norm], dim=1)

            prob = self.controller(x).item()

        print(
            f"  [MLP | Step {step_idx} | Tokens: {num_tokens}] "
            f"Controller Confidence: {prob*100:.2f}%"
        )

        if prob >= self.threshold:
            if self._has_clear_answer(current_output, context):
                print(
                    f"  >>> MLP EXIT TRIGGERED "
                    f"(Confidence {prob*100:.2f}% >= {self.threshold*100:.2f}%)"
                )
                return True
            else:
                print(
                    f"  >>> Controller confident ({prob*100:.2f}%), "
                    "but answer not fully printed yet. Continuing..."
                )

        return False

    # ------------------------------------------------------------------
    # Helpers (ported directly from old EarlyExitPipeline)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_clear_answer(text: str, prompt: str) -> bool:
        """Returns True if the generation already contains a clearly formatted final answer."""
        text_lower   = text.lower()
        prompt_lower = prompt.lower()

        if "multiple-choice" in prompt_lower or "options:" in prompt_lower:
            if re.search(r"\b(?:option|answer)(?:\s+is)?\s+([a-e])\b", text_lower):
                return True
        elif "'yes' or 'no'" in prompt_lower or "logical question" in prompt_lower:
            if re.search(r"\b(yes|no|true|false)\b", text_lower):
                return True

        # Universal checks
        if re.search(r"\\boxed\{([^\}]+)\}", text) or ("the final answer is" in text_lower):
            return True

        return False
