import re
from strategies.base import EarlyExitStrategy


class EarlyExitPipeline:
    """
    Model-agnostic early-exit inference wrapper.

    The stopping decision is fully delegated to a pluggable `strategy`
    (any subclass of EarlyExitStrategy), making it trivial to swap between
    Approach 1 (LearningBasedController) and Approach 3 (ConsistencyController)
    without changing the generation loop.

    Args:
        llm_wrapper: An initialised HuggingFaceLLMWrapper for the reasoning model.
        strategy: The early-exit strategy to use.
        max_steps: Maximum number of generation chunks before forced stop.
        step_tokens: Tokens generated per chunk / step.
        min_steps_before_check: Number of warmup steps before the strategy is
                                 consulted. Prevents premature exits on the
                                 first few tokens. (Incorporated from dev branch.)
    """

    def __init__(
        self,
        llm_wrapper,
        strategy: EarlyExitStrategy,
        max_steps: int = 60,
        step_tokens: int = 20,
        min_steps_before_check: int = 4,
    ):
        self.llm                  = llm_wrapper
        self.strategy             = strategy
        self.max_steps            = max_steps
        self.step_tokens          = step_tokens
        self.min_steps_before_check = min_steps_before_check

    # ------------------------------------------------------------------
    # Answer utilities (kept here so evaluate_pipeline.py can still call
    # pipeline.extract_final_answer / pipeline.has_clear_answer)
    # ------------------------------------------------------------------

    @staticmethod
    def has_clear_answer(text: str, prompt: str) -> bool:
        """Returns True if the generation contains a clearly formatted final answer."""
        text_lower   = text.lower()
        prompt_lower = prompt.lower()

        if "multiple-choice" in prompt_lower or "options:" in prompt_lower:
            if re.search(r"\b(?:option|answer)(?:\s+is)?\s+([a-e])\b", text_lower):
                return True
        elif "'yes' or 'no'" in prompt_lower or "logical question" in prompt_lower:
            if re.search(r"\b(yes|no|true|false)\b", text_lower):
                return True

        if re.search(r"\\boxed\{([^\}]+)\}", text) or ("the final answer is" in text_lower):
            return True

        return False

    @staticmethod
    def extract_final_answer(text: str, prompt: str) -> str:
        """
        Extracts the final answer from the (possibly truncated) reasoning trace.
        Uses the original prompt to determine the expected answer format.
        Merged from both branches — handles MathQA, StrategyQA, and standard math.
        """
        text_lower   = text.lower()
        prompt_lower = prompt.lower() if prompt else ""

        # Multiple-choice (MathQA style)
        if "multiple-choice" in prompt_lower or "options:" in prompt_lower:
            matches = re.findall(
                r"\b(?:option|answer)(?:\s+is)?\s+([a-e])\b", text_lower
            )
            if matches:
                return matches[-1]

        # Boolean QA (StrategyQA / HotpotQA style)
        elif "'yes' or 'no'" in prompt_lower or "logical question" in prompt_lower or "yes/no question" in prompt_lower:
            if "the answer is yes" in text_lower or "'yes'" in text_lower:
                return "yes"
            elif "the answer is no" in text_lower or "'no'" in text_lower:
                return "no"
            matches = re.findall(r"\b(yes|no)\b", text_lower)
            if matches:
                return matches[-1]
            return ""

        # Standard math – priority ladder
        # 1: \boxed{...}
        boxed = re.findall(r"\\boxed\{([^\}]+)\}", text)
        if boxed:
            nums = re.findall(r"-?\d+(?:\.\d+)?", boxed[-1])
            if nums:
                return nums[-1]
            return boxed[-1]

        # 2: "The final answer is …"
        if "the final answer is" in text_lower:
            part = text_lower.split("the final answer is")[-1]
            nums = re.findall(r"-?\d+(?:\.\d+)?", part)
            if nums:
                return nums[0]

        # 3: last number before the double-check section
        if "double-check:" in text_lower:
            main = text_lower.split("double-check:")[0]
            nums = re.findall(r"-?\d+(?:\.\d+)?", main)
            if nums:
                return nums[-1]

        # 4: absolute fallback — last number in text
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if nums:
            return nums[-1]

        return ""

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------

    def generate_with_early_exit(
        self,
        prompt: str,
        max_steps: int = None,
        step_tokens: int = None,
    ) -> dict:
        """
        Generates a response incrementally, checking for early exit after every
        chunk via `self.strategy.should_exit(...)`.

        Includes a warmup period (`min_steps_before_check`) during which the
        strategy is not consulted, preventing premature exits on opening tokens.

        Returns a dict with:
            generation, extracted_answer, early_exit_triggered,
            stopped_at_step, total_tokens
        """
        max_steps   = max_steps   or self.max_steps
        step_tokens = step_tokens or self.step_tokens

        current_generation   = ""
        early_exit_triggered = False
        stopped_at_step      = -1
        total_tokens         = 0

        # Reset per-question state on strategies that track history (e.g. ConsistencyController)
        if hasattr(self.strategy, "reset"):
            self.strategy.reset()

        for step_idx in range(max_steps):
            step_info = self.llm.generate_step(
                prompt=prompt,
                current_generation=current_generation,
                step_tokens_limit=step_tokens,
            )

            step_text           = step_info["step_text"]
            current_generation += step_text
            total_tokens       += step_info["num_tokens"]

            # Augment step_info with step index for strategies that need it
            step_info["step_idx"] = step_idx

            # ── Warmup: skip the first N steps before asking the strategy ──
            if step_idx < self.min_steps_before_check and not step_info["is_eos"]:
                print(f"  [Step {step_idx} | Tokens: {total_tokens}] Warmup — skipping early exit check")
                continue

            # ── Early Exit Check ──────────────────────────────────────────
            if self.strategy.should_exit(
                context=prompt,
                current_output=current_generation,
                step_info=step_info,
            ):
                early_exit_triggered = True
                stopped_at_step      = step_idx
                break

            if step_info["is_eos"]:
                break

        final_answer = self.extract_final_answer(current_generation, prompt)

        return {
            "generation":           current_generation,
            "extracted_answer":     final_answer,
            "early_exit_triggered": early_exit_triggered,
            "stopped_at_step":      stopped_at_step,
            "total_tokens":         total_tokens,
        }


if __name__ == "__main__":
    print(
        "EarlyExitPipeline loaded successfully.\n"
        "Initialise with a strategy:\n"
        "  from strategies.learning_based import LearningBasedController\n"
        "  from strategies.consistency    import ConsistencyController\n"
        "  pipeline = EarlyExitPipeline(wrapper, strategy=LearningBasedController(...))\n"
        "  pipeline = EarlyExitPipeline(wrapper, strategy=ConsistencyController(judge_wrapper))"
    )
