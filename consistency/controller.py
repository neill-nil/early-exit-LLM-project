import re
from utils.base_strategy import EarlyExitStrategy
from typing import Dict, Any, Optional


class ConsistencyController(EarlyExitStrategy):
    """
    Approach 3: Consistency-Based Early Exit.

    Uses a small local LLM judge (HuggingFaceLLMWrapper) to extract the
    intermediate answer from the current reasoning trace at every generation
    step.  If the extracted answer is identical and non-null for
    `consistency_threshold` consecutive steps, generation is halted early.

    Args:
        judge_wrapper: An initialised HuggingFaceLLMWrapper instance.
                       Recommended: Qwen/Qwen2.5-3B-Instruct.
        consistency_threshold: Number of consecutive identical non-NONE
                               answers required before exiting. Default: 2.
        dataset_name: Dataset name used to select the right extraction prompt
                      ("gsm8k", "math_qa", "strategy_qa", etc.).
        verbose: If True, prints extraction decisions each step.
    """

    def __init__(
        self,
        judge_wrapper,
        consistency_threshold: int = 2,
        dataset_name: str = "gsm8k",
        verbose: bool = True,
    ):
        self.judge = judge_wrapper
        self.consistency_threshold = consistency_threshold
        self.dataset_name = dataset_name.lower()
        self.verbose = verbose
        # Per-question state – reset via reset() between questions
        self.history: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Must be called between questions to clear the answer history."""
        self.history = []

    def should_exit(
        self,
        context: str,
        current_output: str,
        step_info: Dict[str, Any] = None,
    ) -> bool:
        """
        Evaluate whether the model has stabilised on an answer.

        Returns True when the last `consistency_threshold` extracted answers
        are identical *and* are not 'NONE'.
        """
        extracted = self._extract_answer_via_judge(context, current_output)
        self.history.append(extracted)

        step_num = len(self.history) - 1
        if self.verbose:
            print(f"  [Consistency | Step {step_num}] Extracted: {extracted}")

        # Need at least `consistency_threshold` entries to compare
        if len(self.history) < self.consistency_threshold:
            return False

        last_n = self.history[-self.consistency_threshold:]
        all_same = all(a == last_n[0] for a in last_n)
        not_none = last_n[0] != "NONE" and last_n[0] != ""

        if all_same and not_none:
            if self.verbose:
                print(
                    f"  >>> CONSISTENCY EXIT: answer '{last_n[0]}' "
                    f"stable for {self.consistency_threshold} steps."
                )
            return True

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_extraction_prompt(self, context: str, current_output: str) -> str:
        """
        Builds a strict extraction prompt tailored to the dataset type so the
        judge model produces a single-token / minimal-token answer.
        """
        if "math_qa" in self.dataset_name:
            task_instruction = (
                "The student is solving a multiple-choice math problem. "
                "Extract ONLY the final option letter (a/b/c/d/e) the student "
                "has concluded as the answer. "
                "If the student has not yet reached a final answer, "
                "output exactly: NONE"
            )
        elif "strategy" in self.dataset_name or "hotpot" in self.dataset_name:
            task_instruction = (
                "The student is answering a logical yes/no question. "
                "Extract ONLY the student's final concluded answer: "
                "'yes', 'no', or the exact entity name. "
                "If the student has not yet reached a final answer, "
                "output exactly: NONE"
            )
        else:  # GSM8K and other numerical benchmarks
            task_instruction = (
                "The student is solving a math problem. "
                "Extract ONLY the final numerical answer the student has "
                "concluded (e.g. '72' or '3.5'). "
                "Do NOT output any intermediate calculation results — "
                "only the student's definitive final answer. "
                "If the student has not yet written their final answer, "
                "output exactly: NONE"
            )

        prompt = (
            f"{task_instruction}\n\n"
            f"Problem:\n{context}\n\n"
            f"Student's current scratchpad:\n\"\"\"\n{current_output}\n\"\"\"\n\n"
            "Your answer (single value or NONE):"
        )
        return prompt

    def _extract_answer_via_judge(self, context: str, current_output: str) -> str:
        """
        Calls the resident judge model to extract the intermediate answer.
        Returns the answer string (uppercased / stripped) or 'NONE'.
        """
        raw_prompt = self._build_extraction_prompt(context, current_output)

        # Apply the instruct chat template so the 3B model acts as an assistant
        chat = [{"role": "user", "content": raw_prompt}]
        formatted_prompt = self.judge.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Temperature ~0 → near-deterministic; max 15 tokens is plenty
        response = self.judge.generate(
            formatted_prompt, max_new_tokens=15, temperature=0.01
        )

        cleaned = self._clean_judge_response(response)
        return cleaned

    @staticmethod
    def _clean_judge_response(raw: str) -> str:
        """
        Normalises the judge's raw output to a canonical form:
        - Strips whitespace and punctuation artefacts
        - Extracts the first meaningful token (number, letter, yes/no, NONE)
        - Returns 'NONE' if nothing meaningful is found
        """
        text = raw.strip().split("\n")[0].strip()   # Take only first line

        # If the judge explicitly says NONE / None / none → normalise
        if re.match(r"^none$", text, re.IGNORECASE):
            return "NONE"

        # Numerical answer (possibly decimal / negative)
        num_match = re.search(r"-?\d+(?:\.\d+)?", text)
        if num_match:
            return num_match.group(0)

        # Option letter for multiple-choice (a–e)
        letter_match = re.match(r"^([a-eA-E])\b", text)
        if letter_match:
            return letter_match.group(1).lower()

        # Boolean answers
        if re.match(r"^(yes|no|true|false)\b", text, re.IGNORECASE):
            return text.split()[0].lower()

        # Fallback: empty → NONE
        return "NONE" if not text else text[:30]   # Cap excessively long garbage
