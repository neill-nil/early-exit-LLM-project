"""
StrategyQA Trace Generation using allenai/OLMo-3-7B-Think

This script generates reasoning traces for the StrategyQA dataset using OLMo-3-7B-Think
as the reasoning model, while using Qwen2.5-3B-Instruct as the LLM judge to check
whether the correct answer has been reached at each intermediate step.

Usage:
    python main_generate_traces_strategyqa.py --start 0 --end 20
"""

import json
import os
import re
import time
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from models.llm_wrapper import HuggingFaceLLMWrapper


def get_question_strategyqa(item) -> str:
    """Extract the question from a StrategyQA dataset item."""
    return item["question"]


def get_true_answer_strategyqa(item) -> str:
    """
    Extract the ground-truth answer from a StrategyQA dataset item.
    StrategyQA answers are boolean (True/False) -> we normalize to 'yes'/'no'.
    """
    ans = item.get("answer", "")
    if isinstance(ans, bool):
        return "yes" if ans else "no"
    # Handle string representations
    ans_str = str(ans).strip().lower()
    if ans_str in ("true", "yes"):
        return "yes"
    elif ans_str in ("false", "no"):
        return "no"
    return ans_str


def check_intermediate_correctness_llm(
    text: str, true_answer: str, question: str, judge_wrapper: HuggingFaceLLMWrapper
) -> bool:
    """
    Uses the Qwen judge model to decide if the reasoning trace has arrived at the
    correct yes/no answer as a FINAL conclusion.
    """
    # Fast path: if neither the answer word nor its boolean equivalent appears, skip the LLM call
    answer_variants = {true_answer}
    if true_answer == "yes":
        answer_variants.update(["yes", "true"])
    elif true_answer == "no":
        answer_variants.update(["no", "false"])

    text_lower = text.lower()
    if not any(variant in text_lower for variant in answer_variants):
        return False

    prompt = f"""You are a strict teacher grading a student's partial reasoning.

Problem: {question}
Correct Final Answer: {true_answer}

Student's current reasoning:
\"\"\"{text}\"\"\"

Task: Has the student explicitly arrived at and stated the final answer "{true_answer}" as their FINAL conclusion?
- If the student's reasoning is still ongoing and has not concluded, output NO.
- If the student mentions "{true_answer}" only as part of intermediate reasoning but hasn't concluded, output NO.
- If the student clearly concludes their reasoning with the final answer "{true_answer}", output YES.

Respond with exactly and ONLY the word "YES" or "NO". Do not explain.
"""

    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = judge_wrapper.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    response_text = judge_wrapper.generate(
        formatted_prompt, max_new_tokens=10, temperature=0.01
    )
    decision = response_text.strip().upper()

    return decision.startswith("YES")


def build_prompt_strategyqa(question: str, model_wrapper: HuggingFaceLLMWrapper) -> str:
    """
    Build the prompt for OLMo-3-7B-Think using its chat template.
    The model will naturally produce <think>...</think> reasoning before answering.
    """
    system_message = (
        "You are a helpful reasoning assistant. Think through the question step by step, "
        "then provide your final answer as either 'Yes' or 'No'."
    )

    user_message = (
        "Answer the following yes/no question by reasoning step-by-step.\n"
        "After your reasoning, clearly state your final answer as 'Yes' or 'No'.\n\n"
        "Example:\n"
        "Question: Do hamsters provide food for any animals?\n"
        "Answer: Hamsters are prey animals. Many predators, such as hawks, owls, and snakes, "
        "hunt and eat hamsters in the wild. Therefore, hamsters do provide food for other animals. "
        "The answer is Yes.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # Use the model's chat template so special tokens are correctly applied
    formatted = model_wrapper.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return formatted


def generate_traces_for_strategyqa(
    model_wrapper: HuggingFaceLLMWrapper,
    judge_wrapper: HuggingFaceLLMWrapper,
    output_path: str,
    split: str = "train",
    start_idx: int = 0,
    end_idx: int = 20,
):
    dataset_name = "ChilleD/StrategyQA"
    print(f"Loading StrategyQA {split} split...")
    try:
        local_path = "data/StrategyQA"
        if os.path.exists(local_path):
            dataset = load_from_disk(local_path)[split]
        else:
            print("Local StrategyQA data not found. Downloading from Hugging Face...")
            dataset = load_dataset(dataset_name)[split]
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Take the specific slice requested
    if start_idx is not None and end_idx is not None:
        end_idx = min(end_idx, len(dataset))
        print(f"Slicing dataset from index {start_idx} to {end_idx}...")
        dataset = dataset.select(range(start_idx, end_idx))

    results = []

    print(f"Starting trace generation for {len(dataset)} samples...")
    for idx, item in enumerate(tqdm(dataset)):
        real_idx = start_idx + idx

        question = get_question_strategyqa(item)
        true_answer_str = get_true_answer_strategyqa(item)

        # Build prompt using OLMo chat template
        prompt = build_prompt_strategyqa(question, model_wrapper)

        # Generate step-by-step to record the trace
        current_generation = ""
        steps_data = []

        max_steps = 60
        already_solved = False
        solved_at_step = -1

        for step_idx in range(max_steps):
            step_info = model_wrapper.generate_step(
                prompt=prompt,
                current_generation=current_generation,
                step_tokens_limit=20,
            )

            step_text = step_info["step_text"]
            current_generation += step_text

            # Bypass the LLM judge if we already found the answer in a previous step
            if already_solved:
                is_step_correct = True
            else:
                # Use the Qwen judge to check if the answer has been reached
                is_step_correct = check_intermediate_correctness_llm(
                    current_generation, true_answer_str, question, judge_wrapper
                )

                if is_step_correct:
                    already_solved = True
                    solved_at_step = step_idx

            # Record the trace step
            steps_data.append(
                {
                    "step_index": step_idx,
                    "text_added": step_text,
                    "cumulative_text": current_generation,
                    "num_tokens": step_info["num_tokens"],
                    "has_correct_answer": is_step_correct,
                }
            )

            # Stop if the model produced an EOS token
            if step_info["is_eos"]:
                break

        # Evaluate if the final answer was correct
        is_correct = already_solved

        results.append(
            {
                "id": real_idx,
                "question": question,
                "true_answer": true_answer_str,
                "model_answer": "",
                "is_correct": is_correct,
                "solved_at_step": solved_at_step,
                "full_generation": current_generation,
                "steps": steps_data,
            }
        )

        # Periodically save to avoid losing data
        if (idx + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Finished! Traces saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate reasoning traces for StrategyQA using OLMo-3-7B-Think"
    )
    parser.add_argument("--start", type=int, default=0, help="Starting index")
    parser.add_argument("--end", type=int, default=20, help="Ending index")
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    args = parser.parse_args()

    os.makedirs("data/traces", exist_ok=True)

    # Initialize OLMo-3-7B-Think as the reasoning model
    print("Initializing allenai/OLMo-3-7B-Think...")
    wrapper = HuggingFaceLLMWrapper(model_name="allenai/OLMo-3-7B-Think")

    # Initialize Qwen2.5-3B-Instruct as the LLM judge (same as math pipeline)
    print("Initializing Qwen2.5-3B-Instruct (Local Judge)...")
    judge_wrapper = HuggingFaceLLMWrapper(model_name="Qwen/Qwen2.5-3B-Instruct")

    filename = f"strategy-qa_{args.split}_traces_{args.start}_to_{args.end}.json"
    output_path = os.path.join("data/traces", filename)

    generate_traces_for_strategyqa(
        model_wrapper=wrapper,
        judge_wrapper=judge_wrapper,
        output_path=output_path,
        split=args.split,
        start_idx=args.start,
        end_idx=args.end,
    )