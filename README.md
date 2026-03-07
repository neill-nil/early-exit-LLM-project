# Adaptive Early-Exit Strategies for Efficient Reasoning in LLMs

This project investigates alternative early-exit strategies to determine when reasoning in Large Language Models (LLMs) can be safely terminated without modifying the underlying model.

## Approaches

1.  **Learning-Based Early-Exit Controller:** uses a lightweight neural classifier to decide whether generation should continue based on step-level features.
2.  **Difficulty-Aware Static Budgeting:** predicts the expected difficulty of a question before generation and assigns a fixed reasoning budget.
3.  **Consistency-Based Early Exit:** relies on behavioral stability; if the answer stops changing, generation terminates.

## Structure

*   `data/`: Data loading and processing logic.
*   `models/`: LLM interfaces (wrappers around HF/APIs).
*   `strategies/`: Implementation of the three early-exit strategies.
*   `evaluation/`: Metrics and evaluation scripts.
*   `utils/`: Helper functions.
*   `main.py`: Main entry point for running experiments.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Usage:
```bash
python main.py --strategy [learning|static|consistency] --dataset [gsm8k|strategyqa]
```
