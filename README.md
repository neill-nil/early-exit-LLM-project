# Adaptive Computation Control for Efficient Reasoning in LLMs

## Overview
Large Language Models (LLMs) often achieve state-of-the-art performance on complex mathematical and logical benchmarks through "Chain-of-Thought" (CoT) prompting. However, this autoregressive step-by-step generation leads to extensive "over-computation." Models predictably continue generating extensive reasoning chains—even after the correct answer has been successfully resolved internally—driven by learned conversational distributions rather than logical necessity.

This project implements a model-agnostic **Adaptive Reasoning Control System** that dynamically determines optimal stopping points during generation. By externalizing the stopping decision to a synchronous, lightweight neural network (MLP) trained on semantic embeddings of reasoning traces, we can trigger an "Early Exit." This safely truncates redundant token generation without modifying the base model's pre-trained weights.

### Key Achievements
- **GSM8K Benchmark:** Retained **92.0% logical accuracy** while saving **17.51%** of computing tokens against the unconstrained baseline.
- **MathQA Benchmark:** Maintained **76.0% accuracy** while reducing generation overhead by **14.8%**.
- **Model Agnosticism:** Operates entirely independently of the core reasoning model's parameter weights (`Qwen2.5-Math-7B-Instruct`), requiring absolutely no fine-tuning of the billion-parameter LLM matrices.

---

## Repository Structure & Working Pipeline

The repository is modularly structured into five distinct operational phases. Each pipeline script transitions the project from raw datasets to an independently-acting adaptive inference loop. The transition ensures decoupling between computationally heavy generation and lightweight neural training.

### 1. Data Collection & Trace Generation (`main_generate_traces.py`)
To train an external controller, we first need empirical data representing how the LLM natively reasons and when it actually solves problems. This script is responsible for building our foundational training dataset.
- Loads mathematical datasets (e.g., `gsm8k`, `math_qa`) via the HuggingFace `datasets` API.
- Solicits step-by-step generations from the LLM, chunked into boundaries of roughly 15-20 tokens per frame.
- **Verification Oracle:** At each 15-20 token step, a deterministic regex-driven oracle inspects the trajectory to see if the ground-truth mathematical answer has appeared within the context boundaries.
- The oracle intelligently handles different problem distributions, supporting strict exact-match arithmetic boundaries for GSM8K and alphabetical option extraction for MathQA.
- Saves detailed JSON traces containing cumulative text, specific token counts, and binary correctness labels for every single intermediate step. This results in our ground truth dataset.

### 2. Feature Embedding & Scaling (`prepare_features.py`)
Standard neural networks cannot ingest raw text strings. We mathematically translate the text trajectories into dense vector representations.
- Loads the raw JSON traces, dropping instances where the model catastrophically failed or hallucinated the answer prematurely (specifically traces solved in Step $\le$ 2).
- Processes every semantic step sequentially through a frozen `sentence-transformers/all-MiniLM-L6-v2` encoder. This specific sentence transformer was selected due to its incredibly fast processing speed, ensuring embedding latency does not bottleneck the reasoning pipeline.
- Concatenates the 384-dimensional dense semantic embedding with two positional scalars: the `step_index` and the local `num_tokens` generated per step. This combination allows the model to understand *where* in the reasoning path it currently resides.
- Outputs the finalized scaled feature matrix `X.npy` and binary label array `y.npy`.

### 3. Controller Training (`train_controller.py`)
We map the dense semantic embeddings to a binarized confidence state representing "solved" vs "unsolved".
- Implements a Multi-Layer Perceptron (MLP) architecture projecting from $\mathbb{R}^{386} \rightarrow \mathbb{R}^1$.
- **Architecture Structure:** `Linear(128)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(64)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(1)` $\rightarrow$ `Sigmoid`.
- Normalizes and scales the concatenated scalar parameters to prevent covariate shift across features with drastically different standard deviations.
- Defends against heavy dataset class-imbalance (where naturally 90% of intermediate sequence steps are 'incorrect' or 'unsolved') utilizing thresholded Precision/Recall monitoring across 20 epochs using an `AdamW` optimizer and a `BCEloss` penalty.
- Exports the highest-performing network weights to `models/early_exit_controller.pt` and the fitted standard scaler states to `models/scaler.pt`.

### 4. Adaptive Inference Execution (`early_exit_inference.py`)
This module provides the central user instantiation wrapper. It combines the foundational reasoning LLM (`HuggingFaceLLMWrapper`) and the trained Early-Exit Controller to execute dynamic, real-time computational halting.
- The LLM begins its Chain-of-Thought process, returning execution focus back to the parent script every ~20 tokens.
- The controller dynamically vector-embeds the running text history and passes it through the pre-loaded MLP. 
- If the resultant confidence breaches the user-defined safety threshold ($\tau \ge 0.85$):
  - A rigorous, format-enforcing protocol (`has_clear_answer`) verifies the model hasn't just printed the expected number in passing without proper conclusive framing (e.g., verifying `\boxed{}` formatting or "the final answer is" verbiage).
  - If formatted successfully, a physical runtime token interruption is passed to the loop, actively terminating GPU compute requirements and exiting mathematically early.
  - If unformatted, the continuous generation skips the exit instruction, effectively forcing the LLM to complete its current reasoning step.

### 5. Empirical Evaluation & Testing (`evaluate_pipeline.py`)
An automated orchestrator to thoroughly empirically quantify the Token Savings Ratio (TSR) and Relative Accuracy Retained (RAR).
- Re-initializes the `EarlyExitPipeline` over entirely unseen `test` datasets (bypassing caching risks).
- Handles robust offline-loading fallbacks for Kaggle computing limitations.
- Analyzes actual truncated token expenditure against the unconstrained historical trace empirical baselines (e.g., ~386 total tokens expected for GSM8K un-truncated).
- Captures absolute accuracy and token reduction distributions, intelligently exporting detailed evaluation JSON files reflecting precisely where the early-exit controller interrupted execution.

---

## Hardware and Execution Environments

Because autoregressive language modeling is fundamentally memory-bandwidth bound, the generation phase of this project is highly sensitive to hardware configurations.
- **Phase 1 Trace Generation:** Strictly requires sufficient VRAM to hold the `Qwen2.5-Math-7B-Instruct` matrices. Generation was natively orchestrated on Kaggle notebook instances utilizing dual NVIDIA T4 (15GB) GPUs configured with `bfloat16` weights. 
- **Phase 3 Controller Training:** The MLP training operates exclusively on pre-calculated NumPy vectors, resulting in extraordinarily small overhead. This execution phase runs comfortably on any standard CPU cluster within minutes.
- **Phase 4 Interfacing:** Embedding computation per step dynamically offloads to CUDA cores during testing to minimize bottleneck delays.

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/neill-nil/early-exit-LLM-project.git
cd early-exit-LLM-project

# Install dependencies 
# We explicitly recommend setting up a virtual environment (e.g. conda or venv)
conda create -n early-exit python=3.10
conda activate early-exit

# Install structural libraries
pip install -r requirements.txt
```

### Required Dependencies
The pipeline demands the following primary dependencies:
- `torch` (PyTorch for MLP propagation)
- `transformers` & `accelerate` (Hugging Face LLM interfacing)
- `sentence-transformers` (Execution of dense encoding)
- `datasets` (Automated benchmark test-train data loading)
- `scikit-learn` & `numpy` (Mathematical matrix scaling calculations)

---

## Usage Guide & Command Line Automation

To organically replicate the experimental behaviors from start to finish:

**Step 1: Generate reasoning behaviors on the training set:**
```bash
python main_generate_traces.py --dataset gsm8k --start 0 --end 200
```
*Note: Ensure High-RAM or GPU instances are attached. Results are exported synchronously inside the `/data/traces` directory.*

**Step 2: Vectorize the semantic text chunks:**
```bash
python prepare_features.py
```

**Step 3: Train the controller logic and fit standardization:**
```bash
python train_controller.py
```
*(Yields `early_exit_controller.pt` mapped inside the explicitly generated `/models/` directory)*

**Step 4: Execute automated pipeline empirical testing:**
To test exact threshold evaluations over independent dataset validations:
```bash
python evaluate_pipeline.py --dataset gsm8k --controller models/early_exit_controller.pt --scaler models/scaler.pt
```
To evaluate against mathematical multiple-choice problems:
```bash
python evaluate_pipeline.py --dataset math_qa --controller models/early_exit_controller.pt --scaler models/scaler.pt
```
