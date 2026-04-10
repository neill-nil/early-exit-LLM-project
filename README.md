# Adaptive Computation Control for Efficient Reasoning in LLMs

## Overview
Large Language Models (LLMs) often achieve state-of-the-art performance on complex mathematical and logical benchmarks through "Chain-of-Thought" (CoT) prompting. However, this autoregressive step-by-step generation leads to extensive "over-computation." Models predictably continue generating extensive reasoning chains—even after the correct answer has been successfully resolved internally—driven by learned conversational distributions rather than logical necessity.

This project implements a model-agnostic **Adaptive Reasoning Control System** that dynamically determines optimal stopping points during generation. By externalizing the stopping decision to a synchronous, lightweight neural network (MLP) trained on semantic embeddings of reasoning traces, we can trigger an "Early Exit." This safely truncates redundant token generation without modifying the base model's pre-trained weights.

### Key Achievements
- **GSM8K:** Retained **90.00% accuracy** while saving **18.99%** of generated tokens.
- **MathQA:** Maintained **59.00% accuracy** while reducing generation overhead by **39.76%**.
- **StrategyQA:** Maintained **65.00% accuracy** while saving **25.02%** of generated tokens.
- **Model Agnosticism:** Operates entirely independently of the core reasoning model's parameter weights. Supports both `Qwen2.5-Math-7B-Instruct` (for math) and `OLMo-3-7B-Think` (for multi-hop logic), requiring absolutely no fine-tuning of the billion-parameter LLM weights.

---

## Repository Structure & Working Pipeline

The repository is modularly structured into distinct operational phases. Each pipeline script transitions the project from raw datasets to an independently-acting adaptive inference loop.

```
Early_exit_project/
├── main_generate_traces.py       # Phase 1: Trace generation with LLM judge
├── mlp/
│   ├── prepare_features.py       # Phase 2: Feature extraction & embedding
│   ├── train_controller.py       # Phase 3: MLP controller training
│   ├── controller.py             # MLP inference controller (strategy)
│   └── advanced_features.py      # Advanced lexical feature extraction
├── early_exit_inference.py       # Phase 4: Adaptive inference engine
├── evaluate_pipeline.py          # Phase 5: Full evaluation orchestrator
├── models/
│   ├── llm_wrapper.py            # HuggingFace LLM wrapper
│   ├── early_exit_controller.pt  # Standard MLP weights (386-dim)
│   ├── early_exit_controller_adv.pt  # Advanced MLP weights (390-dim)
│   ├── scaler.pt                 # Standard feature scaler
│   └── scaler_adv.pt             # Advanced feature scaler
├── consistency/
│   └── controller.py             # Consistency-based exit strategy
├── difficulty/
│   └── static_budget.py          # Static budget exit strategy
├── utils/
│   ├── base_strategy.py          # Abstract base class for strategies
│   └── evaluate_baseline.py      # Unconstrained baseline evaluation
├── data/
│   ├── traces/                   # Generated reasoning traces (JSON)
│   └── features/                 # Extracted feature matrices (NumPy)
├── results/                      # Evaluation metrics & detailed logs
└── requirements.txt
```

### 1. Data Collection & Trace Generation (`main_generate_traces.py`)
To train an external controller, we first need empirical data representing how the LLM natively reasons and when it actually solves problems.
- Loads datasets (`gsm8k`, `math_qa`, `ChilleD/StrategyQA`) via the HuggingFace `datasets` API.
- Solicits step-by-step generations from the base LLM, chunked into boundaries of ~20 tokens per step.
- **LLM Judge Verification:** At each 20-token step, a secondary `Qwen2.5-3B-Instruct` judge model evaluates whether the reasoning trace has arrived at the correct answer. This replaces simple regex matching to handle nuanced multi-step reasoning.
- Supports different problem domains: exact-match arithmetic for GSM8K, alphabetical option extraction for MathQA, and boolean Yes/No for StrategyQA.
- Saves detailed JSON traces containing cumulative text, token counts, and binary correctness labels for every intermediate step.

### 2. Feature Embedding & Scaling (`mlp/prepare_features.py`)
Standard neural networks cannot ingest raw text strings. We translate the text trajectories into dense vector representations.
- Loads the raw JSON traces, filtering out incorrectly-solved instances and deduplicating by question ID.
- Processes every step through a frozen `sentence-transformers/all-MiniLM-L6-v2` encoder, producing a 384-dimensional dense semantic embedding.
- Concatenates the embedding with two positional scalars: `step_index` and `num_tokens` generated per step, yielding a 386-dimensional standard feature vector.
- **Advanced mode** (`--use_advanced_features`): Appends four additional lexical diagnostics (chunk entropy, repetition ratio, step density, Jaccard similarity), yielding a 390-dimensional feature vector.
- Outputs the finalized feature matrix `X.npy` (or `X_adv.npy`) and binary label array `y.npy` (or `y_adv.npy`) to `data/features/`.

### 3. Controller Training (`mlp/train_controller.py`)
Maps the dense semantic embeddings to a binarized confidence state representing "solved" vs "unsolved."
- Implements a Multi-Layer Perceptron (MLP) architecture: `Linear(input_dim, 128)` → `BatchNorm1d` → `ReLU` → `Dropout(0.3)` → `Linear(128, 32)` → `BatchNorm1d` → `ReLU` → `Dropout(0.3)` → `Linear(32, 1)` → `Sigmoid`.
- Input dimension is 386 (standard) or 390 (advanced).
- Z-score normalizes scalar features (dimensions 384+) using training set statistics.
- Trains for 20 epochs using the `Adam` optimizer with `BCELoss`, saving the best checkpoint by validation loss.
- Exports weights to `models/early_exit_controller.pt` (or `_adv.pt`) and scaler states to `models/scaler.pt` (or `_adv.pt`).

### 4. Adaptive Inference Execution (`early_exit_inference.py`)
Combines the base reasoning LLM and the trained Early-Exit Controller for dynamic, real-time computational halting.
- The LLM begins its Chain-of-Thought process, returning focus back every ~20 tokens.
- The controller dynamically embeds the running text history and passes it through the pre-loaded MLP.
- If the confidence breaches the threshold ($\tau \ge 0.85$) and a clear answer format is detected (`\boxed{}`, "the final answer is", or option letters), a physical runtime interruption terminates generation early.
- **Hallucination pruning:** If confidence drops below 10% after step 20, generation is forcibly aborted to prevent degenerate loops.
- Supports modular strategy injection via the `strategy` parameter, allowing seamless switching between MLP, Consistency, and Static Budget approaches.

### 5. Empirical Evaluation & Testing (`evaluate_pipeline.py`)
An automated orchestrator to quantify Token Savings Ratio (TSR) and accuracy.
- Supports `--strategy mlp` (default) and `--strategy consistency` for switching between exit approaches.
- Handles three datasets: `gsm8k` (test split), `math_qa` (train split, offset 600+), `strategy_qa` (train split, offset 600+) to ensure zero overlap with training data.
- Automatically selects the appropriate base model: `Qwen2.5-Math-7B-Instruct` for math datasets, `OLMo-3-7B-Think` for StrategyQA.
- Computes baseline token expenditure dynamically from training traces and exports detailed evaluation JSON files.

---

## Hardware and Execution Environments

Because autoregressive language modeling is fundamentally memory-bandwidth bound, the generation phase is highly sensitive to hardware configurations.
- **Trace Generation:** Requires sufficient VRAM for `Qwen2.5-Math-7B-Instruct` + `Qwen2.5-3B-Instruct` (judge). Executed on Kaggle instances with NVIDIA A100 (40GB) / T4 (15GB) GPUs with `bfloat16` precision.
- **Controller Training:** The MLP trains exclusively on pre-calculated NumPy vectors. Runs comfortably on any CPU within minutes.
- **Inference:** Embedding computation per step offloads dynamically to available CUDA cores.

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/neill-nil/early-exit-LLM-project.git
cd early-exit-LLM-project

# Install dependencies 
conda create -n early-exit python=3.10
conda activate early-exit
pip install -r requirements.txt
```

### Required Dependencies
- `torch` — MLP propagation
- `transformers` & `accelerate` — HuggingFace LLM interfacing
- `sentence-transformers` — Dense semantic encoding
- `datasets` — Benchmark data loading
- `scikit-learn` & `numpy` — Feature scaling
- `tqdm` — Progress bars

---

## Usage Guide & Commands

### Step 1: Generate reasoning traces
```bash
python main_generate_traces.py --dataset gsm8k --start 0 --end 500
python main_generate_traces.py --dataset math_qa --start 0 --end 500
python main_generate_traces.py --dataset ChilleD/StrategyQA --start 0 --end 500
```
> **Note:** Requires GPU instances. Traces are saved to `data/traces/`.

### Step 2: Extract features from traces
```bash
# Standard features (386-dim)
python mlp/prepare_features.py

# Advanced features (390-dim, includes entropy/repetition metrics)
python mlp/prepare_features.py --use_advanced_features
```

### Step 3: Train the MLP controller
```bash
# Standard model
python mlp/train_controller.py

# Advanced model
python mlp/train_controller.py --use_advanced_features
```
> Outputs: `models/early_exit_controller.pt` and `models/scaler.pt` (or `_adv` variants).

### Step 4: Evaluate the pipeline

**MLP strategy (default) on all datasets:**
```bash
python evaluate_pipeline.py --dataset all --num_samples 100 --strategy mlp
```

**MLP with advanced controller:**
```bash
python evaluate_pipeline.py --dataset all --num_samples 100 --strategy mlp \
    --controller models/early_exit_controller_adv.pt \
    --scaler models/scaler_adv.pt
```

**Per-dataset evaluation:**
```bash
# GSM8K (test split)
python evaluate_pipeline.py --dataset gsm8k --num_samples 100 --strategy mlp

# MathQA (train split, auto-offset to index 600+)
python evaluate_pipeline.py --dataset math_qa --num_samples 100 --strategy mlp

# StrategyQA (train split, auto-offset to index 600+)
python evaluate_pipeline.py --dataset strategy_qa --num_samples 100 --strategy mlp
```

**Consistency strategy:**
```bash
python evaluate_pipeline.py --dataset gsm8k --num_samples 30 --strategy consistency
```

### Key CLI Arguments for `evaluate_pipeline.py`
| Argument | Default | Description |
|---|---|---|
| `--strategy` | `mlp` | Exit strategy: `mlp` or `consistency` |
| `--dataset` | `all` | Dataset: `all`, `gsm8k`, `math_qa`, or `strategy_qa` |
| `--num_samples` | `25` | Number of evaluation samples |
| `--start` | `0` | Start index (auto-set to 600 for math_qa/strategy_qa) |
| `--controller` | `models/early_exit_controller.pt` | Path to MLP weights |
| `--scaler` | `models/scaler.pt` | Path to scaler weights |
| `--threshold` | `0.85` | MLP confidence threshold |

---

## Models & Checkpoints

Pre-trained controller weights and scalers are available at:
- **HuggingFace:** [Neillmate/early-exit-controller](https://huggingface.co/Neillmate/early-exit-controller/tree/main)

| File | Description | Input Dim |
|---|---|---|
| `early_exit_controller.pt` | Standard MLP controller | 386 |
| `early_exit_controller_adv.pt` | Advanced MLP controller (with lexical features) | 390 |
| `scaler.pt` | Scalar feature normalizer (standard) | — |
| `scaler_adv.pt` | Scalar feature normalizer (advanced) | — |
