# 🦾 GPT-Style LLM from Scratch with GPT-2 Weight Loading and Fine-Tuning

A complete end-to-end implementation of a **GPT-style Large Language Model (LLM)** built entirely from scratch using PyTorch. This project covers the full pipeline from **tokenization and Transformer architecture design** to **pretraining, GPT-2 weight loading, text generation, checkpointing, SMS spam classification fine-tuning, instruction fine-tuning, and automated response evaluation**.

---

## 🚀 Highlights

* GPT-style decoder-only Transformer built entirely from scratch
* Custom tokenization and preprocessing pipeline
* Self-Attention, Causal Attention, and Multi-Head Attention implementations
* Custom LayerNorm, GELU, and Feed-Forward Network (FFN) modules
* Next-token prediction training pipeline
* GPT-2 pretrained weight loading for all official model variants
* Temperature scaling and Top-k sampling for text generation
* Model and optimizer checkpointing for training resumption
* Cross-Entropy Loss and Perplexity evaluation
* Fine-tuning on supervised classification tasks
* SMS Spam Detection using a fine-tuned GPT model
* Instruction fine-tuning pipeline with custom collate functions and ignore-index loss masking
* Supervised training loop with loss tracking, visualization, and timed evaluation
* Automated response scoring via a local Ollama model with integer-output pipeline
* Modular and extensible PyTorch implementation

---

## ⚙️ Preprocessing Pipeline

### 1. Load Raw Text

The training corpus is loaded as raw text.

### 2. Tokenization

A custom tokenizer is implemented using regular expressions.

```python
import re

tokens = re.findall(r"\b\w+\b", raw_text.lower())
```

### 3. Vocabulary Construction

Word-to-index mappings are created with support for an `<UNK>` token to handle unseen words.

### 4. Encoding & Decoding

Utility functions convert:

```text
Raw Text ↔ Token IDs
```

for both training and inference.

### 5. Embeddings

* Token Embeddings

* Learned Positional Embeddings

are combined before entering the Transformer.

---

## 🧠 Transformer Architecture

### Self-Attention

Computes relationships between tokens using scaled dot-product attention.

### Causal Attention

Applies a lower-triangular mask to ensure tokens only attend to past and current positions.

### Multi-Head Attention

Multiple attention heads run in parallel to capture information from different representation subspaces.

---

### Transformer Block

Each Transformer block follows:

```text
Multi-Head Attention
        ↓
Residual Connection
        ↓
Layer Normalization
        ↓
Feed-Forward Network
        ↓
Residual Connection
        ↓
Layer Normalization
```

Components include:

* Multi-Head Attention
* Feed-Forward Network (FFN)
* GELU Activation
* Layer Normalization
* Residual Connections

---

## 🏗️ GPT-Style Model

```text
Token Embeddings
        +
Positional Embeddings
        ↓
Stacked Transformer Blocks
        ↓
Final LayerNorm
        ↓
Linear Projection
        ↓
Vocabulary Logits
```

The model follows a decoder-only Transformer architecture similar to GPT.

---

## 🏋️ Pretraining

The model is trained using:

* Next-token prediction
* Cross-Entropy Loss
* AdamW Optimizer

Training utilities include:

* Training loss tracking
* Validation loss tracking
* Perplexity evaluation
* Learning curve visualization

---

## 🎲 Text Generation

Supports autoregressive generation using:

### Temperature Scaling

Controls randomness during generation.

* Low Temperature → More deterministic
* High Temperature → More creative

### Top-k Sampling

Restricts sampling to the top-k most likely tokens.

### Multinomial Sampling

Samples tokens according to probability distributions.

### EOS Handling

Stops generation when an end-of-sequence token is produced.

---

### Generation Pipeline

```text
Input Text
    ↓
Token IDs
    ↓
GPT Model
    ↓
Next-Token Probabilities
    ↓
Temperature Scaling
    ↓
Top-k Filtering
    ↓
Multinomial Sampling
    ↓
Generated Text
```

---

## 🔄 GPT-2 Weight Loading

Supports loading all official GPT-2 variants:

| Model        | Parameters |
| ------------ | ---------- |
| GPT-2 Small  | 124M       |
| GPT-2 Medium | 355M       |
| GPT-2 Large  | 774M       |
| GPT-2 XL     | 1.5B       |

The project includes utilities to:

* Download GPT-2 checkpoints
* Map weights into the custom GPT implementation
* Validate tensor shapes before assignment

Loaded components include:

* Token embeddings
* Positional embeddings
* Attention weights
* Feed-forward layers
* LayerNorm parameters
* Output head

---

## 💾 Checkpointing

Full training state (model weights, optimizer state, momentum statistics) persisted and restored via `torch.save` / `torch.load` for seamless training resumption.

---

## 📊 Fine-Tuning for SMS Spam Classification

The pretrained GPT model is fine-tuned on the SMS Spam Collection dataset for binary text classification.

### Dataset Preparation

* Downloaded from the UCI repository
* Loaded into a Pandas DataFrame
* Labels mapped:

```text
ham  → 0
spam → 1
```

### Class Balancing

The majority class (ham) is randomly downsampled to match the spam class size.

### Dataset Splits

The dataset is shuffled and divided into:

* Training Set
* Validation Set
* Test Set

### Custom Dataset

A custom `SpamDataset`:

* Tokenizes messages
* Truncates long sequences
* Pads shorter sequences
* Returns tensors suitable for GPT fine-tuning

### DataLoaders

* Training DataLoader (shuffle enabled)
* Validation DataLoader
* Test DataLoader

---

### Fine-Tuning Pipeline

```text
Raw SMS Text
    ↓
Tokenization
    ↓
Padding / Truncation
    ↓
SpamDataset
    ↓
DataLoader
    ↓
GPT Fine-Tuning
    ↓
Spam / Ham Prediction
```

---

## 📈 Results

### SMS Spam Classification

| Epoch | Train Accuracy | Validation Accuracy |
| ----- | -------------- | ------------------- |
| 1     | 70.0%          | 72.5%               |
| 2     | 82.5%          | 85.0%               |
| 3     | 90.0%          | 90.0%               |
| 4     | 100.0%         | 97.5%               |
| 5     | 100.0%         | 97.5%               |

**Training Time:** 15.82 minutes

The model successfully learned to distinguish spam from legitimate SMS messages while maintaining strong validation performance.

---

## 🎓 Instruction Fine-Tuning

The GPT model is further fine-tuned on an instruction-following dataset using supervised fine-tuning (SFT), enabling the model to respond to structured prompts.

### Dataset

- Source: `instruction-data.json` (1,100 entries) from the [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository
- Each entry contains an `instruction`, optional `input`, and `output`

### Dataset Splits

```text
85% Training | 10% Test | 5% Validation
```

### Prompt Format

Entries are formatted as structured prompt-response pairs:

```text
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
<instruction>

### Input:
<input>

### Response:
<output>
```

### Custom Dataset & Collation

`InstructionDataset` pre-tokenizes each formatted entry end-to-end. A custom collate function handles batching with:

- `<|endoftext|>` appended as sequence terminator
- Variable-length sequences padded to batch maximum
- Inputs: `tokens[:-1]` — Targets: `tokens[1:]` (next-token shift)
- Padding tokens in targets replaced with `ignore_index=-100` to exclude them from loss

Collate function pre-configured via `functools.partial` with `device` and `allowed_max_length=1024`. Loss correctness verified — cross-entropy with `ignore_index` confirmed to skip padding tokens consistently across batches.

### DataLoaders

- Training DataLoader — shuffle enabled, `drop_last=True`
- Validation DataLoader
- Test DataLoader

### Instruction Fine-Tuning Pipeline

```text
Raw JSON Entries
    ↓
Prompt Formatting (Instruction + Input + Response)
    ↓
Tokenization (tiktoken GPT-2)
    ↓
InstructionDataset
    ↓
Custom Collate (padding + ignore_index masking)
    ↓
DataLoader
    ↓
GPT Instruction Fine-Tuning
    ↓
Instruction-Following Response Generation
```

---

## 🏃 Training & Evaluation Pipeline

### Supervised Training Loop

- Reproducibility seed set before training
- Optimizer and device handling configured
- Training and validation losses collected at periodic intervals
- Tokens-seen counter tracked alongside epoch progress
- Wall-clock timing measured end-to-end

### Loss Visualization

Training vs. validation loss plotted against both epochs and tokens seen for full learning curve diagnostics.

### Response Generation & Comparison

Model responses generated for all test set entries, compared against reference outputs, and stored for downstream scoring.

### Checkpoint Saving

Trained model state dictionary saved with a sanitized filename for future reuse via `torch.save`.

---

## 🤖 Automated Response Evaluation (Ollama)

Instruction-following quality is evaluated automatically using a locally running Ollama model.

### Process Monitoring

`psutil` checks whether the Ollama process is active before any evaluation begins — raises an error immediately if not running.

### Local Model Query

`query_model` sends prompts to the local Ollama REST API via `urllib.request`:

- Fixed seed and `temperature=0` for deterministic, reproducible scoring
- Returns the model's raw text response

### Integer Scoring Pipeline

Evaluation responses are parsed to extract integer scores only, keeping results clean and analysis-ready. Conversion errors are caught and handled gracefully.

### Batch Scoring

`generate_model_scores` loops through all dataset entries, queries Ollama for a score per entry, and returns a collected results list.

### Evaluation Pipeline

```text
Test Dataset Entries
    ↓
Ollama Process Check (psutil)
    ↓
Prompt Construction
    ↓
query_model (urllib → Ollama REST API)
    ↓
Integer Score Extraction
    ↓
generate_model_scores (batch loop)
    ↓
Aggregated Score Results
```

---

## 📦 Tech Stack

### Language

* Python

### Deep Learning

* PyTorch

### Data Processing

* NumPy
* Pandas
* Regular Expressions (re)

### Visualization

* Matplotlib

### System & Evaluation

* psutil
* urllib (standard library)
* Ollama (local LLM server)

---

## 🔄 End-to-End Pipeline

```text
Raw Text
    ↓
Tokenization
    ↓
Vocabulary & Encoding
    ↓
Embeddings
    ↓
Transformer Layers
    ↓
Next Token Prediction
    ↓
Pretraining
    ↓
GPT-2 Weight Loading
    ↓
Checkpoint Save / Load
    ↓
Text Generation
    ↓
Fine-Tuning (SMS Spam Classification)
    ↓
Instruction Fine-Tuning (SFT)
    ↓
Instruction-Following Generation
    ↓
Automated Response Evaluation (Ollama)
```

---

## 👨‍💻 Author

Built as a foundational deep learning project to understand and implement the core mechanics behind modern Large Language Models (LLMs), moving beyond APIs to develop every major component from first principles using PyTorch.

✅ Project Status

Status: Completed

This project has been successfully implemented end-to-end, covering the complete lifecycle of a GPT-style Large Language Model from scratch, including:

## 🔗 Next Project
[Reasoning-LLMs](https://github.com/Arjun-Bhattarai/Reasoning-LLMs) — extending this foundation to train LLMs to reason step-by-step.
