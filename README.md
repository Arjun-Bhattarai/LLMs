# 🦾 GPT-Style LLM from Scratch — GPT-2 Architecture, Weight Loading & Continued Pretraining

A complete end-to-end implementation of a **GPT-style decoder-only Large Language Model (LLM)** built with PyTorch from first principles.

This project focuses on understanding the internal mechanics of modern LLMs rather than treating them as black-box APIs. It implements the Transformer architecture, GPT-style tokenization and generation, GPT-2 weight loading, checkpointing, continued pretraining, supervised fine-tuning, instruction fine-tuning, and automated evaluation.

The project ultimately takes the custom GPT-2-compatible architecture, loads **OpenAI GPT-2 124M pretrained weights**, and performs **continued pretraining on WikiText-103**.

> This repository is an educational GPT-2 implementation and experimentation
> project. It is designed to make the training pipeline understandable and
> reproducible, not to compete with production-scale foundation models.

## 📚 Contents

* [Highlights](#-highlights)
* [Quick Start](#-quick-start)
* [Architecture](#-architecture)
* [Training Experiments](#-training-experiments)
* [Kaggle GPU Training](#-kaggle-gpu-continued-training)
* [Text Generation](#-text-generation)
* [Fine-Tuning](#-fine-tuning-sms-spam-classification)
* [Testing](#-testing)
* [Project Structure](#-project-structure)
* [Project Status](#-project-status)

## 🚀 Quick Start

### Requirements

* Python 3.11 or newer
* PyTorch 2.0 or newer
* A CUDA-capable GPU is recommended for GPT-2 training
* `tiktoken` for GPT-2 BPE tokenization

Install the project dependencies from the repository environment, or install
the packages required by the specific workflow you want to run. The complete
training environment is substantially larger than the minimal inference
requirements.

### Choose a Workflow

| Goal | Recommended entry point |
| --- | --- |
| Learn the components | `notebooks/` |
| Run continued pretraining locally | `scripts/train.py` |
| Continue a Kaggle checkpoint | `continue_gpt2_training.ipynb` |
| Generate text | `src/llm/inference/` and `scripts/inference.py` |
| Run the test suite | `tests/` |

The repository includes both instructional notebooks and reusable Python
modules. The notebooks explain the concepts step by step; the modules and
scripts provide the repeatable implementation.

---

## 🚀 Highlights

* 🧠 GPT-style decoder-only Transformer implemented from scratch in PyTorch
* 🔢 GPT-2 BPE tokenization using `tiktoken`
* ⚡ Scaled dot-product self-attention
* 🔒 Causal attention masking
* 🧩 Multi-head attention
* 📐 Custom LayerNorm implementation
* 🔥 GELU activation
* 🏗️ GPT-style Feed-Forward Networks
* 🔁 Residual connections
* 📍 Learned positional embeddings
* 🎯 Next-token prediction
* 📦 GPT-2 pretrained weight loading
* 🔄 Continued pretraining on WikiText-103
* 💾 Full model + optimizer checkpointing and training resumption
* 🎲 Temperature, Top-k and multinomial sampling
* 📊 Loss and perplexity evaluation
* 📨 SMS spam classification fine-tuning
* 🎓 Instruction fine-tuning with supervised learning
* 🧮 Padding-aware `ignore_index` loss masking
* 🤖 Automated response evaluation using a local Ollama model
* 🧪 Deterministic baseline-vs-trained model evaluation
* 🧱 Modular PyTorch architecture with tests

---

# 🧪 Training Experiments

This project contains two related experiments:

| Experiment | Initialization | Dataset | Status |
| --- | --- | --- | --- |
| Original baseline | Pretrained GPT-2 124M | WikiText-103 | Completed at 10K steps |
| Kaggle continuation | Original 10K checkpoint | WikiText-103 | Verified at 20K; 30K/50K are continuation targets |

The model is not trained from random initialization in the main experiment.
It starts from pretrained GPT-2 weights and adapts them to the WikiText-103
corpus through continued pretraining.

The reported results are intentionally separated by experiment. The original
10K run provides the documented baseline comparison, while the Kaggle section
records the later continuation checkpoint and its validation loss.

---

# 🧠 Architecture

The model follows a **decoder-only Transformer architecture inspired by GPT-2**.

```text
Input Text
    │
    ▼
GPT-2 BPE Tokenization
    │
    ▼
Token IDs
    │
    ├───────────────┐
    ▼               ▼
Token Embedding   Positional Embedding
    │               │
    └───────┬───────┘
            ▼
      Embedding Sum
            │
            ▼
    ┌──────────────────────┐
    │ Transformer Block ×12 │
    │                      │
    │ LayerNorm            │
    │    ↓                 │
    │ Causal Multi-Head    │
    │ Self-Attention       │
    │    ↓                 │
    │ Residual Connection  │
    │    ↓                 │
    │ LayerNorm            │
    │    ↓                 │
    │ Feed-Forward Network │
    │    ↓                 │
    │ Residual Connection  │
    └──────────────────────┘
            │
            ▼
       Final LayerNorm
            │
            ▼
      Output Projection
            │
            ▼
      Vocabulary Logits
            │
            ▼
       Next Token
```

### GPT-2 124M Configuration

| Component           | Configuration |
| ------------------- | ------------: |
| Parameters          |   163,037,184 |
| Transformer Layers  |            12 |
| Attention Heads     |            12 |
| Embedding Dimension |           768 |
| Head Dimension      |            64 |
| Context Length      |          1024 |
| Vocabulary Size     |        50,257 |
| FFN Dimension       |         3,072 |
| Dropout             |           0.1 |
| QKV Bias            |       Enabled |

---

# ⚙️ Tokenization & Preprocessing

The project initially explores the fundamentals of language-model preprocessing using a custom tokenizer.

### Basic Tokenization

```python
import re

tokens = re.findall(
    r"\b\w+\b",
    raw_text.lower()
)
```

The project then progresses to the **GPT-2 BPE tokenizer** using `tiktoken` for GPT-2-compatible training, evaluation, and generation.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
```

The GPT-2 tokenizer provides:

* 50,257 vocabulary tokens
* Subword-level tokenization
* GPT-2-compatible token IDs
* Consistent encoding across training and inference

---

# 🏗️ Transformer Implementation

The Transformer is implemented modularly using PyTorch.

## Self-Attention

The implementation computes scaled dot-product attention:

```text
Q = Query
K = Key
V = Value

Attention(Q,K,V)
        │
        ▼
   QKᵀ / √d
        │
        ▼
   Causal Mask
        │
        ▼
      Softmax
        │
        ▼
        × V
```

## Causal Attention

A lower-triangular mask prevents a token from attending to future tokens.

```text
Token 1 → ✓
Token 2 → ✓ ✓
Token 3 → ✓ ✓ ✓
Token 4 → ✓ ✓ ✓ ✓
```

This allows the model to perform autoregressive next-token prediction.

## Multi-Head Attention

The model uses multiple attention heads in parallel:

```text
Input
  │
  ├── Head 1
  ├── Head 2
  ├── Head 3
  ├── ...
  └── Head 12
       │
       ▼
   Concatenate
       │
       ▼
   Projection
```

Different heads can learn different relationships between tokens.

---

# 🔄 GPT-2 Weight Loading

One of the major components of this project is loading **official pretrained GPT-2 weights into the custom PyTorch implementation**.

The loader maps the original GPT-2 checkpoint tensors into the custom architecture.

```text
Official GPT-2 Checkpoint
          │
          ▼
   Weight Conversion
          │
          ▼
Custom PyTorch GPT Model
```

The loader handles:

* Token embeddings
* Positional embeddings
* Attention Q/K/V projections
* Attention output projections
* Feed-forward layers
* LayerNorm scale and bias
* Output projection

The project also validates tensor shapes during weight assignment.

### Supported GPT-2 Configurations

| Model        | Parameters |
| ------------ | ---------: |
| GPT-2 Small  |       124M |
| GPT-2 Medium |       355M |
| GPT-2 Large  |       774M |
| GPT-2 XL     |       1.5B |

The primary experiment in this repository uses **GPT-2 124M**.

---

# 🏋️ Continued Pretraining on WikiText-103

The main training experiment performs **continued pretraining**, rather than training a 124M-parameter language model from random initialization.

The training lineage is:

```text
Official OpenAI GPT-2 124M
          │
          ▼
Pretrained GPT-2 Weights
          │
          ▼
WikiText-103 Continued Pretraining
          │
          ▼
Checkpoint Resume
          │
          ▼
10,000 Optimization Steps
```

### Dataset

WikiText-103 was used as the continued-pretraining corpus.

```text
Training tokens:   119,085,168
Validation tokens:     249,749
Context length:           1,024
```

### Training Objective

The model is trained using autoregressive next-token prediction:

```text
Input:
The history of

Target:
history of the world
```

The training objective is standard cross-entropy language-model loss.

### Training Configuration

* Model: GPT-2 124M
* Optimizer: AdamW
* Learning rate: `1e-5`
* Context length: 1024
* Batch size: 1
* Mixed precision: FP16 AMP
* Dataset: WikiText-103
* Maximum training steps: 10,000

The final checkpoint contains:

* Model state
* Optimizer state
* Training metadata
* Validation loss
* Training loss
* Configuration

---

# ☁️ Kaggle GPU Continued Training

The 10K-step checkpoint was continued on Kaggle using a Tesla T4 GPU. The
Kaggle notebook uses the same GPT-2 tokenizer, model configuration, dataset
preprocessing, and optimizer settings as the original experiment.

## Kaggle Dataset Contents

The project dataset contains the source code, prepared WikiText training and
validation files, GPT-2 124M files, the original checkpoint, and the project
configuration files.

The continued-training checkpoint is stored separately because PyTorch
checkpoint files are large and are ignored by Git. The 20K checkpoint is
named `gpt2_wikitext_20k.pt`.

The checkpoint should be attached to the Kaggle notebook as a Kaggle Dataset,
not committed to GitHub.

## Kaggle Environment

Enable a GPU in the Kaggle notebook settings and install `tiktoken`. The
project and checkpoint are mounted read-only under `/kaggle/input`, while
training outputs must be written to `/kaggle/working`. If Kaggle assigns a
different dataset slug, locate the mounted `gpt2_wikitext_20k.pt` file and use
that path as the resume checkpoint.

## Resuming Training

The 20K checkpoint was created by resuming the original 10K checkpoint. Its
verified metadata is:

| Metric | Value |
| --- | ---: |
| Model | GPT-2 124M |
| Global step | 19,999 |
| Completed steps | 20,000 |
| Epoch | 5 |
| Training loss | 3.0135 |
| Validation loss | 2.7882 |
| Context length | 1,024 |
| Batch size | 1 |
| Learning rate | `1e-5` |
| Weight decay | `0.1` |
| Precision | FP16 AMP on CUDA |

To continue from 20K to 30K, recreate the model and optimizer with the same
configuration and pass the 20K checkpoint to the existing training loop. For
a direct 20K-to-50K run, use `gpt2_wikitext_50k.pt` as the output filename and
set the total target to 50,000 steps.

The target is a total global step, not the number of additional steps. Keep
the 20K checkpoint attached as a backup until the longer run has loaded it
successfully.

## Tokens Processed

With a batch size of 1 and a context length of 1,024, each optimization step
processes approximately 1,024 token positions. This corresponds to
approximately 20.48 million processed token positions at 20,000 steps and
51.2 million at 50,000 steps.

These are processed token positions, not necessarily unique tokens. The
training corpus is reused across optimization steps.

## Saving Checkpoints

Checkpoint files must be saved under `/kaggle/working`, such as
`gpt2_wikitext_30k.pt` or `gpt2_wikitext_50k.pt`. After a run finishes,
verify the saved global step and validation loss before closing the session.

Publish the resulting `.pt` file as a Kaggle Dataset or store it locally.
Do not commit it to GitHub; the repository `.gitignore` excludes `*.pt`, and
GitHub's regular file limit is much smaller than these checkpoints.

## Generating Text from a Trained Checkpoint

After training, load the saved model state with the GPT-2 tokenizer and use
the repository's generation utilities to generate text from a prompt. A
temperature of 0.8 and top-k sampling of 50 are suitable starting settings.
The model is trained for autoregressive continuation. It is not an
instruction-following chatbot unless it is additionally fine-tuned on
instruction-response data.

---

# 📊 Continued Pretraining Results

The strongest experiment compares the original pretrained GPT-2 against the **10K-step WikiText-103 continued-pretrained model** on the same deterministic validation batches.

### WikiText-103 Validation

| Metric             | Original GPT-2 | 10K Continued Pretraining |
| ------------------ | -------------: | ------------------------: |
| Cross-Entropy Loss |         3.2210 |                **2.7844** |
| Perplexity         |        25.0531 |               **16.1897** |
| PPL Improvement    |              — |                **35.38%** |
| Loss Improvement   |              — |                **13.56%** |

### Key Result

> **WikiText-103 validation perplexity decreased from 25.05 to 16.19 — a 35.38% improvement after 10K continued-pretraining steps.**

This provides a direct quantitative comparison between the original GPT-2 weights and the continued-pretrained model.

---

# 🔬 Domain-Shift Evaluation

The model was also evaluated on **The Verdict**, a separate text corpus that was not the continued-pretraining target.

| Metric     | Original GPT-2 | 10K WikiText GPT-2 |
| ---------- | -------------: | -----------------: |
| Loss       |         3.5328 |             4.3719 |
| Perplexity |        34.2201 |            79.1920 |

The increase in perplexity demonstrates an important property of continued pretraining:

**Improving performance on a target corpus does not necessarily improve performance on every other domain.**

The model became more adapted to WikiText-style text while becoming less suited to this separate evaluation corpus.

This result is retained as part of the evaluation rather than hidden, providing a more complete picture of the experiment.

---

# 🎲 Text Generation

The project supports autoregressive text generation.

### Generation Pipeline

```text
Prompt
  │
  ▼
GPT-2 Tokenizer
  │
  ▼
Token IDs
  │
  ▼
GPT Model
  │
  ▼
Next-Token Logits
  │
  ▼
Temperature Scaling
  │
  ▼
Top-k Filtering
  │
  ▼
Multinomial Sampling
  │
  ▼
Generated Token
  │
  └──────────► Repeat
```

### Temperature

Controls randomness during generation:

```text
Low temperature
      ↓
More deterministic

High temperature
      ↓
More diverse
```

### Top-k Sampling

Restricts sampling to the `k` most probable tokens before sampling.

The project also supports:

* Multinomial sampling
* Autoregressive decoding
* Context-window management
* EOS handling where applicable

---

# 💾 Checkpointing

Training checkpoints store the complete training state.

```text
Checkpoint
    │
    ├── Model State
    ├── Optimizer State
    ├── Training Step
    ├── Epoch
    ├── Training Loss
    ├── Validation Loss
    └── Configuration
```

This allows training to resume without restarting from the beginning.

The original 10K WikiText checkpoint is:

```text
checkpoints/gpt2_wikitext.pt
```

Original baseline checkpoint:

```text
Global step: 9999
Epoch:       4
```

The verified Kaggle continuation checkpoint reached 20,000 completed steps
and is stored separately as `gpt2_wikitext_20k.pt`. Large PyTorch checkpoint
files are ignored by Git and should be stored in Kaggle Datasets, local
storage, or another model-artifact registry rather than committed to GitHub.

---

# 📱 Fine-Tuning: SMS Spam Classification

The pretrained GPT model is also fine-tuned for binary SMS classification using the SMS Spam Collection dataset.

### Labels

```text
ham  → 0
spam → 1
```

### Pipeline

```text
Raw SMS
   │
   ▼
Tokenization
   │
   ▼
Padding / Truncation
   │
   ▼
SpamDataset
   │
   ▼
DataLoader
   │
   ▼
GPT Fine-Tuning
   │
   ▼
Spam / Ham Prediction
```

### Dataset Processing

The pipeline includes:

* Pandas-based dataset loading
* Class balancing
* Random shuffling
* Train / validation / test splitting
* Custom `SpamDataset`
* Sequence truncation
* Sequence padding
* GPT-based classification

### Classification Results

| Epoch | Train Accuracy | Validation Accuracy |
| ----: | -------------: | ------------------: |
|     1 |          70.0% |               72.5% |
|     2 |          82.5% |               85.0% |
|     3 |          90.0% |               90.0% |
|     4 |         100.0% |               97.5% |
|     5 |         100.0% |               97.5% |

**Training time:** 15.82 minutes

---

# 🎓 Instruction Fine-Tuning

The project also implements supervised instruction fine-tuning (SFT).

The instruction dataset contains approximately 1,100 instruction-response examples.

Each sample contains:

```text
Instruction
Input
Response
```

### Prompt Format

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

### Dataset Split

```text
85% Training
10% Test
5% Validation
```

### Custom Collation

The instruction fine-tuning pipeline implements:

* GPT-2 tokenization
* End-of-sequence token handling
* Variable-length sequence padding
* Input/target next-token shifting
* `ignore_index=-100` masking
* Padding exclusion from loss calculation
* Maximum context length of 1024

The resulting training objective focuses the model on learning the response sequence while ignoring padded positions.

### Pipeline

```text
Instruction Dataset
        │
        ▼
Prompt Formatting
        │
        ▼
GPT-2 Tokenization
        │
        ▼
InstructionDataset
        │
        ▼
Custom Collate Function
        │
        ├── Padding
        ├── Next-token shifting
        └── ignore_index masking
        │
        ▼
DataLoader
        │
        ▼
Supervised Fine-Tuning
        │
        ▼
Instruction-Following Generation
```

---

# 🏃 Training & Evaluation Infrastructure

The repository contains reusable training and evaluation utilities.

### Training Loop

The training infrastructure supports:

* Reproducibility seeds
* GPU/CPU device selection
* Automatic mixed precision
* Gradient scaling
* Training loss tracking
* Validation loss tracking
* Tokens-seen tracking
* Periodic evaluation
* Checkpoint saving
* Checkpoint resumption
* Wall-clock timing

### Evaluation

The evaluation framework supports:

* Cross-entropy loss
* Perplexity
* Deterministic validation batches
* Baseline-vs-trained comparisons
* Text generation comparisons
* Dataset-specific evaluation
* JSON result storage
* Generation sample storage

Evaluation results are stored in:

```text
results/
├── evaluation.json
└── generation_samples.txt
```

---

# 🤖 Automated Response Evaluation with Ollama

Instruction-following responses can be evaluated automatically using a locally running Ollama model.

### Evaluation Pipeline

```text
Test Dataset
     │
     ▼
Ollama Process Check
     │
     ▼
Prompt Construction
     │
     ▼
Local Ollama API
     │
     ▼
Response
     │
     ▼
Integer Score Extraction
     │
     ▼
Aggregated Evaluation
```

The implementation uses:

* `psutil` for process monitoring
* `urllib.request` for local API communication
* Fixed generation settings for reproducibility
* Integer-only score extraction
* Graceful conversion error handling
* Batch evaluation

No external hosted LLM API is required for this evaluation pipeline.

---

# 🧪 Testing

The repository contains tests covering major project components:

```text
tests/
├── attention/
├── decoding/
├── finetuning/
├── gpt/
├── inference/
├── instruction_tuning/
├── preprocessing/
├── training/
├── transformer/
└── weights/
```

The project is structured so that individual components can be tested independently.

---

# 📁 Project Structure

```text
LLMs/
│
├── checkpoints/
│   ├── gpt2_wikitext-103.pt
│   └── gpt_checkpoint.pt
│
├── data/
│   ├── the-verdict.txt
│   └── wikitext-103/
│       ├── train.txt
│       └── validation.txt
│
├── docs/
│   ├── LLM_From_Scratch_Documentation.pdf
│   └── LLM_From_Scratch_workflow.png
│
├── gpt2/
│   ├── 124M/
│   └── 355M/
│
├── notebooks/
│   ├── attention/
│   ├── decoding/
│   ├── finetuning/
│   ├── gpt/
│   ├── instruction_tuning/
│   ├── preprocessing/
│   ├── training/
│   ├── transformer/
│   └── weights/
│
├── scripts/
│   ├── benchmark_gpt2.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── prepare_wikitext.py
│   ├── train.py
│   └── verify_training_setup.py
│
├── src/
│   └── llm/
│       ├── attention/
│       ├── decoding/
│       ├── finetuning/
│       ├── gpt/
│       ├── inference/
│       ├── instruction_tuning/
│       ├── preprocessing/
│       ├── training/
│       ├── transformer/
│       └── weights/
│
├── tests/
│   ├── attention/
│   ├── decoding/
│   ├── finetuning/
│   ├── gpt/
│   ├── inference/
│   ├── instruction_tuning/
│   ├── preprocessing/
│   ├── training/
│   ├── transformer/
│   └── weights/
│
├── results/
│   ├── evaluation.json
│   └── generation_samples.txt
│
└── workflow/
    └── LLM_From_Scratch_workflow.png
```

---

# 🧰 Tech Stack

### Programming

* Python

### Deep Learning

* PyTorch
* CUDA
* Automatic Mixed Precision

### NLP

* tiktoken
* GPT-2 BPE tokenizer

### Data Processing

* NumPy
* Pandas
* Regular Expressions

### Visualization

* Matplotlib

### Local AI

* Ollama

### System / Utilities

* psutil
* urllib
* JSON
* PyTorch checkpointing

---

# 🔄 Complete Project Pipeline

```text
                    ┌──────────────────────┐
                    │    Raw Text/Data     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Tokenization &       │
                    │ Preprocessing        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ GPT-Style Transformer│
                    │ Implemented in PyTorch│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Official GPT-2       │
                    │ Weight Loading       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Continued Pretraining│
                    │ on WikiText-103      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Checkpointing &      │
                    │ Training Resumption  │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       ┌─────────────────┐          ┌─────────────────┐
       │ Text Generation │          │ Fine-Tuning     │
       └────────┬────────┘          └────────┬────────┘
                │                            │
                │                   ┌────────┴────────┐
                │                   ▼                 ▼
                │            SMS Classification   Instruction SFT
                │                                      │
                │                                      ▼
                │                              Response Generation
                │                                      │
                │                                      ▼
                │                               Ollama Evaluation
                │
                └──────────────┬───────────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Evaluation & Results │
                    └──────────────────────┘
```

---

# 🤖 MCP Integration

The repository can be accessed by MCP-compatible AI development tools through GitMCP.

Repository:

`https://gitmcp.io/Arjun-Bhattarai/LLMs`

Example configuration:

```json
{
  "servers": {
    "LLMs Docs": {
      "type": "sse",
      "url": "https://gitmcp.io/Arjun-Bhattarai/LLMs"
    }
  }
}
```

This can be used by compatible AI coding assistants to:

* Explain the Transformer implementation
* Locate specific modules
* Navigate the training pipeline
* Understand fine-tuning components
* Explore the project architecture

---

# 📌 Project Status

## ✅ Verified Project Status

The project currently covers the main experimental lifecycle of a GPT-style
language model:

```text
Architecture
     ↓
Tokenization
     ↓
Transformer Implementation
     ↓
GPT-2 Weight Loading
     ↓
Continued Pretraining
     ↓
Checkpointing
     ↓
Text Generation
     ↓
Classification Fine-Tuning
     ↓
Instruction Fine-Tuning
     ↓
Automated Evaluation
```

### Verified Milestones

* GPT-2 124M architecture implemented and tested.
* OpenAI GPT-2 weights loaded into the custom PyTorch model.
* WikiText-103 continued pretraining completed for 10,000 steps.
* Kaggle GPU continuation completed for 20,000 total steps.
* Kaggle 30,000- and 50,000-step runs remain optional continuation targets.

### Original GPT-2 Experiment

```text
Model:              GPT-2 124M
Parameters:         163M
Dataset:            WikiText-103
Training:           Continued pretraining
Optimization Steps: 10,000
Validation PPL:     25.05 → 16.19
PPL Improvement:    35.38%
```

The original experiment above is the 10K-step baseline. The verified Kaggle
continuation reached 20K total steps and reduced validation loss to `2.7882`.
Further 30K and 50K runs are continuation targets using the same checkpoint
resume workflow documented above.

The project demonstrates the progression from implementing Transformer components manually to loading a real pretrained GPT-2 model and adapting it to a new corpus.

---

# 👨‍💻 Author

Built by **Arjun Bhattarai** as a deep learning systems project focused on understanding the internal mechanics of modern Large Language Models.

Rather than relying exclusively on high-level LLM APIs, this project explores the underlying components directly:

**attention → Transformer blocks → GPT architecture → pretrained weight loading → continued pretraining → fine-tuning → instruction tuning → evaluation.**

---

# 🔗 Next Project

**RAG From Scratch**

Extending this LLM foundation into a retrieval-augmented generation system covering retrieval, reranking, query transformation, advanced indexing, CRAG, Self-RAG, long-context processing, and local LLM inference.

Repository:

**[RAG-with-LangChain](https://github.com/Arjun-Bhattarai/RAG-with-LangChain)**


````
