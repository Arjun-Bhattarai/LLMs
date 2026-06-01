# 🦾 LLM Tokenization & Transformer from Scratch

A complete end-to-end implementation of a **GPT-style Transformer** built entirely from scratch using PyTorch — transforming raw text into tokens and processing them through a full Transformer architecture for **next-token prediction and text generation**.

---

## 🚀 Highlights

- GPT-style Transformer model built from scratch
- Self-attention, causal attention, and multi-head attention
- Custom LayerNorm, GELU, and Feed-Forward Network (FFN)
- Full tokenization + preprocessing pipeline
- Autoregressive text generation system
- Cross-Entropy Loss and Perplexity evaluation
- Train/validation pipeline with PyTorch DataLoaders
- Parameter, tensor shape, and memory usage analysis

---

## ⚙️ Preprocessing Pipeline

**1. Load Raw Text** — Dataset (corpus) loaded as input text.

**2. Text Normalization** — Convert to lowercase for consistency.

**3. Tokenization**
```python
import re
tokens = re.findall(r"\b\w+\b", raw_data.lower())
```

**4. Vocabulary Building** — Word-to-index mappings with `<UNK>` token for out-of-vocabulary words.

**5. Encoding / Decoding** — Convert text ↔ token IDs for training and inference.

---

## 🧠 Model Architecture

### Attention Mechanisms
| Component | Description |
|---|---|
| **Self-Attention** | Dot-product attention over token relationships |
| **Causal Attention** | Masked — tokens attend only to previous positions |
| **Multi-Head Attention** | Parallel heads capturing different representation subspaces |

### Transformer Block
Each block contains: Multi-Head Attention → Feed-Forward Network → Layer Normalization + Residual Connections.

### Key Components
- **FFN**: `Linear → GELU → Linear` (expands then compresses feature dimension)
- **LayerNorm**: Stabilizes training with learnable γ (gamma) and β (beta)
- **GELU**: Smooth nonlinear activation used in modern Transformers

### GPT-Style Model Stack
```
Token Embeddings + Positional Embeddings
        ↓
Stacked Transformer Blocks
        ↓
Final Layer Normalization
        ↓
Linear Projection → Vocabulary Logits
(weight-tied with embedding layer)
```

---

## 🔄 Text Generation Pipeline

```
Input Text → Token IDs → Model → Next-Token Probabilities
     ↑                                        ↓
  Decoded Text ← Output Tokens ← Autoregressive Sampling
```

Causal masking ensures sequential, left-to-right generation.

---

## 📊 Training & Evaluation

- Sliding window approach for input–target pair creation
- **Loss**: Cross-Entropy | **Metric**: Perplexity = `exp(loss)`
- **Train/Val split**: 90% / 10%
- Helper functions: `calc_loss_batch`, `calc_loss_loader`
- Evaluated on both training and validation sets
- Dataset integrity and token coverage verified

---

## 🧪 Key Features

- Regex-based tokenizer with custom `<UNK>` vocabulary
- Token + positional embeddings in PyTorch
- Full causal Transformer architecture from scratch
- Modular, reusable design
- End-to-end training and evaluation framework

---

## 📦 Tech Stack

`Python` · `PyTorch` · `NumPy` · `Regex (re)` · `Matplotlib` · `tiktoken` (conceptual inspiration)

---

## 📌 Pipeline Summary

```
Raw Text → Tokenization → Embeddings → Transformer Layers → Next Token Prediction → Generated Text
```

---



## 👨‍💻 Author

Built as a deep learning project to understand how Large Language Models work internally.
