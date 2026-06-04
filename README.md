# 🦾 LLM Tokenization & Transformer from Scratch

A complete end-to-end implementation of a **GPT-style Transformer** built entirely from scratch using PyTorch — transforming raw text into tokens and processing them through a full Transformer architecture for **next-token prediction and text generation**.

---

## 🚀 Highlights

* **GPT-style Transformer** model built entirely from scratch.
* **Self-attention, causal attention, and multi-head attention** mechanisms.
* Custom **LayerNorm, GELU, and Feed-Forward Network (FFN)** layers.
* Full **tokenization and preprocessing** pipeline.
* Autoregressive text generation with **temperature scaling and top-k sampling**.
* **Cross-Entropy Loss** and **Perplexity** evaluation metrics.
* Train/validation pipeline integrated with **PyTorch DataLoaders**.
* Comprehensive parameter, tensor shape, and memory usage analysis.

---

## ⚙️ Preprocessing Pipeline

1.  **Load Raw Text** — The training dataset (corpus) is loaded as raw input text.
2.  **Text Normalization** — Text is converted to lowercase for vocabulary consistency.
3.  **Tokenization** — Words and punctuation are split using a regular expression:
    ```python
    import re
    tokens = re.findall(r"\b\w+\b", raw_data.lower())
    ```
4.  **Vocabulary Building** — Word-to-index mappings are created, incorporating an `<UNK>` token to gracefully handle out-of-vocabulary words.
5.  **Encoding / Decoding** — Utility functions convert raw text ↔ token IDs for seamless training and inference.

---

## 🧠 Model Architecture

### Attention Mechanisms

| Component | Description |
| :--- | :--- |
| **Self-Attention** | Computes scaled dot-product attention to map token-to-token relationships. |
| **Causal Attention** | Applies a lower-triangular mask ensuring tokens only attend to past and current positions. |
| **Multi-Head Attention** | Runs multiple attention heads in parallel to capture diverse representation subspaces. |

### Transformer Block

Each block follows the standard decoder-only paradigm: 
$$\text{Multi-Head Attention} \longrightarrow \text{Feed-Forward Network} \longrightarrow \text{LayerNorm + Residual Connections}$$

* **FFN** — `Linear → GELU → Linear` architecture that expands the feature dimension (typically by $4\times$) and compresses it back.
* **LayerNorm** — Stabilizes internal dynamics during training using learnable $\gamma$ (gamma) and $\beta$ (beta) parameters.
* **GELU** — Smooth, non-linear activation function crucial for modern high-performing Transformers.

### GPT-Style Model Stack
---
Token Embeddings + Positional Embeddings
│
▼
Stacked Transformer Blocks
│
▼
Final Layer Normalization
│
▼
Linear Projection → Vocabulary Logits
(Weight-tied with embedding layer)
---

## 🔄 Text Generation Pipeline

Input Text ──> Token IDs ──> Model ──> Next-Token Probabilities
▲                                               │
│                                               ▼
Decoded Text <── Output Tokens <── Autoregressive Sampling


> **Note:** Causal masking ensures that generation remains strictly autoregressive (sequential and left-to-right), preventing the model from looking ahead.

---

## 🎲 Sampling & Temperature Scaling

Controls the diversity, creativity, and randomness during text generation.

* **`softmax_with_temperature`** — Scales raw logits before passing them to the softmax function. Low temperature pushes the model toward sharp, deterministic (greedy) outputs, while high temperature flattens the distribution for creative, diverse text.
* **Top-k Filtering** — Restricts the sampling pool to the top `k` most probable tokens, effectively filtering out low-probability tail noise.
* **Multinomial Sampling** — Randomly samples the next token proportionally based on the newly filtered and scaled probability distribution.
* **EOS Handling** — Monitors for an end-of-sequence token to stop generation cleanly.
* **Context Cropping** — Trims the active generation context to fit within the model's maximum allowed context window ($T$).
* **Visualization** — Includes built-in Matplotlib bar charts tracking shifts in token probabilities across different temperature thresholds.


Logits ──> Top-k Filter ──> Temperature Scale ──> Softmax ──> Multinomial Sample ──> Next Token


vice selection (`cpu`, `cuda`, or `mps`) and explicit random seeds for reproducibility.
* **`plot_losses`** — Generates clean plots visualizing training vs. validation loss decay across both training epochs and total tokens seen.

---

## 🧪 Key Features

* Regex-based custom tokenizer with custom `<UNK>` out-of-vocabulary handling.
* From-scratch PyTorch implementations of Token + Learned Positional Embeddings.
* Full decoder-only causal Transformer architecture.
* Advanced generation utilities including temperature scaling and top-k sampling.
* Modular, extensible, and clean object-oriented design.

---

## 📦 Tech Stack

* **Language:** `Python`
* **Deep Learning:** `PyTorch`
* **Data Processing:** `NumPy` · `Regex (re)`
* **Visualization:** `Matplotlib`
* **Conceptual Inspiration:** `tiktoken` / OpenAI GPT architectures

---

## 📌 Full Pipeline Summary
Raw Text ──> Tokenization ──> Embeddings ──> Transformer Layers ──> Next Token Prediction ──> Training Loop ──> Generated Text

---

## 👨‍💻 Author

Built as a foundational deep learning project to pull back the curtain on Large Language Models, moving past APIs to implement their core mechanics entirely from first principles.