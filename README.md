# 🦾 LLM Tokenization & Transformer from Scratch

A complete end-to-end implementation of a **GPT-style Transformer** built entirely from scratch using PyTorch — transforming raw text into tokens and processing them through a full Transformer architecture for **next-token prediction and text generation**.

---

## 🚀 Highlights

- **GPT-style Transformer** model built entirely from scratch
- **Self-attention, causal attention, and multi-head attention** mechanisms
- Custom **LayerNorm, GELU, and Feed-Forward Network (FFN)** layers
- Full **tokenization and preprocessing** pipeline
- Autoregressive text generation with **temperature scaling and top-k sampling**
- **GPT-2 pretrained weight loading** across all four model variants
- Model and optimizer **checkpointing** for seamless training resumption
- **Cross-Entropy Loss** and **Perplexity** evaluation metrics
- Train/validation pipeline integrated with **PyTorch DataLoaders**
- Comprehensive parameter, tensor shape, and memory usage analysis

---

## ⚙️ Preprocessing Pipeline

1. **Load Raw Text** — The training dataset (corpus) is loaded as raw input text
2. **Text Normalization** — Text is converted to lowercase for vocabulary consistency
3. **Tokenization** — Words and punctuation are split using a regular expression:
   ```python
   import re
   tokens = re.findall(r"\b\w+\b", raw_data.lower())
   ```
4. **Vocabulary Building** — Word-to-index mappings are created, incorporating an `<UNK>` token to gracefully handle out-of-vocabulary words
5. **Encoding / Decoding** — Utility functions convert raw text ↔ token IDs for seamless training and inference

---

## 🧠 Model Architecture

### Attention Mechanisms

| Component | Description |
| :--- | :--- |
| **Self-Attention** | Computes scaled dot-product attention to map token-to-token relationships |
| **Causal Attention** | Applies a lower-triangular mask ensuring tokens only attend to past and current positions |
| **Multi-Head Attention** | Runs multiple attention heads in parallel to capture diverse representation subspaces |

### Transformer Block

Each block follows the standard decoder-only paradigm:

```
Multi-Head Attention
        ↓
Feed-Forward Network
        ↓
LayerNorm + Residual Connections
```

- **FFN** — `Linear → GELU → Linear` architecture that expands the feature dimension (typically by 4×) and compresses it back
- **LayerNorm** — Stabilizes internal dynamics during training using learnable γ (gamma) and β (beta) parameters
- **GELU** — Smooth, non-linear activation function crucial for modern high-performing Transformers

### GPT-Style Model Stack

```
Token Embeddings + Positional Embeddings
        ↓
Stacked Transformer Blocks
        ↓
Final Layer Normalization
        ↓
Linear Projection → Vocabulary Logits
(Weight-tied with embedding layer)
```

---

## 🔄 Text Generation Pipeline

```
Input Text → Token IDs → Model → Next-Token Probabilities
    ↑                                      ↓
    ←───── Decoded Text ←───── Output Tokens ←───── Autoregressive Sampling
```

> **Note:** Causal masking ensures that generation remains strictly autoregressive (sequential and left-to-right), preventing the model from looking ahead.

---

## 🎲 Sampling & Temperature Scaling

Controls the diversity, creativity, and randomness during text generation.

- **`softmax_with_temperature`** — Scales raw logits before passing them to the softmax function. Low temperature pushes the model toward sharp, deterministic (greedy) outputs, while high temperature flattens the distribution for creative, diverse text
- **Top-k Filtering** — Restricts the sampling pool to the top `k` most probable tokens, effectively filtering out low-probability tail noise
- **Multinomial Sampling** — Randomly samples the next token proportionally based on the newly filtered and scaled probability distribution
- **EOS Handling** — Monitors for an end-of-sequence token to stop generation cleanly
- **Context Cropping** — Trims the active generation context to fit within the model's maximum allowed context window (T)
- **Visualization** — Includes built-in Matplotlib bar charts tracking shifts in token probabilities across different temperature thresholds

### Sampling Pipeline

```
Logits → Top-k Filter → Temperature Scale → Softmax → Multinomial Sample → Next Token
```

---

## 🏋️ Pretrained Weights & Checkpointing

### GPT-2 Weight Loading

Supports all four OpenAI GPT-2 variants via `download_and_load_gpt2`:

| Variant | Parameters |
| :--- | :--- |
| `gpt2-small` | 124M |
| `gpt2-medium` | 355M |
| `gpt2-large` | 774M |
| `gpt2-xl` | 1558M |

Weights are mapped into the custom `GPTModel` via `load_weights_into_gpt()` — covering token/positional embeddings, QKV attention weights and biases, feed-forward layers, layer norms, and the output head. An `assign()` utility enforces shape safety and wraps tensors as `nn.Parameter`.

### Checkpointing

Full training state is persisted and restored:

```python
# Save
torch.save(
    {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, 
    "checkpoint.pt"
)

# Resume
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

- **AdamW** optimizer with configurable `lr` and `weight_decay`
- Resuming from a checkpoint restores both model weights and optimizer momentum/variance state

---

## 📊 Classification Fine-tuning — SMS Spam Detection

Fine-tunes the GPT-style model on binary text classification using the SMS Spam Collection dataset.

- **Data prep** — Loaded as `.tsv` into a Pandas DataFrame (`Label`, `Text` columns)
- **Class balancing** — Ham randomly downsampled to match spam count; labels mapped to numeric (`ham=0`, `spam=1`)
- **Splits** — Shuffled and divided via `random_split()` into `train.csv`, `validation.csv`, `test.csv`
- **`SpamDataset`** — Custom PyTorch Dataset with tokenization, truncation, and padding; `max_length` derived from train set and shared across all splits
- **DataLoaders** — `train_loader` with `shuffle=True`, `drop_last=True`; val/test loaders without shuffling; batch dims verified across all splits

### Fine-tuning Pipeline

```
Raw SMS Text
    ↓
TSV Load → Class Balance → random_split()
    ↓
SpamDataset (tokenize + pad)
    ↓
DataLoader (batch + shuffle)
    ↓
GPT Fine-tune
    ↓
Spam / Ham Classification
```

---

## 🧪 Key Features

- Regex-based custom tokenizer with custom `<UNK>` out-of-vocabulary handling
- From-scratch PyTorch implementations of Token + Learned Positional Embeddings
- Full decoder-only causal Transformer architecture
- GPT-2 pretrained weight loading with shape-safe `assign()` utility
- Model + optimizer checkpointing for training resumption
- Advanced generation utilities including temperature scaling and top-k sampling
- Modular, extensible, and clean object-oriented design

---

## 📦 Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Data Processing:** NumPy · Regex (re)
- **Visualization:** Matplotlib
- **Conceptual Inspiration:** tiktoken / OpenAI GPT architectures

---

## 📌 Full Pipeline Summary

```
Raw Text
    ↓
Tokenization
    ↓
Embeddings
    ↓
Transformer Layers
    ↓
Next Token Prediction
    ↓
Training Loop ← GPT-2 Pretrained Weights
    ↓
Checkpoint Save/Load
    ↓
Generated Text
```

---

## 👨‍💻 Author

Built as a foundational deep learning project to pull back the curtain on Large Language Models, moving past APIs to implement their core mechanics entirely from first principles.