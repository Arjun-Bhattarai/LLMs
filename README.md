# LLM Tokenization & Preprocessing Project

This project demonstrates fundamental **text preprocessing and tokenization techniques used in Large Language Models (LLMs)**. It explains how raw text is converted into numerical representations that machine learning models can understand.

The pipeline includes cleaning text using regex, tokenizing into words, building vocabulary, handling `<UNK>` tokens, and converting text into training-ready sequences. It also explores **BPE concepts** and **PyTorch-based token embeddings**.

Additionally, this project implements a **basic self-attention mechanism from scratch**, including dot-product similarity, softmax normalization, and context vector generation, as well as a **causal (masked) self-attention mechanism** to understand how Transformers work internally.

---

## ⚙️ Preprocessing Pipeline

1. **Load Raw Text**
   - Input dataset (corpus)

2. **Text Normalization**
   - Convert text to lowercase

3. **Tokenization**
```python
import re
tokens = re.findall(r"\b\w+\b", raw_data.lower())
```

4. **Vocabulary Building**
   - Map words → token IDs
   - Handle unknown words using `<UNK>`

5. **Encoding / Decoding**
   - Convert text ↔ token IDs

---

## Implementations
- **Self-Attention** — dot-product scores, softmax normalization, context vectors
- **Causal Attention** — upper-triangular `-inf` mask for autoregressive generation
- **Multi-Head Attention** — parallel heads via `MultiHeadAttentionWrapper`, concatenated outputs

## 🚀 Key Features

## Key Features
- Regex-based text cleaning & word-level tokenization
- Vocabulary builder with `<UNK>` token support
- Encode/decode functions + sliding window dataset creation
- `torch.nn.Embedding` for token + positional embeddings
- Self-attention from scratch (dot-product, softmax, context vectors)
- Causal masked attention with upper-triangular `-inf` fill
- Multi-head attention via `MultiHeadAttentionWrapper`
- PyTorch `DataLoader` integration for next-token prediction training

---

## 📦 Tech Stack

- Python
- Regular Expressions (`re`)
- PyTorch
- tiktoken (GPT-style tokenizer concept)
- NumPy (optional for preprocessing)