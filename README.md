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

## 🧠 Self-Attention (Implemented from Scratch)

- Created token embeddings using PyTorch tensors
- Computed **dot-product attention scores**
- Applied **softmax normalization**
- Generated **attention weights**
- Computed **context vectors using weighted sum**

---

## 🔒 Causal Attention (Masked Self-Attention)

Causal attention extends self-attention by ensuring that each token can only attend to **itself and previous tokens** — never future ones. This is essential for autoregressive language model training and generation.

### Key Concepts

- **Causal Mask**: An upper-triangular mask (filled with `-inf` or `0`) is applied to the raw attention scores before softmax, zeroing out all future positions
- **Masked Softmax**: After masking, softmax converts scores to attention weights where future positions have zero weight
- **Autoregressive Property**: Guarantees the model cannot "cheat" by looking ahead during training

#

---

## 🚀 Key Features

- Raw text preprocessing using regex
- Word-level tokenization with `<UNK>` handling
- Basic BPE understanding
- Encode/decode functions
- Input-target dataset creation for next-token prediction
- Sliding window training data
- Token embeddings using `torch.nn.Embedding`
- Manual self-attention implementation (from scratch)
- **Causal (masked) self-attention implementation**
- **Upper-triangular masking with `-inf` fill**
- **Autoregressive attention weight verification**
- Context vector computation
- PyTorch DataLoader integration

---

## 📦 Tech Stack

- Python
- Regular Expressions (`re`)
- PyTorch
- tiktoken (GPT-style tokenizer concept)
- NumPy (optional for preprocessing)