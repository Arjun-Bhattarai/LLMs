# LLM Tokenization & Preprocessing Project

This project demonstrates fundamental **text preprocessing and tokenization techniques used in Large Language Models (LLMs)**. It explains how raw text is converted into numerical representations that machine learning models can understand.

The pipeline includes cleaning text using regex, tokenizing into words, building vocabulary, handling `<UNK>` tokens, and converting text into training-ready sequences. It also explores **BPE concepts** and **PyTorch-based token embeddings**.

Additionally, this project implements core Transformer building blocks from scratch, including **self-attention, causal (masked) attention, and multi-head attention**, to understand how modern LLMs operate internally.


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
## 🧠 Implementations
Self-Attention — dot-product attention scores, softmax normalization, and context vector computation
Causal Attention — upper-triangular -inf masking for autoregressive (left-to-right) generation
Multi-Head Attention — parallel attention heads via MultiHeadAttentionWrapper and full MultiHeadAttention implementation with head splitting, scaled attention, dropout, and output projection
Dummy GPT Model — token embeddings, positional embeddings, transformer block placeholders, layer normalization placeholders, and vocabulary logits generatio
## 🚀 Key Features
Regex-based text cleaning and word-level tokenization
Vocabulary builder with <UNK> token handling
Encode/decode pipeline with sliding window dataset creation
Token and positional embeddings using torch.nn.Embedding
Self-attention implemented from scratch (dot-product, softmax, context vectors)
Causal masked attention with upper-triangular -inf masking
Multi-head attention implementation with parallel heads and concatenation
Dummy GPT-style architecture implementation
PyTorch DataLoader integration for next-token prediction training

---

## 📦Tech Stack
Python
Regular Expressions (re)
PyTorch
tiktoken (GPT-style tokenizer concept)
NumPy (optional for preprocessing)