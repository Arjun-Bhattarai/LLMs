# LLM Tokenization & Preprocessing Project

This project demonstrates fundamental text preprocessing and tokenization techniques used in Large Language Models (LLMs). It explains how raw text is converted into numerical representations that machine learning models can understand.

The pipeline includes cleaning text using regex, tokenizing into words, building vocabulary, handling <UNK> tokens, and converting text into training-ready sequences. It also explores BPE concepts and PyTorch-based token embeddings.

Additionally, this project implements core Transformer building blocks from scratch, including self-attention, causal (masked) attention, multi-head attention, dummy GPT architecture, custom Layer Normalization, GELU activation function, and Feed Forward Neural Networks (FFN) to understand how modern LLMs operate internally.

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

-Self-Attention — dot-product attention scores, softmax normalization, and context vector computation
-Causal Attention — upper-triangular -inf masking for autoregressive (left-to-right) generation
-Multi-Head Attention — parallel attention heads via splitting, scaled attention, concatenation, and output projection
-Dummy GPT Model — token embeddings, positional embeddings, transformer block placeholders, and vocabulary logits generation
-Custom Layer Normalization — implemented LayerNorm from scratch using mean-variance normalization with learnable scale (gamma) and shift (beta) parameters
-GELU Activation Function — smooth probabilistic activation used in GPT, BERT, and modern LLMs instead of ReLU
-Feed Forward Neural Network (FFN) — linear expansion → GELU activation → projection back to embedding size
---

## 🚀 Key Features

- Regex-based text cleaning and word-level tokenization
-Vocabulary builder with <UNK> token handling
-Encode/decode pipeline with sliding window dataset creation
-Token and positional embeddings using torch.nn.Embedding
-Self-attention implemented from scratch (dot-product, softmax, context vectors)
-Causal masked attention with upper-triangular -inf masking
-Multi-head attention implementation with parallel heads and concatenation
-Dummy GPT-style architecture implementation
-Custom LayerNorm implementation from scratch
-GELU activation implementation
-Transformer Feed Forward Network (FFN) implementation
-PyTorch DataLoader integration for next-token prediction training

---

## 📦 Tech Stack

-Python
-Regular Expressions (re)
-PyTorch
-Matplotlib
-NumPy (optional for preprocessing)
-tiktoken (GPT-style tokenizer concept)
