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

<<<<<<< HEAD
-Self-Attention — dot-product attention scores, softmax normalization, and context vector computation
Causal Attention — upper-triangular -inf masking for autoregressive (left-to-right) generation
Multi-Head Attention — parallel attention heads via splitting, scaled attention, concatenation, and output projection
Dummy GPT Model — token embeddings, positional embeddings, transformer block placeholders, and vocabulary logits generation
Custom Layer Normalization — implemented LayerNorm from scratch using mean-variance normalization with learnable scale (gamma) and shift (beta) parameters
GELU Activation Function — smooth probabilistic activation used in GPT, BERT, and modern LLMs instead of ReLU
Feed Forward Neural Network (FFN) — linear expansion → GELU activation → projection back to embedding size
=======
-Self-Attention — dot-product attention scores, softmax normalization, and context vector computation<br>
-Causal Attention — upper-triangular -inf masking for autoregressive (left-to-right) generation<br>
-Multi-Head Attention — parallel attention heads via splitting, scaled attention, concatenation, and output projection<br>
-Dummy GPT Model — token embeddings, positional embeddings, transformer block placeholders, and vocabulary logits generation<br>
-Custom Layer Normalization — implemented LayerNorm from scratch using mean-variance normalization with learnable scale (gamma) and shift (beta) parameters<br>
-GELU Activation Function — smooth probabilistic activation used in GPT, BERT, and modern LLMs instead of ReLU<br>
-Feed Forward Neural Network (FFN) — linear expansion → GELU activation → projection back to embedding size<br>
>>>>>>> d32e5bee31ce0b90e3ab52c4ca51995015993d3a

## 🚀 Key Features

- Regex-based text cleaning and word-level tokenization<br>
-Vocabulary builder with <UNK> token handling<br>
-Encode/decode pipeline with sliding window dataset creation<br>
-Token and positional embeddings using torch.nn.Embedding<br>
-Self-attention implemented from scratch (dot-product, softmax, context vectors)<br>
-Causal masked attention with upper-triangular -inf masking<br>
-Multi-head attention implementation with parallel heads and concatenation<br>
-Dummy GPT-style architecture implementation<br>
-Custom LayerNorm implementation from scratch<br>
-GELU activation implementation<br>
-Transformer Feed Forward Network (FFN) implementation<br>
-PyTorch DataLoader integration for next-token prediction training<br>

---

## 📦 Tech Stack

-Python<br>
-Regular Expressions (re)<br>
-PyTorch<br>
-Matplotlib<br>
-NumPy (optional for preprocessing)<br>
-tiktoken (GPT-style tokenizer concept)<br>
