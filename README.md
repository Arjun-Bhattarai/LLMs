# LLM Tokenization & Preprocessing Project

This project demonstrates fundamental **text preprocessing and tokenization techniques used in Large Language Models (LLMs)**. It explains how raw text is converted into numerical representations that machine learning models can understand.

The pipeline includes cleaning text using regex, tokenizing into words, building vocabulary, handling `<UNK>` tokens, and converting text into training-ready sequences. It also explores **BPE concepts** and **PyTorch-based token embeddings**.

Additionally, this project implements a **basic self-attention mechanism from scratch**, including dot-product similarity, softmax normalization, and context vector generation to understand how Transformers work internally.

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



## 🚀 Key Features

- Raw text preprocessing using regex
- Word-level tokenization with `<UNK>` handling
- Basic BPE understanding
- Encode/decode functions
- Input-target dataset creation for next-token prediction
- Sliding window training data
- Token embeddings using `torch.nn.Embedding`
- Manual self-attention implementation (from scratch)
- Context vector computation
- PyTorch DataLoader integration

---

## 📦 Tech Stack

- Python
- Regular Expressions (`re`)
- PyTorch
- tiktoken (GPT-style tokenizer concept)
- NumPy (optional for preprocessing)



