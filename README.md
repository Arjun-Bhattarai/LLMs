# LLM Tokenization & Preprocessing Project

## 📌 Project Overview

This project demonstrates fundamental **text preprocessing and tokenization techniques used in Large Language Models (LLMs)**. It explains how raw text is converted into numerical representations that machine learning models can understand.

The pipeline includes reading raw text, cleaning it using regular expressions (`re module`), and converting it into tokens for training. It also explores how tokenizers work internally using a simple **word-level tokenizer with vocabulary mapping and `<UNK>` handling**.

Additionally, the project introduces **Byte Pair Encoding (BPE)** concepts and **token embeddings using PyTorch**, showing how token IDs are converted into dense vector representations for neural networks.

---

## ⚙️ Preprocessing Pipeline

The preprocessing pipeline used in this project:

1. **Load Raw Text**
   - Input raw dataset (corpus)

2. **Text Normalization**
   - Convert text to lowercase

3. **Tokenization**
   ```python
   import re
   tokens = re.findall(r"\b\w+\b", raw_data.lower())

## 🚀 Key Features

- Raw text loading and preprocessing
- Text cleaning using regular expressions (`re`)
- Word-level tokenization with `<UNK>` handling
- Custom Byte Pair Encoding (BPE) implementation
- Encode and decode functions for text ↔ token conversion
- Special token support like `<|endoftext|>`
- Input-target pair creation for next-token prediction
- Sliding window dataset for training LLMs
- Token embedding layer using `torch.nn.Embedding`
- PyTorch DataLoader integration for batching

---

## 📦 Tech Stack

- Python
- Regular Expressions (`re`)
- PyTorch
- tiktoken (GPT-style tokenizer)
- Token Embeddings using `torch.nn.Embedding`

---



## ⚙️ Future Improvements

- Train a small language model using this pipeline
- Add visualization of embeddings
- Improve BPE efficiency
- Extend dataset to multiple documents
