# LLM Tokenization & Preprocessing Project

## 📌 Project Overview

This project demonstrates fundamental **text preprocessing and tokenization techniques** used in Large Language Models (LLMs). It focuses on how raw text is converted into numerical form that models can understand.

The pipeline includes reading raw text, cleaning it using regular expressions (`re` module), and converting it into tokens for training.

It also explores internal workings of tokenizers, including **word-level tokenization** and a simple implementation of **Byte Pair Encoding (BPE)** used in modern LLMs like GPT.

Additionally, the project introduces **token embeddings using PyTorch**, showing how token IDs are converted into dense vector representations for neural networks.

---

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
