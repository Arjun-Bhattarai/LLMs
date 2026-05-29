# 🦾 LLM Tokenization & Preprocessing Project

This project showcases essential text preprocessing and tokenization techniques fundamental to Large Language Models (LLMs)—demonstrating how raw text is transformed into numerical representations suitable for machine learning. The pipeline covers text cleaning, word tokenization, vocabulary creation (with `<UNK>` handling), the encoding-decoding process, and dives into both classic and modern LLM architectural components.

In addition to tokenizer concepts (including BPE and tiktoken), this project implements core Transformer modules **built from scratch**: self-attention, causal (masked) attention, multi-head attention, a dummy GPT-style model, custom LayerNorm, the GELU activation, Feed-Forward Neural Network (FFN) blocks, and more. As an advanced extension, a deep neural network with GELU activation, residual (shortcut) connections, and gradient analysis is included to help understand the flow of information and gradients in deep models.

---

## ⚙️ Preprocessing Pipeline

1. **Load Raw Text**
   - Input your dataset (the corpus).

2. **Text Normalization**
   - Convert to lowercase.

3. **Tokenization**
   ```python
   import re
   tokens = re.findall(r"\b\w+\b", raw_data.lower())
   ```

4. **Vocabulary Building**
   - Map words to token IDs.
   - Handle out-of-vocabulary with `<UNK>`.

5. **Encoding / Decoding**
   - Convert text ↔ token IDs.

---

## 🧠 Algorithm Implementations

- **Self-Attention**
  - Dot-product attention, softmax normalization, context vector calculation.
- **Causal (Masked) Attention**
  - Upper-triangular `-inf` masking for autoregressive language modeling.
- **Multi-Head Attention**
  - Parallel attention heads, scaling, concatenation, and output projection.
- **Dummy GPT Model**
  - Token and positional embeddings, transformer block structure, logit generation.
- **Custom Layer Normalization**
  - Implemented from scratch with learnable `gamma` and `beta`.
- **GELU Activation Function**
  - Probabilistic activation, replacing ReLU in modern LLMs.
- **Feed-Forward Neural Network (FFN)**
  - Linear expansion → GELU → projection back to embedding size.
- **Deep Neural Network with Residual Connections & GELU**
  - Residual (shortcut) connections ensure information/gradient flow in deep stacks.
  - Integrated gradient analysis for deeper understanding.
- **Tokenizer Concepts**
  - Regex-based and BPE/tokenizer (tiktoken-inspired) approaches.
- **Sliding Window Dataset Preparation**
  - For next-token prediction training with DataLoader.
- **TransformerBlock**
  - Integrated MultiHeadAttention with causal masking and dropout.
  - Added FeedForward (FFN) network with GELU activation.
  - Applied LayerNorm before attention and feed-forward layers.
  - Implemented Residual (shortcut) connections for stable gradient flow.
  - Verified input/output tensor shapes with sample run.

---

## 🚀 Key Features

- Regex-based text cleaning & word-level tokenization.
- Vocabulary builder with robust `<UNK>` handling.
- Sliding-window encoding/decoding pipeline.
- PyTorch `nn.Embedding` for token & positional embeddings.
- Transformer building blocks from scratch:
  - Self-attention, causal masking, multi-head attention.
  - Feed-forward (FFN) blocks with GELU activation.
  - Custom LayerNorm.
  - Residual connections.
- Dummy GPT-style architecture for direct experimentation.
- PyTorch DataLoader integration for training.
- Gradient analysis tools to inspect training dynamics in deep networks.
- TransformerBlock
  - Integrated MultiHeadAttention with causal masking and dropout.
  - Added FeedForward (FFN) network with GELU activation.
  - Applied LayerNorm before attention and feed-forward layers.
  - Implemented Residual (shortcut) connections for stable gradient flow.
  - Verified input/output tensor shapes with sample run.
- GPTModel embeddings, stacked TransformerBlocks, final norm, output head.  
  - Verified input/output shapes.  
  - Counted parameters and memory size.  
  - Checked embedding/output layer dimensions.  
  - Added GPT‑2 style weight tying analysis. 

---

## 📦 Tech Stack

- **Python**
- **Regular Expressions** (`re`)
- **PyTorch**
- **Matplotlib** (visualization)
- **NumPy** (for preprocessing, optional)
- **tiktoken** (inspiration for tokenizer concepts)

---

## 📚 Learning Outcomes

- Master LLM preprocessing and tokenizer fundamentals.
- Understand Transformer inner workings through hands-on module implementations.
- Practice building GPT-like models and components from scratch.
- Visualize and analyze deep learning model gradients and residual flows.

---

*Explore the notebook and code to experiment with every stage—see how words become tokens, how transformers work internally, and how advanced techniques ensure deep models can learn effectively!*
