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

- **Self-Attention** → dot-product attention, softmax normalization.  
- **Causal Attention** → autoregressive masking with upper-triangular `-inf`.  
- **Multi-Head Attention** → parallel heads, scaling, concatenation, projection.  
- **Dummy GPT Model** → embeddings, transformer block, logits.  
- **Custom LayerNorm** → learnable `gamma` and `beta`.  
- **GELU Activation** → smooth nonlinear activation.  
- **Feed-Forward Network (FFN)** → expand → GELU → project back.  
- **Residual Connections** → shortcut paths for gradient flow.  
- **Tokenizer Concepts** → regex, BPE, tiktoken-inspired.  
- **Sliding Window Dataset** → sequences for next-token prediction.  
- **TransformerBlock** → attention + FFN + LayerNorm + residuals.  
- **GPTModel**  
  - Embeddings, stacked TransformerBlocks, final norm, output head.  
  - Verified input/output shapes.  
  - Counted parameters and memory size.  
  - Checked embedding/output layer dimensions.  
  - GPT‑2 style weight tying analysis.  
- **Text Generation Pipeline**  
  - Implemented `generate_text_simple` for autoregressive generation.  
  - Context cropping, logits extraction, softmax probability, argmax sampling.  
  - Encoded starting context into token IDs and prepared tensor input.  
  - Ran model in eval mode to generate new tokens.  
  - Decoded output sequence back into human-readable text.  
  - Verified output length and shapes during generation.  

---

## 🚀 Key Features

- Regex-based text cleaning & tokenization.  
- Vocabulary builder with `<UNK>` handling.  
- Encoding/decoding pipeline.  
- PyTorch embeddings for tokens & positions.  
- Transformer modules from scratch: attention, FFN, LayerNorm, residuals.  
- Dummy GPT-style architecture for experimentation.  
- Gradient analysis tools for deep networks.  
- TransformerBlock with causal masking, dropout, GELU, residuals.  
- GPTModel with parameter/memory analysis and weight tying.  
- **Text Generation** → end-to-end pipeline: encoding, inference, decoding.  

---

## 📦 Tech Stack

- Python  
- Regex (`re`)  
- PyTorch  
- NumPy  
- Matplotlib  
- tiktoken (concept inspiration)  

---

## 📚 Learning Outcomes

- Master LLM preprocessing and tokenizer fundamentals.  
- Understand Transformer internals through hands-on implementation.  
- Build GPT-like models from scratch.  
- Generate text sequences with context handling and decoding.  
- Analyze parameters, memory, and gradient flow in deep networks.  

---

*Explore the notebook and code to experiment with every stage—see how words become tokens, how transformers work internally, and how advanced techniques ensure deep models can learn effectively!*