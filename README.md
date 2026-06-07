# 🦾 GPT-Style LLM from Scratch with GPT-2 Weight Loading and Fine-Tuning

A complete end-to-end implementation of a **GPT-style Large Language Model (LLM)** built entirely from scratch using PyTorch. This project covers the full pipeline from **tokenization and Transformer architecture design** to **pretraining, GPT-2 weight loading, text generation, checkpointing, and downstream fine-tuning for SMS spam classification**.

---

## 🚀 Highlights

* GPT-style decoder-only Transformer built entirely from scratch
* Custom tokenization and preprocessing pipeline
* Self-Attention, Causal Attention, and Multi-Head Attention implementations
* Custom LayerNorm, GELU, and Feed-Forward Network (FFN) modules
* Next-token prediction training pipeline
* GPT-2 pretrained weight loading for all official model variants
* Temperature scaling and Top-k sampling for text generation
* Model and optimizer checkpointing for training resumption
* Cross-Entropy Loss and Perplexity evaluation
* Fine-tuning on supervised classification tasks
* SMS Spam Detection using a fine-tuned GPT model
* Modular and extensible PyTorch implementation

---

## ⚙️ Preprocessing Pipeline

### 1. Load Raw Text

The training corpus is loaded as raw text.

### 2. Tokenization

A custom tokenizer is implemented using regular expressions.

```python
import re

tokens = re.findall(r"\b\w+\b", raw_text.lower())
```

### 3. Vocabulary Construction

Word-to-index mappings are created with support for an `<UNK>` token to handle unseen words.

### 4. Encoding & Decoding

Utility functions convert:

```text
Raw Text ↔ Token IDs
```

for both training and inference.

### 5. Embeddings

* Token Embeddings
* Learned Positional Embeddings

are combined before entering the Transformer.

---

## 🧠 Transformer Architecture

### Self-Attention

Computes relationships between tokens using scaled dot-product attention.

### Causal Attention

Applies a lower-triangular mask to ensure tokens only attend to past and current positions.

### Multi-Head Attention

Multiple attention heads run in parallel to capture information from different representation subspaces.

---

### Transformer Block

Each Transformer block follows:

```text
Multi-Head Attention
        ↓
Residual Connection
        ↓
Layer Normalization
        ↓
Feed-Forward Network
        ↓
Residual Connection
        ↓
Layer Normalization
```

Components include:

* Multi-Head Attention
* Feed-Forward Network (FFN)
* GELU Activation
* Layer Normalization
* Residual Connections

---

## 🏗️ GPT-Style Model

```text
Token Embeddings
        +
Positional Embeddings
        ↓
Stacked Transformer Blocks
        ↓
Final LayerNorm
        ↓
Linear Projection
        ↓
Vocabulary Logits
```

The model follows a decoder-only Transformer architecture similar to GPT.

---

## 🏋️ Pretraining

The model is trained using:

* Next-token prediction
* Cross-Entropy Loss
* AdamW Optimizer

Training utilities include:

* Training loss tracking
* Validation loss tracking
* Perplexity evaluation
* Learning curve visualization

---

## 🎲 Text Generation

Supports autoregressive generation using:

### Temperature Scaling

Controls randomness during generation.

* Low Temperature → More deterministic
* High Temperature → More creative

### Top-k Sampling

Restricts sampling to the top-k most likely tokens.

### Multinomial Sampling

Samples tokens according to probability distributions.

### EOS Handling

Stops generation when an end-of-sequence token is produced.

---

### Generation Pipeline

```text
Input Text
    ↓
Token IDs
    ↓
GPT Model
    ↓
Next-Token Probabilities
    ↓
Temperature Scaling
    ↓
Top-k Filtering
    ↓
Multinomial Sampling
    ↓
Generated Text
```

---

## 🔄 GPT-2 Weight Loading

Supports loading all official GPT-2 variants:

| Model        | Parameters |
| ------------ | ---------- |
| GPT-2 Small  | 124M       |
| GPT-2 Medium | 355M       |
| GPT-2 Large  | 774M       |
| GPT-2 XL     | 1.5B       |

The project includes utilities to:

* Download GPT-2 checkpoints
* Map weights into the custom GPT implementation
* Validate tensor shapes before assignment

Loaded components include:

* Token embeddings
* Positional embeddings
* Attention weights
* Feed-forward layers
* LayerNorm parameters
* Output head

---

## 💾 Checkpointing

Full training state (model weights, optimizer state, momentum statistics) persisted and restored via `torch.save` / `torch.load` for seamless training resumption.

---

## 📊 Fine-Tuning for SMS Spam Classification

The pretrained GPT model is fine-tuned on the SMS Spam Collection dataset for binary text classification.

### Dataset Preparation

* Downloaded from the UCI repository
* Loaded into a Pandas DataFrame
* Labels mapped:

```text
ham  → 0
spam → 1
```

### Class Balancing

The majority class (ham) is randomly downsampled to match the spam class size.

### Dataset Splits

The dataset is shuffled and divided into:

* Training Set
* Validation Set
* Test Set

### Custom Dataset

A custom `SpamDataset`:

* Tokenizes messages
* Truncates long sequences
* Pads shorter sequences
* Returns tensors suitable for GPT fine-tuning

### DataLoaders

* Training DataLoader (shuffle enabled)
* Validation DataLoader
* Test DataLoader

---

### Fine-Tuning Pipeline

```text
Raw SMS Text
    ↓
Tokenization
    ↓
Padding / Truncation
    ↓
SpamDataset
    ↓
DataLoader
    ↓
GPT Fine-Tuning
    ↓
Spam / Ham Prediction
```

---

## 📈 Results

### SMS Spam Classification

| Epoch | Train Accuracy | Validation Accuracy |
| ----- | -------------- | ------------------- |
| 1     | 70.0%          | 72.5%               |
| 2     | 82.5%          | 85.0%               |
| 3     | 90.0%          | 90.0%               |
| 4     | 100.0%         | 97.5%               |
| 5     | 100.0%         | 97.5%               |

**Training Time:** 15.82 minutes

The model successfully learned to distinguish spam from legitimate SMS messages while maintaining strong validation performance.

---

## 📦 Tech Stack

### Language

* Python

### Deep Learning

* PyTorch

### Data Processing

* NumPy
* Pandas
* Regular Expressions (re)

### Visualization

* Matplotlib

---

## 🔄 End-to-End Pipeline

```text
Raw Text
    ↓
Tokenization
    ↓
Vocabulary & Encoding
    ↓
Embeddings
    ↓
Transformer Layers
    ↓
Next Token Prediction
    ↓
Pretraining
    ↓
GPT-2 Weight Loading
    ↓
Checkpoint Save / Load
    ↓
Text Generation
    ↓
Fine-Tuning
    ↓
SMS Spam Classification
```

---

## 👨‍💻 Author

Built as a foundational deep learning project to understand and implement the core mechanics behind modern Large Language Models (LLMs), moving beyond APIs to develop every major component from first principles using PyTorch.