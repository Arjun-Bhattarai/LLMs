import time
from pathlib import Path

import torch
import tiktoken

from llm.gpt.config import GPT_CONFIG_124M
from llm.gpt.model import GPTModel
from llm.training.loss import perplexity
from llm.preprocessing.dataset import create_data_loader
from llm.training.trainer import train_model_simple


# Configuration

DATA_PATH = Path("data/the-verdict.txt")
CHECKPOINT_PATH = Path("checkpoints/gpt_checkpoint.pt")

TRAIN_RATIO = 0.90

BATCH_SIZE = 2
NUM_EPOCHS = 10

EVAL_FREQ = 50
EVAL_ITER = 5

LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1

START_CONTEXT = "Every effort moves you"


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# Load dataset

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text_data = f.read()

print(f"Characters: {len(text_data)}")


# Tokenizer

tokenizer = tiktoken.get_encoding("gpt2")

total_tokens = len(tokenizer.encode(text_data))

print(f"Tokens: {total_tokens}")


# Train / validation split

split_idx = int(TRAIN_RATIO * len(text_data))

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

print(f"Training characters: {len(train_data)}")
print(f"Validation characters: {len(val_data)}")


# Create DataLoaders

train_loader = create_data_loader(
    train_data,
    batch_size=BATCH_SIZE,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_data_loader(
    val_data,
    batch_size=BATCH_SIZE,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# Create model

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)

model.to(device)

print("Model created.")



# Optimizer

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# Training

print("\nStarting training...\n")

start_time = time.time()

train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=NUM_EPOCHS,
    eval_freq=EVAL_FREQ,
    eval_iter=EVAL_ITER,
    start_context=START_CONTEXT,
    tokenizer=tokenizer,
    checkpoint_path=str(CHECKPOINT_PATH),
    config=GPT_CONFIG_124M,
)

end_time = time.time()


# Training summary

end_time = time.time()

training_time = (
    end_time - start_time
) / 60

print("\nTraining completed.")

print(
    f"Training time: "
    f"{training_time:.2f} minutes"
)

if train_losses and val_losses:

    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]

    train_ppl = perplexity(
        torch.tensor(final_train_loss)
    )

    val_ppl = perplexity(
        torch.tensor(final_val_loss)
    )

    print(
        f"Final training loss: "
        f"{final_train_loss:.4f}"
    )

    print(
        f"Final validation loss: "
        f"{final_val_loss:.4f}"
    )

    print(
        f"Final training perplexity: "
        f"{train_ppl.item():.4f}"
    )

    print(
        f"Final validation perplexity: "
        f"{val_ppl.item():.4f}"
    )

print(
    f"Checkpoint: "
    f"{CHECKPOINT_PATH}"
)