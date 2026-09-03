
import time
from pathlib import Path

import torch
import tiktoken

from llm.gpt.config import GPT_CONFIG_124M
from llm.weights.gpt2_loader import load_pretrained_gpt2
from llm.preprocessing.dataset import create_data_loader
from llm.training.loss import calc_loss_batch, perplexity
from llm.training.trainer import train_model_simple


# Configuration

TRAIN_DATA_PATH = Path(
    "data/wikitext-103/train.txt"
)

VAL_DATA_PATH = Path(
    "data/wikitext-103/validation.txt"
)

VERDICT_DATA_PATH = Path(
    "data/the-verdict.txt"
)

CHECKPOINT_PATH = Path(
    "checkpoints/gpt2_wikitext.pt"
)


# Training configuration
BATCH_SIZE = 1
NUM_EPOCHS = 4
MAX_STEPS = 10_000

# Resume from checkpoint if it exists
RESUME_TRAINING = True

# Evaluation
EVAL_FREQ = 500
EVAL_ITER = 5

# Optimizer
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1

# Text generation
START_CONTEXT = "The history of the world"


# Device

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# Load WikiText-103

print("\nLoading WikiText dataset...")

with open(
    TRAIN_DATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    train_text = f.read()

with open(
    VAL_DATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    val_text = f.read()


print(
    f"Training characters:   {len(train_text):,}"
)

print(
    f"Validation characters: {len(val_text):,}"
)


# Tokenizer

tokenizer = tiktoken.get_encoding("gpt2")

train_tokens = len(
    tokenizer.encode(train_text)
)

val_tokens = len(
    tokenizer.encode(val_text)
)


print(
    f"Training tokens:        {train_tokens:,}"
)

print(
    f"Validation tokens:      {val_tokens:,}"
)


# Context length

context_length = GPT_CONFIG_124M[
    "context_length"
]

print(
    f"Context length:         {context_length}"
)


# Training DataLoader

print("\nCreating training DataLoader...")

train_loader = create_data_loader(
    train_text,
    batch_size=BATCH_SIZE,
    max_length=context_length,
    stride=context_length,
    drop_last=True,
    shuffle=True,
    num_workers=0,
)


# Validation DataLoader

print("Creating validation DataLoader...")

val_loader = create_data_loader(
    val_text,
    batch_size=BATCH_SIZE,
    max_length=context_length,
    stride=context_length,
    drop_last=False,
    shuffle=False,
    num_workers=0,
)


print(
    f"Training batches:   {len(train_loader):,}"
)

print(
    f"Validation batches: {len(val_loader):,}"
)


# Load pretrained GPT-2

print("\nInitializing GPT-2 124M...")

model = load_pretrained_gpt2(
    model_size="124M",
    models_dir="gpt2",
    device=device,
)

print(
    "Pretrained GPT-2 124M loaded successfully."
)


# Optimizer

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# Checkpoint / Resume configuration

resume_checkpoint = None

if (
    RESUME_TRAINING
    and CHECKPOINT_PATH.exists()
):
    resume_checkpoint = str(
        CHECKPOINT_PATH
    )

    print(
        f"\nCheckpoint found: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        "Training will resume "
        "from the checkpoint."
    )

elif RESUME_TRAINING:
    print(
        "\nNo checkpoint found."
    )

    print(
        "Training will start from "
        "pretrained GPT-2 weights."
    )

else:
    print(
        "\nFresh training mode enabled."
    )

    print(
        "Training will start from "
        "pretrained GPT-2 weights."
    )


# Training

print(
    "\nStarting continued pretraining..."
)

print(
    f"Maximum steps: {MAX_STEPS:,}"
)

start_time = time.time()


train_losses, val_losses, tokens_seen = (
    train_model_simple(
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
        checkpoint_path=str(
            CHECKPOINT_PATH
        ),
        config=GPT_CONFIG_124M,
        max_steps=MAX_STEPS,
        resume_from_checkpoint=(
            resume_checkpoint
        ),
    )
)


training_time = (
    time.time() - start_time
) / 60


# Training Summary

print(
    "\n========================================"
)

print(
    "Training completed."
)

print(
    "========================================"
)

print(
    f"Training time: "
    f"{training_time:.2f} minutes"
)


if tokens_seen:
    print(
        f"Tokens processed: "
        f"{tokens_seen[-1]:,}"
    )
else:
    print(
        "Tokens processed: 0"
    )


if train_losses and val_losses:

    final_train_loss = (
        train_losses[-1]
    )

    final_val_loss = (
        val_losses[-1]
    )

    train_ppl = perplexity(
        torch.tensor(
            final_train_loss
        )
    )

    val_ppl = perplexity(
        torch.tensor(
            final_val_loss
        )
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


# The Verdict Evaluation

print(
    "\n========================================"
)

print(
    "The Verdict evaluation"
)

print(
    "========================================"
)


with open(
    VERDICT_DATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    verdict_text = f.read()


verdict_loader = create_data_loader(
    verdict_text,
    batch_size=BATCH_SIZE,
    max_length=context_length,
    stride=context_length,
    drop_last=False,
    shuffle=False,
    num_workers=0,
)


model.eval()

verdict_loss = 0.0
batches = 0


with torch.no_grad():

    for input_batch, target_batch in (
        verdict_loader
    ):

        loss = calc_loss_batch(
            input_batch,
            target_batch,
            model,
            device,
        )

        verdict_loss += loss.item()

        batches += 1

        if batches >= EVAL_ITER:
            break


if batches > 0:

    verdict_loss /= batches

    verdict_ppl = perplexity(
        torch.tensor(
            verdict_loss
        )
    )

    print(
        f"The Verdict loss: "
        f"{verdict_loss:.4f}"
    )

    print(
        f"The Verdict perplexity: "
        f"{verdict_ppl.item():.4f}"
    )

else:

    print(
        "Not enough data for "
        "The Verdict evaluation."
    )

