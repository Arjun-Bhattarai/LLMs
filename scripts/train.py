
import time
import math
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
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
    os.environ.get("CHECKPOINT_PATH", "checkpoints/latest_checkpoint.pt")
)
BEST_CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
METRICS_PATH = Path("results/training_metrics.csv")


# Training configuration
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "1"))
NUM_EPOCHS = 100
MAX_STEPS = int(os.environ.get("MAX_STEPS", "300000"))

# Resume from checkpoint if it exists
RESUME_TRAINING = os.environ.get("RESUME_TRAINING", "1") == "1"
DDP = int(os.environ.get("WORLD_SIZE", "1")) > 1
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))

# Evaluation
EVAL_FREQ = 500
EVAL_ITER = 5

# Optimizer
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1

# Text generation
START_CONTEXT = "The history of the world"


# Device

if DDP:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda", LOCAL_RANK)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    shuffle=not DDP,
    num_workers=NUM_WORKERS,
)
if NUM_WORKERS > 0:
    train_loader.pin_memory = True
    train_loader.persistent_workers = True
if DDP:
    train_sampler = DistributedSampler(train_loader.dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        drop_last=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)


# Validation DataLoader

print("Creating validation DataLoader...")

val_loader = create_data_loader(
    val_text,
    batch_size=BATCH_SIZE,
    max_length=context_length,
    stride=context_length,
    drop_last=False,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
if NUM_WORKERS > 0:
    val_loader.pin_memory = True
    val_loader.persistent_workers = True


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
    [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2], "weight_decay": 0.0},
    ],
    lr=LEARNING_RATE, betas=(0.9, 0.95),
)

# The legacy 100K checkpoint has optimizer LR 1e-5 but no scheduler state.
# Cosine decays over absolute steps 100K..300K; warm resume does not restart Adam.
resume_step = 100_000
if RESUME_TRAINING and CHECKPOINT_PATH.exists():
    try:
        meta = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        resume_step = int(meta.get("global_step", resume_step))
    except Exception as exc:
        print(f"Could not inspect resume checkpoint metadata: {exc}")
schedule_steps = max(1, MAX_STEPS - resume_step)
def lr_lambda(step):
    progress = min(max(step / schedule_steps, 0.0), 1.0)
    return (1e-6 + 0.5 * (LEARNING_RATE - 1e-6) * (1 + math.cos(math.pi * progress))) / LEARNING_RATE
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
if DDP:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])


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
        scheduler=scheduler,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        metrics_path=str(METRICS_PATH),
        best_checkpoint_path=str(BEST_CHECKPOINT_PATH),
        rank=RANK, world_size=WORLD_SIZE,
    )
)
if DDP:
    dist.destroy_process_group()


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

