from pathlib import Path

import torch
import tiktoken

from llm.gpt.config import GPT_CONFIG_124M
from llm.weights.gpt2_loader import load_pretrained_gpt2
from llm.training.checkpoint import load_checkpoint


# Configuration

CHECKPOINT_PATH = Path(
    "checkpoints/gpt2_wikitext.pt"
)


# Device

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("GPT-2 TRAINING SETUP VERIFICATION")
print("=" * 60)

print(f"\nDevice: {device}")


# Check checkpoint exists

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found: "
        f"{CHECKPOINT_PATH}"
    )

print(
    f"Checkpoint: {CHECKPOINT_PATH}"
)

print(
    f"Checkpoint size: "
    f"{CHECKPOINT_PATH.stat().st_size / (1024 ** 3):.2f} GB"
)


# Load pretrained GPT-2 architecture

print(
    "\n[1/5] Loading pretrained GPT-2 124M..."
)

model = load_pretrained_gpt2(
    model_size="124M",
    models_dir="gpt2",
    device=device,
)

print(
    "✓ Pretrained GPT-2 loaded."
)


# Create optimizer

print(
    "\n[2/5] Creating optimizer..."
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.1,
)

print(
    "✓ Optimizer created."
)


# Create AMP scaler

print(
    "\n[3/5] Creating AMP GradScaler..."
)

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)

print(
    f"✓ AMP enabled: {use_amp}"
)


# Load checkpoint

print(
    "\n[4/5] Loading 10K checkpoint..."
)

checkpoint = load_checkpoint(
    model=model,
    optimizer=optimizer,
    scaler=scaler,
    path=str(CHECKPOINT_PATH),
    device=device,
)

print(
    "✓ Checkpoint loaded successfully."
)


# Verify checkpoint metadata

print(
    "\n[5/5] Verifying checkpoint..."
)

required_keys = {
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "global_step",
    "train_loss",
    "val_loss",
    "config",
}

checkpoint_keys = set(
    checkpoint.keys()
)

missing_keys = (
    required_keys - checkpoint_keys
)

if missing_keys:

    raise RuntimeError(
        f"Missing checkpoint keys: "
        f"{missing_keys}"
    )

print(
    "✓ Required checkpoint keys present."
)


# Model parameter count

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"\nModel parameters: "
    f"{parameter_count:,}"
)


# Verify configuration

print(
    "\nModel configuration:"
)

for key, value in GPT_CONFIG_124M.items():

    print(
        f"  {key}: {value}"
    )


# Verify training state

global_step = checkpoint[
    "global_step"
]

epoch = checkpoint[
    "epoch"
]

train_loss = checkpoint[
    "train_loss"
]

val_loss = checkpoint[
    "val_loss"
]


print(
    "\nCheckpoint training state:"
)

print(
    f"  Global step:  {global_step}"
)

print(
    f"  Epoch:        {epoch}"
)

print(
    f"  Train loss:   {train_loss:.6f}"
)

print(
    f"  Val loss:     {val_loss:.6f}"
)


# Verify model tensors

print(
    "\nChecking model tensors..."
)

invalid_parameters = []

for name, parameter in model.named_parameters():

    if not torch.isfinite(
        parameter
    ).all():

        invalid_parameters.append(
            name
        )


if invalid_parameters:

    raise RuntimeError(
        "Non-finite parameters found:\n"
        + "\n".join(
            invalid_parameters
        )
    )

print(
    "✓ All model parameters are finite."
)


# Verify tokenizer

print(
    "\nChecking GPT-2 tokenizer..."
)

tokenizer = tiktoken.get_encoding(
    "gpt2"
)

test_text = (
    "The history of the world"
)

tokens = tokenizer.encode(
    test_text
)

print(
    f"Test text: {test_text}"
)

print(
    f"Token IDs: {tokens}"
)

print(
    f"Token count: {len(tokens)}"
)

if not tokens:

    raise RuntimeError(
        "Tokenizer returned no tokens."
    )

print(
    "✓ GPT-2 tokenizer working."
)


# Forward-pass verification

print(
    "\nRunning zero-training forward pass..."
)

model.eval()

test_input = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device,
)

with torch.no_grad():

    logits = model(
        test_input
    )

print(
    f"Input shape:  "
    f"{tuple(test_input.shape)}"
)

print(
    f"Logits shape: "
    f"{tuple(logits.shape)}"
)

expected_vocab_size = (
    GPT_CONFIG_124M["vocab_size"]
)

if logits.shape[-1] != expected_vocab_size:

    raise RuntimeError(
        "Unexpected vocabulary dimension: "
        f"{logits.shape[-1]}"
    )

if not torch.isfinite(
    logits
).all():

    raise RuntimeError(
        "Forward pass produced "
        "non-finite logits."
    )

print(
    "✓ Forward pass successful."
)


# Final result

print("\n" + "=" * 60)
print("VERIFICATION PASSED")
print("=" * 60)

print(
    "\nYour 10K checkpoint is compatible "
    "with the updated training code."
)

print(
    "No training steps were performed."
)

print(
    "The checkpoint was not modified."
)

print(
    f"\nFinal checkpoint step: "
    f"{global_step}"
)

print(
    f"Final validation loss: "
    f"{val_loss:.6f}"
)

