import math
import torch
import tiktoken

from llm.gpt.model import GPTModel
from llm.gpt.config import GPT_CONFIG_124M
from llm.weights.gpt2_loader import load_pretrained_gpt2
from llm.preprocessing.dataset import create_data_loader
from llm.training.loss import calc_loss_loader



# Configuration
#The benchmark is basically a fair test between your original GPT-2 and your trained GPT-2.

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = "checkpoints/gpt2_wikitext.pt"

# Use a small number of batches so the benchmark
# doesn't take a very long time.
EVAL_BATCHES = 20

BATCH_SIZE = 1
CONTEXT_LENGTH = 1024
STRIDE = 1024

DATA_PATH = "data/wikitext-103/validation.txt"



# Load validation data


print("=" * 60)
print("GPT-2 Benchmark")
print("=" * 60)

print(f"Device: {DEVICE}")
print("Loading validation data...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

val_loader = create_data_loader(
    text=text,
    batch_size=BATCH_SIZE,
    max_length=CONTEXT_LENGTH,
    stride=STRIDE,
    shuffle=False,
    drop_last=True,
)

print("Validation data loaded.")



# Original GPT-2


print("\n" + "=" * 60)
print("1. ORIGINAL GPT-2 124M")
print("=" * 60)

original_model = load_pretrained_gpt2(
    model_size="124M",
    models_dir="gpt2",
    device=DEVICE,
)

original_model.eval()

with torch.no_grad():
    original_loss = calc_loss_loader(
        val_loader,
        original_model,
        DEVICE,
        num_batches=EVAL_BATCHES,
    )

original_ppl = math.exp(original_loss)

print(f"Original GPT-2 loss:       {original_loss:.4f}")
print(f"Original GPT-2 perplexity: {original_ppl:.4f}")

del original_model

if torch.cuda.is_available():
    torch.cuda.empty_cache()



# WikiText-trained GPT-2


print("\n" + "=" * 60)
print("2. WIKITEXT-TRAINED GPT-2")
print("=" * 60)

trained_model = GPTModel(GPT_CONFIG_124M)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

trained_model.load_state_dict(
    checkpoint["model_state_dict"]
)

trained_model.to(DEVICE)
trained_model.eval()

with torch.no_grad():
    trained_loss = calc_loss_loader(
        val_loader,
        trained_model,
        DEVICE,
        num_batches=EVAL_BATCHES,
    )

trained_ppl = math.exp(trained_loss)

print(f"Trained GPT-2 loss:       {trained_loss:.4f}")
print(f"Trained GPT-2 perplexity: {trained_ppl:.4f}")



# Comparison


print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"Original GPT-2:")
print(f"  Loss:       {original_loss:.4f}")
print(f"  Perplexity: {original_ppl:.4f}")

print()

print(f"WikiText-trained GPT-2:")
print(f"  Loss:       {trained_loss:.4f}")
print(f"  Perplexity: {trained_ppl:.4f}")

print()

loss_improvement = (
    (original_loss - trained_loss)
    / original_loss
) * 100

ppl_improvement = (
    (original_ppl - trained_ppl)
    / original_ppl
) * 100

print(f"Loss improvement:       {loss_improvement:.2f}%")
print(f"Perplexity improvement: {ppl_improvement:.2f}%")

print("=" * 60)