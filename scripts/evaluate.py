import json
import math
import os

import torch
import tiktoken

from llm.gpt.config import GPT_CONFIG_124M
from llm.weights.gpt2_loader import load_pretrained_gpt2
from llm.decoding.sampling import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


# Configuration

CHECKPOINT_PATH = "checkpoints/gpt2_wikitext.pt"

VAL_PATH = "data/wikitext-103/validation.txt"
VERDICT_PATH = "data/the-verdict.txt"

RESULTS_DIR = "results"
EVALUATION_JSON_PATH = os.path.join(
    RESULTS_DIR,
    "evaluation.json",
)
GENERATION_TXT_PATH = os.path.join(
    RESULTS_DIR,
    "generation_samples.txt",
)

CONTEXT_LENGTH = GPT_CONFIG_124M["context_length"]

# Use the same fixed number of batches for a fair comparison.
VAL_EVAL_BATCHES = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Helpers

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_eval_batches(
    text,
    tokenizer,
    context_length,
    max_batches=None,
):
    """
    Create deterministic, non-overlapping evaluation batches.

    Each batch contains:
        input  = tokens[i : i + context_length]
        target = tokens[i + 1 : i + context_length + 1]
    """

    token_ids = tokenizer.encode(text)

    batches = []

    for i in range(
        0,
        len(token_ids) - context_length,
        context_length,
    ):
        input_ids = token_ids[
            i : i + context_length
        ]

        target_ids = token_ids[
            i + 1 : i + context_length + 1
        ]

        if (
            len(input_ids) != context_length
            or len(target_ids) != context_length
        ):
            continue

        input_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
        ).unsqueeze(0)

        target_tensor = torch.tensor(
            target_ids,
            dtype=torch.long,
        ).unsqueeze(0)

        batches.append(
            (
                input_tensor,
                target_tensor,
            )
        )

        if (
            max_batches is not None
            and len(batches) >= max_batches
        ):
            break

    return batches


def calculate_loss(
    model,
    batches,
    device,
):
    """
    Calculate average cross-entropy loss
    over a fixed collection of batches.
    """

    model.eval()

    total_loss = 0.0

    use_amp = device.type == "cuda"

    with torch.no_grad():

        for input_batch, target_batch in batches:

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(input_batch)

            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten(),
            )

            total_loss += loss.item()

    return total_loss / len(batches)


def calculate_perplexity(loss):
    return math.exp(loss)


def load_trained_model():
    """
    Load the original GPT-2 architecture/weights first,
    then replace them with the 10K checkpoint weights.
    """

    model = load_pretrained_gpt2(
        model_size="124M",
        models_dir="gpt2",
        device=DEVICE,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def generate_sample(
    model,
    prompt,
    tokenizer,
    max_new_tokens=80,
):
    """
    Generate text using the existing project's
    simple autoregressive decoding implementation.
    """

    model.eval()

    context_size = (
        model.pos_emb.weight.shape[0]
    )

    encoded = text_to_token_ids(
        prompt,
        tokenizer,
    ).to(DEVICE)

    with torch.no_grad():

        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
        )

    return token_ids_to_text(
        token_ids,
        tokenizer,
    )


# Main

def main():

    # Make sure results directory exists.
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    print("=" * 72)
    print("GPT-2 124M EVALUATION")
    print("=" * 72)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()

    # ========================================================
    # Tokenizer
    # ========================================================

    print("Loading GPT-2 tokenizer...")

    tokenizer = tiktoken.get_encoding(
        "gpt2"
    )

    print("✓ Tokenizer loaded.")
    print()

    # ========================================================
    # Dataset
    # ========================================================

    print("Loading evaluation datasets...")

    validation_text = load_text(
        VAL_PATH
    )

    verdict_text = load_text(
        VERDICT_PATH
    )

    validation_batches = create_eval_batches(
        validation_text,
        tokenizer,
        CONTEXT_LENGTH,
        max_batches=VAL_EVAL_BATCHES,
    )

    verdict_batches = create_eval_batches(
        verdict_text,
        tokenizer,
        CONTEXT_LENGTH,
        max_batches=None,
    )

    print(
        f"WikiText validation batches: "
        f"{len(validation_batches)}"
    )

    print(
        f"The Verdict batches: "
        f"{len(verdict_batches)}"
    )

    if not validation_batches:
        raise RuntimeError(
            "No WikiText validation batches were created."
        )

    if not verdict_batches:
        raise RuntimeError(
            "No The Verdict evaluation batches were created."
        )

    print()

    # ========================================================
    # Load baseline GPT-2
    # ========================================================

    print("-" * 72)
    print("1. ORIGINAL PRETRAINED GPT-2")
    print("-" * 72)

    print(
        "Loading original GPT-2 124M..."
    )

    baseline_model = load_pretrained_gpt2(
        model_size="124M",
        models_dir="gpt2",
        device=DEVICE,
    )

    baseline_model.eval()

    print("✓ Original GPT-2 loaded.")
    print()

    # ========================================================
    # Load trained checkpoint
    # ========================================================

    print("-" * 72)
    print("2. 10K CONTINUED-PRETRAINED GPT-2")
    print("-" * 72)

    print(
        "Loading 10K checkpoint..."
    )

    trained_model, checkpoint = (
        load_trained_model()
    )

    print("✓ 10K checkpoint loaded.")

    print(
        f"Checkpoint step: "
        f"{checkpoint['global_step']}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print()

    # ========================================================
    # WikiText evaluation
    # ========================================================

    print("=" * 72)
    print("WIKITEXT-103 VALIDATION")
    print("=" * 72)

    print(
        f"Evaluating both models on "
        f"{len(validation_batches)} identical batches..."
    )

    baseline_val_loss = calculate_loss(
        baseline_model,
        validation_batches,
        DEVICE,
    )

    trained_val_loss = calculate_loss(
        trained_model,
        validation_batches,
        DEVICE,
    )

    baseline_val_ppl = calculate_perplexity(
        baseline_val_loss
    )

    trained_val_ppl = calculate_perplexity(
        trained_val_loss
    )

    loss_improvement = (
        (baseline_val_loss - trained_val_loss)
        / baseline_val_loss
        * 100
    )

    ppl_improvement = (
        (baseline_val_ppl - trained_val_ppl)
        / baseline_val_ppl
        * 100
    )

    print()

    print(
        f"Original GPT-2 loss : "
        f"{baseline_val_loss:.4f}"
    )

    print(
        f"10K GPT-2 loss      : "
        f"{trained_val_loss:.4f}"
    )

    print()

    print(
        f"Original GPT-2 PPL  : "
        f"{baseline_val_ppl:.4f}"
    )

    print(
        f"10K GPT-2 PPL       : "
        f"{trained_val_ppl:.4f}"
    )

    print()

    print(
        f"Loss improvement     : "
        f"{loss_improvement:.2f}%"
    )

    print(
        f"PPL improvement      : "
        f"{ppl_improvement:.2f}%"
    )

    print()

    # ========================================================
    # The Verdict evaluation
    # ========================================================

    print("=" * 72)
    print("THE VERDICT")
    print("=" * 72)

    print(
        f"Evaluating both models on "
        f"{len(verdict_batches)} batches..."
    )

    baseline_verdict_loss = calculate_loss(
        baseline_model,
        verdict_batches,
        DEVICE,
    )

    trained_verdict_loss = calculate_loss(
        trained_model,
        verdict_batches,
        DEVICE,
    )

    baseline_verdict_ppl = (
        calculate_perplexity(
            baseline_verdict_loss
        )
    )

    trained_verdict_ppl = (
        calculate_perplexity(
            trained_verdict_loss
        )
    )

    print()

    print(
        f"Original GPT-2 loss : "
        f"{baseline_verdict_loss:.4f}"
    )

    print(
        f"10K GPT-2 loss      : "
        f"{trained_verdict_loss:.4f}"
    )

    print()

    print(
        f"Original GPT-2 PPL  : "
        f"{baseline_verdict_ppl:.4f}"
    )

    print(
        f"10K GPT-2 PPL       : "
        f"{trained_verdict_ppl:.4f}"
    )

    print()

    # ========================================================
    # Text generation
    # ========================================================

    print("=" * 72)
    print("TEXT GENERATION COMPARISON")
    print("=" * 72)

    prompts = [
        "The history of the world",
        "Artificial intelligence is",
        "The future of technology",
    ]

    generation_results = []

    for prompt in prompts:

        print()
        print("-" * 72)
        print(f"PROMPT: {prompt}")
        print("-" * 72)

        print()
        print("[Original GPT-2]")
        print()

        baseline_text = generate_sample(
            baseline_model,
            prompt,
            tokenizer,
        )

        print(
            baseline_text
        )

        print()
        print("[10K WikiText GPT-2]")
        print()

        trained_text = generate_sample(
            trained_model,
            prompt,
            tokenizer,
        )

        print(
            trained_text
        )

        generation_results.append(
            {
                "prompt": prompt,
                "original_gpt2": baseline_text,
                "trained_gpt2": trained_text,
            }
        )

    # ========================================================
    # Save evaluation metrics
    # ========================================================

    evaluation_results = {
        "model": {
            "architecture": "GPT-2 124M",
            "parameters": 163037184,
            "training_method": "continued_pretraining",
            "training_dataset": "WikiText-103",
            "checkpoint": CHECKPOINT_PATH,
            "checkpoint_step": checkpoint["global_step"],
            "checkpoint_epoch": checkpoint["epoch"],
        },
        "evaluation": {
            "device": str(DEVICE),
            "context_length": CONTEXT_LENGTH,
            "wikitext_validation_batches": len(
                validation_batches
            ),
            "the_verdict_batches": len(
                verdict_batches
            ),
        },
        "wikitext_103_validation": {
            "baseline_loss": baseline_val_loss,
            "trained_loss": trained_val_loss,
            "baseline_perplexity": baseline_val_ppl,
            "trained_perplexity": trained_val_ppl,
            "loss_improvement_percent": loss_improvement,
            "perplexity_improvement_percent": ppl_improvement,
        },
        "the_verdict": {
            "baseline_loss": baseline_verdict_loss,
            "trained_loss": trained_verdict_loss,
            "baseline_perplexity": baseline_verdict_ppl,
            "trained_perplexity": trained_verdict_ppl,
        },
    }

    with open(
        EVALUATION_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            evaluation_results,
            f,
            indent=4,
        )

    # ========================================================
    # Save generation samples
    # ========================================================

    with open(
        GENERATION_TXT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "GPT-2 124M GENERATION COMPARISON\n"
        )
        f.write("=" * 72)
        f.write("\n\n")

        f.write(
            "Original model: OpenAI GPT-2 124M\n"
        )
        f.write(
            "Trained model: GPT-2 124M continued "
            "pretraining on WikiText-103\n"
        )
        f.write(
            f"Checkpoint step: "
            f"{checkpoint['global_step']}\n"
        )
        f.write("\n")

        for result in generation_results:

            f.write("-" * 72)
            f.write("\n")

            f.write(
                f"PROMPT: {result['prompt']}\n"
            )

            f.write("-" * 72)
            f.write("\n\n")

            f.write("[Original GPT-2]\n\n")
            f.write(
                result["original_gpt2"]
            )
            f.write("\n\n")

            f.write(
                "[10K WikiText GPT-2]\n\n"
            )
            f.write(
                result["trained_gpt2"]
            )
            f.write("\n\n")

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print()
    print("Model:")
    print("  GPT-2 124M")
    print("  Continued pretraining on WikiText-103")
    print(
        f"  Checkpoint step: "
        f"{checkpoint['global_step']}"
    )
    print()

    print("WikiText-103:")
    print(
        f"  Original PPL : "
        f"{baseline_val_ppl:.4f}"
    )
    print(
        f"  Trained PPL  : "
        f"{trained_val_ppl:.4f}"
    )
    print(
        f"  Improvement  : "
        f"{ppl_improvement:.2f}%"
    )
    print()

    print("The Verdict:")
    print(
        f"  Original PPL : "
        f"{baseline_verdict_ppl:.4f}"
    )
    print(
        f"  Trained PPL  : "
        f"{trained_verdict_ppl:.4f}"
    )
    print()

    print("Results saved:")
    print(
        f"  {EVALUATION_JSON_PATH}"
    )
    print(
        f"  {GENERATION_TXT_PATH}"
    )

    print()

    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
