import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tiktoken

from llm.decoding.sampling import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)
from llm.gpt.model import GPTModel


CHECKPOINTS = {
    "50K": Path("checkpoints/gpt2_wikitext_50k.pt"),
    "100K": Path("checkpoints/gpt2_wikitext_100k.pt"),
}

VALIDATION_PATH = Path(
    "data/wikitext-103/validation.txt"
)

RESULTS_DIR = Path("results/checkpoint_comparison")

PROMPTS = [
    "The history of the world",
    "The development of modern science",
    "In the future, technology will",
]

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 50
EVAL_BATCHES = 20
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )


def build_model(checkpoint):
    config = checkpoint.get("config")

    if config is None:
        raise KeyError(
            "Checkpoint does not contain a model config."
        )

    model = GPTModel(config).to(DEVICE)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, config


def create_validation_batches(
    text,
    tokenizer,
    context_length,
):
    token_ids = tokenizer.encode(text)
    batches = []

    for start in range(
        0,
        len(token_ids) - context_length,
        context_length,
    ):
        input_ids = token_ids[
            start : start + context_length
        ]
        target_ids = token_ids[
            start + 1 : start + context_length + 1
        ]

        if len(target_ids) != context_length:
            continue

        batches.append(
            (
                torch.tensor(
                    input_ids,
                    dtype=torch.long,
                ).unsqueeze(0),
                torch.tensor(
                    target_ids,
                    dtype=torch.long,
                ).unsqueeze(0),
            )
        )

        if len(batches) >= EVAL_BATCHES:
            break

    if not batches:
        raise ValueError(
            "Validation text is too short for the configured context length."
        )

    return batches


def evaluate_model(model, batches):
    total_loss = 0.0
    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():
        for input_batch, target_batch in batches:
            input_batch = input_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(input_batch)

            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten(),
            )
            total_loss += loss.item()

    loss = total_loss / len(batches)
    return loss, math.exp(loss)


def generate_sample(model, tokenizer, config, prompt):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    input_ids = text_to_token_ids(
        prompt,
        tokenizer,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            context_size=config["context_length"],
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )

    return token_ids_to_text(
        output_ids,
        tokenizer,
    )


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    validation_text = VALIDATION_PATH.read_text(
        encoding="utf-8"
    )

    results = {}
    generated = {}

    for label, checkpoint_path in CHECKPOINTS.items():
        print(f"Evaluating {label}: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        model, config = build_model(checkpoint)
        batches = create_validation_batches(
            validation_text,
            tokenizer,
            config["context_length"],
        )

        loss, perplexity = evaluate_model(
            model,
            batches,
        )

        results[label] = {
            "checkpoint": str(checkpoint_path),
            "global_step": checkpoint.get("global_step"),
            "training_loss": checkpoint.get("train_loss"),
            "validation_loss_from_checkpoint": checkpoint.get("val_loss"),
            "comparison_validation_loss": loss,
            "comparison_perplexity": perplexity,
        }

        generated[label] = {
            prompt: generate_sample(
                model,
                tokenizer,
                config,
                prompt,
            )
            for prompt in PROMPTS
        }

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "generations.json").write_text(
        json.dumps(generated, indent=2),
        encoding="utf-8",
    )

    labels = list(results)
    losses = [
        results[label]["comparison_validation_loss"]
        for label in labels
    ]
    perplexities = [
        results[label]["comparison_perplexity"]
        for label in labels
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )
    axes[0].bar(labels, losses)
    axes[0].set_title("Validation loss")
    axes[0].set_ylabel("Cross-entropy")
    axes[1].bar(labels, perplexities)
    axes[1].set_title("Validation perplexity")
    axes[1].set_ylabel("Perplexity")
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "metrics.png",
        dpi=160,
    )
    plt.close(figure)

    with (RESULTS_DIR / "generations.txt").open(
        "w",
        encoding="utf-8",
    ) as output:
        for prompt in PROMPTS:
            output.write(f"Prompt: {prompt}\n")
            for label in labels:
                output.write(
                    f"\n[{label}]\n{generated[label][prompt]}\n"
                )
            output.write("\n" + "=" * 80 + "\n\n")

    print("\nMetrics:")
    for label in labels:
        print(
            f"{label}: "
            f"loss={results[label]['comparison_validation_loss']:.4f}, "
            f"perplexity={results[label]['comparison_perplexity']:.4f}"
        )
    print(f"\nSaved comparison files to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
