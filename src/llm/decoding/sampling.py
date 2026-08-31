import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(
        text,
        allowed_special={"<|endoftext|>"},
    )

    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())


def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size,
):
    """
    Generate text using greedy decoding.
    """

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        idx_next = torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

        idx = torch.cat(
            (idx, idx_next),
            dim=1,
        )

    return idx


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    """
    Generate text using greedy decoding or temperature/top-k sampling.
    """

    for _ in range(max_new_tokens):

        # Keep only supported context
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Last token logits
        logits = logits[:, -1, :]

        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)

            min_val = top_logits[:, -1]

            logits = torch.where(
                logits < min_val,
                torch.tensor(
                    float("-inf"),
                    device=logits.device,
                ),
                logits,
            )

        # Temperature sampling
        if temperature > 0.0:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )

        # Greedy decoding
        else:
            idx_next = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )

        # EOS stopping
        if eos_id is not None and (idx_next == eos_id).any():
            break

        # Append token
        idx = torch.cat(
            (idx, idx_next),
            dim=1,
        )

    return idx