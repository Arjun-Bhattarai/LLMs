import torch

from llm.training.loss import calc_loss_batch, calc_loss_loader
from llm.training.checkpoint import save_checkpoint
from llm.decoding.sampling import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    checkpoint_path=None,
    config=None,
    max_steps=None,
):
    """
    Train a GPT-style language model.

    Supports:
    - Training
    - Validation
    - Periodic evaluation
    - Sample generation
    - Checkpoint saving
    - Maximum training steps
    """

    train_losses = []
    val_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            # Stop if maximum steps reached
            if (
                max_steps is not None
                and global_step >= max_steps - 1
            ):
                break

            optimizer.zero_grad()

            # Calculate loss
            loss = calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device,
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Track progress
            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluation
            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    eval_iter=eval_iter,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch + 1} "
                    f"(Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        # Generate sample after epoch
        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
        )

        # Save checkpoint
        if (
            checkpoint_path is not None
            and train_losses
            and val_losses
        ):

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                train_loss=train_losses[-1],
                val_loss=val_losses[-1],
                config=config,
                path=checkpoint_path,
            )

            print(
                f"Checkpoint saved: {checkpoint_path}"
            )

        # Stop after max steps
        if (
            max_steps is not None
            and global_step >= max_steps - 1
        ):
            print(
                f"\nReached max_steps={max_steps}."
            )
            break

    return (
        train_losses,
        val_losses,
        track_tokens_seen,
    )


def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter,
):
    """
    Evaluate model on training and validation data.
    """

    model.eval()

    with torch.no_grad():

        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            num_batches=eval_iter,
        )

        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            num_batches=eval_iter,
        )

    model.train()

    return train_loss, val_loss


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
):
    """
    Generate sample text after training epoch.
    """

    model.eval()

    context_size = model.pos_emb.weight.shape[0]

    encoded = text_to_token_ids(
        start_context,
        tokenizer,
    ).to(device)

    with torch.no_grad():

        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )

    decoded_text = token_ids_to_text(
        token_ids,
        tokenizer,
    )

    print(
        decoded_text.replace("\n", " ")
    )

    model.train()