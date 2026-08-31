import torch

from llm.training.loss import calc_loss_batch, calc_loss_loader
from llm.training.checkpoint import save_checkpoint
from llm.decoding.sampling import generate_text_simple
from llm.decoding.sampling import text_to_token_ids, token_ids_to_text


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
):
    """
    Train a GPT-style language model.

    Performs:
    - Forward propagation
    - Loss calculation
    - Backpropagation
    - Optimizer updates
    - Training/validation evaluation
    - Sample text generation
    - Optional checkpoint saving
    """

    train_losses = []
    val_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            # Clear previous gradients
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

            # Update model parameters
            optimizer.step()

            # Track training progress
            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluate model periodically
            if global_step % eval_freq == 0:

                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
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

        # Generate sample text after each epoch
        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context,
        )

        # Save checkpoint after each epoch
        if checkpoint_path is not None:

            # Make sure at least one evaluation has happened
            if train_losses and val_losses:

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
    Evaluate the model on training and validation data.
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

    # Return model to training mode
    model.train()

    return train_loss, val_loss


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
):
    """
    Generate and print a sample of text after an epoch.
    """

    model.eval()

    # Get context length from positional embeddings
    context_size = model.pos_emb.weight.shape[0]

    # Convert starting text to token IDs
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

    # Convert token IDs back to text
    decoded_text = token_ids_to_text(
        token_ids,
        tokenizer,
    )

    print(
        decoded_text.replace("\n", " ")
    )

    # Return to training mode
    model.train()