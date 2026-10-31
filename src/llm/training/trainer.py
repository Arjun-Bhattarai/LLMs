"""Continued pretraining loop with resumable state and optional DDP."""

import csv
import math
import os
import time

import torch
import torch.distributed as dist

from llm.training.loss import calc_loss_batch, calc_loss_loader
from llm.training.checkpoint import save_checkpoint, load_checkpoint
from llm.decoding.sampling import generate_text_simple, text_to_token_ids, token_ids_to_text


def _rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq,
    eval_iter, start_context, tokenizer, checkpoint_path=None, config=None,
    max_steps=None, resume_from_checkpoint=None, scheduler=None,
    gradient_accumulation_steps=1, grad_clip=1.0, metrics_path=None,
    periodic_checkpoint_steps=5000, best_checkpoint_path=None,
    rank=None, world_size=None, eval_generation=False,
):
    """Train while retaining the historic three-list return value."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    rank = _rank() if rank is None else rank
    distributed = _is_distributed()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step, start_epoch, batch_position = 0, 0, 0, 0
    best_val_loss = float("inf")
    if resume_from_checkpoint:
        checkpoint = load_checkpoint(model, optimizer, scaler, resume_from_checkpoint,
                                     device, scheduler=scheduler)
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"])
        batch_position = int(checkpoint["batch_in_epoch"])
        if not checkpoint.get("rng_state") and batch_position:
            # Legacy files never stored the sampler RNG. Do not pretend their
            # historical random order can be reconstructed.
            if rank == 0:
                print("Legacy checkpoint has no RNG/batch cursor; restarting at the saved epoch boundary.")
            start_epoch += 1
            batch_position = 0
        tokens_seen = int(checkpoint["tokens_seen"])
        best_val_loss = float(checkpoint["best_val_loss"])
        if rank == 0:
            print(f"Resumed step {global_step}, epoch {start_epoch}, batch {batch_position}")
            if not checkpoint.get("scheduler_state_dict"):
                print("Legacy checkpoint has no scheduler state; schedule starts from its restored optimizer LR.")

    base_model = _unwrap(model)
    if rank == 0 and use_amp:
        print("AMP enabled: FP16")
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    window_tokens = 0
    running_loss = 0.0
    started = time.perf_counter()
    if metrics_path and rank == 0 and not os.path.exists(metrics_path):
        os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "epoch", "train_loss", "val_loss", "val_perplexity", "lr", "tokens_seen", "steps_per_second", "gpu_memory_bytes", "grad_norm"])

    for epoch in range(start_epoch, num_epochs):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        elif hasattr(train_loader, "generator") and train_loader.generator is not None:
            train_loader.generator.manual_seed(epoch)
        model.train()
        for batch_index, (input_batch, target_batch) in enumerate(train_loader):
            if epoch == start_epoch and batch_index < batch_position:
                continue
            if max_steps is not None and global_step >= max_steps:
                break
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            window_tokens += input_batch.numel()
            running_loss += float(loss.detach())
            end_window = micro_step >= gradient_accumulation_steps
            last_batch = batch_index + 1 == len(train_loader)
            if not end_window and not last_batch:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) if grad_clip else torch.tensor(float("nan"), device=device)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            tokens_seen += window_tokens
            window_tokens = 0
            current_batch = batch_index + 1
            step_loss = running_loss / micro_step
            running_loss = 0.0
            micro_step = 0
            should_eval = eval_freq > 0 and global_step % eval_freq == 0
            if should_eval:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss); val_losses.append(val_loss); track_tokens_seen.append(tokens_seen)
                if _is_distributed():
                    values = torch.tensor([train_loss, val_loss], device=device, dtype=torch.float64)
                    dist.all_reduce(values, op=dist.ReduceOp.SUM); values /= dist.get_world_size()
                    train_loss, val_loss = values.tolist()
                    train_losses[-1], val_losses[-1] = train_loss, val_loss
                if rank == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    lr = optimizer.param_groups[0]["lr"]
                    mem = torch.cuda.max_memory_allocated(device) if use_amp else 0
                    print(f"Step {global_step:06d}: train {train_loss:.4f}, val {val_loss:.4f}, ppl {math.exp(min(val_loss, 50)):.3f}, lr {lr:.3g}")
                    if metrics_path:
                        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([global_step, epoch, train_loss, val_loss, math.exp(min(val_loss, 50)), lr, tokens_seen, global_step / elapsed, mem, float(grad_norm)])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if best_checkpoint_path:
                            save_checkpoint(model, optimizer, scaler, epoch, global_step, train_loss, val_loss, config, best_checkpoint_path,
                                            scheduler=scheduler, batch_in_epoch=current_batch, tokens_seen=tokens_seen, best_val_loss=best_val_loss)
                if _is_distributed():
                    best_tensor = torch.tensor(best_val_loss if rank == 0 else 0.0, device=device)
                    dist.broadcast(best_tensor, src=0); best_val_loss = float(best_tensor.item())
                model.train()
            due_periodic = periodic_checkpoint_steps and global_step % periodic_checkpoint_steps == 0
            if checkpoint_path and (due_periodic or (max_steps is not None and global_step >= max_steps)) and rank == 0:
                save_checkpoint(model, optimizer, scaler, epoch, global_step, step_loss,
                                val_losses[-1] if val_losses else None, config, checkpoint_path,
                                scheduler=scheduler, batch_in_epoch=current_batch, tokens_seen=tokens_seen,
                                best_val_loss=best_val_loss)
            if max_steps is not None and global_step >= max_steps:
                break
        else:
            batch_position = 0
            continue
        if max_steps is not None and global_step >= max_steps:
            break
    if checkpoint_path and rank == 0 and (not max_steps or global_step % (periodic_checkpoint_steps or 1)):
        save_checkpoint(model, optimizer, scaler, epoch, global_step,
                        train_losses[-1] if train_losses else None,
                        val_losses[-1] if val_losses else None, config, checkpoint_path,
                        scheduler=scheduler, batch_in_epoch=current_batch if 'current_batch' in locals() else batch_position,
                        tokens_seen=tokens_seen, best_val_loss=best_val_loss)
    if eval_generation and rank == 0:
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    if was_training:
        model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = _unwrap(model).pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.inference_mode():
        token_ids = generate_text_simple(model=_unwrap(model), idx=encoded, max_new_tokens=50, context_size=context_size)
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()
