# Kaggle WikiText-103 continuation

The training entry point is `scripts/train.py`. It defaults to one GPU, a
per-device batch of 1, accumulation of 1, FP16 AMP, AdamW `(0.9, 0.95)`,
gradient clipping at 1.0, evaluation every 500 optimizer steps, and a 300,000
absolute-step target. Checkpoints are atomic. `latest_checkpoint.pt` is the
resumption point; `best_model.pt` is selected by validation loss; periodic
latest saves happen every 5,000 steps. Metrics append to
`results/training_metrics.csv`.

## Kaggle setup and resume

Copy the repository and the desired input checkpoint into writable
`/kaggle/working` (Kaggle input mounts are read-only). Preserve the current
100K artifact as an input and set the input path explicitly:

```python
%cd /kaggle/working/LLMs
!pip install -q -e . tiktoken pandas
import os
os.environ["CHECKPOINT_PATH"] = "/kaggle/input/<dataset>/gpt2_wikitext_100k.pt"
os.environ["MAX_STEPS"] = "300000"
os.environ["BATCH_SIZE"] = "1"
os.environ["GRAD_ACCUM_STEPS"] = "1"
os.environ["RESUME_TRAINING"] = "1"
!python scripts/train.py
```

First verify that the checkpoint metadata reports `global_step: 99999` (the
legacy code increments from zero). A checkpoint at a higher verified step can
also be resumed: the script reads the step from its metadata and continues to
the absolute 300,000 target. Do not choose a checkpoint by filename alone: in
this repository, `gpt2_wikitext_300k.pt` is actually at step 216,293. The
optimizer state must accompany the model; do not initialize a fresh optimizer.

The legacy checkpoint has no scheduler/RNG/batch-position fields. Model,
optimizer, and GradScaler restore normally. Its old shuffled batch position
and RNG stream cannot be reconstructed, so the first migration resumes at the
stored epoch boundary and may replay some examples from that epoch. New
checkpoints persist epoch batch offset, token count, scheduler, and RNG state.
For exact future resume, use `latest_checkpoint.pt` created by this version.

## Two T4 DDP launch

Run from a writable copy containing `src/`, dataset files, and checkpoint:

```python
import os
os.environ["CHECKPOINT_PATH"] = "/kaggle/input/<dataset>/gpt2_wikitext_100k.pt"
os.environ["MAX_STEPS"] = "300000"
os.environ["BATCH_SIZE"] = "1"          # per GPU
os.environ["GRAD_ACCUM_STEPS"] = "1"
!torchrun --standalone --nproc_per_node=2 scripts/train.py
```

DDP has effective batch `BATCH_SIZE × GRAD_ACCUM_STEPS × WORLD_SIZE`; the
default above therefore doubles the single-GPU effective batch from 1 to 2.
To preserve effective batch 1, use per-GPU batch 1 with accumulation 1 on one
GPU, and DDP with per-GPU batch 1 plus accumulation cannot go below 2; DDP
changes the experiment's effective batch. Benchmark separately before adopting
it. The two GPUs have separate 16 GB memory and do not combine into 32 GB.

## Measurement and data size

The bundled `train.txt` is 545,120,690 UTF-8 bytes and contains 119,085,168
GPT-2 BPE tokens; validation is 1,156,549 bytes. The count was measured with
`tiktoken` on the training split independently.
With stride 1,024, each sequence predicts 1,024 next-token targets; the final
incomplete region is dropped for training. At batch 1, accumulation 1, a
300,000-step run sees approximately 307.2 million target tokens at effective
batch 1, about 2.58 passes over the training tokens. At effective batch 2,
that is about 614.4 million tokens or 5.16 passes. The 500-step validation
interval is appropriate for monitoring, with `EVAL_ITER=5` a quick signal, not
a full WikiText-103 validation score. Increase evaluation batches for milestone
comparisons and use `best_model.pt` for final evaluation/generation.

Each T4 has 16 GB physical VRAM (roughly 14.5 GB commonly available to the
process). GPT-2 124M with 1,024 context and batch 1 is expected to fit FP16
AMP, but exact headroom depends on CUDA/cuDNN workspace and DataLoader host
memory. Gradient accumulation increases effective batch without increasing
activation memory per microbatch.

DDP initializes NCCL and uses a distributed train sampler. Current evaluation
uses each rank's full validation loader before averaging rank losses; use
identical validation data and limits on every rank. Single-rank writes keep
checkpoint files from colliding. Generation runs on rank zero only.
