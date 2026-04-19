"""Joint NTP + DPO training loop for SP-DPO alignment.

Alternates between NTP loss (prevents forgetting) and Softmax-DPO loss
(alignment signal from preference pairs). Supports progressive difficulty
stages: Easy → Medium → Hard.

Usage:
    # Single GPU
    python run.py sp-dpo-train \
        --sft_checkpoint experiments/ntp_checkpoints/exp015-scale-04-11M \
        --preference_dir experiments/sp_dpo_data/exp017/easy \
        --preprocessed_dir experiments/ntp_data/exp013 \
        --output_dir experiments/ntp_checkpoints/exp017-spdpo-easy \
        --dpo_weight 0.1 --dpo_beta 0.1 --lr 1e-4

    # Multi-GPU DDP
    torchrun --nproc_per_node=8 run.py sp-dpo-train ...
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from gr_demo.ntp.model import NTPModel
from gr_demo.ntp.baseline import NTPProbe
from gr_demo.ntp.train import (
    UnifiedSequenceDataset, unified_collate_fn,
    setup_ddp, cleanup_ddp, log, format_eta, save_checkpoint,
)
from gr_demo.rl.dpo import (
    compute_sid_logprobs_batch, softmax_dpo_loss, _freeze_moe_bias,
)
from gr_demo.rl.preference import load_preference_shard


# ============================================================
# Preference pair dataset + collate
# ============================================================

class PreferencePairDataset(Dataset):
    """Dataset for DPO preference pairs with difficulty filtering."""

    def __init__(self, pairs, difficulty='all', n_rejected=20, n_layers=3):
        """
        Args:
            pairs: list of dicts from load_preference_shard()
            difficulty: 'easy', 'medium', 'hard', or 'all'
            n_rejected: max rejected per sample
            n_layers: SID layers
        """
        self.n_layers = n_layers
        self.n_rejected = n_rejected
        self.items = []

        for pair in pairs:
            if difficulty == 'easy':
                rejected = pair['rejected_easy']
            elif difficulty == 'medium':
                rejected = pair['rejected_medium']
            elif difficulty == 'hard':
                rejected = pair['rejected_hard']
            else:  # 'all'
                rejected = (pair['rejected_easy'] +
                            pair['rejected_medium'] +
                            pair['rejected_hard'])

            if not rejected:
                continue

            rejected = rejected[:n_rejected]
            self.items.append({
                'context': pair['context'],
                'chosen': pair['chosen'],
                'rejected': rejected,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return (
            torch.tensor(item['context'], dtype=torch.long),
            torch.tensor(item['chosen'], dtype=torch.long),
            [torch.tensor(r, dtype=torch.long) for r in item['rejected']],
        )


def preference_collate_fn(batch):
    """Collate preference pairs as packed flat candidates (no padding).

    Each sample's candidates are packed contiguously: [chosen, rej_0, rej_1, ...]
    with sample_offsets marking boundaries. This avoids wasting compute on
    zero-padded rejected SIDs (significant when rejection counts vary, e.g.
    Hard difficulty averages 5.9/pair but padding would expand to max≈20).

    Returns:
        context_padded: (B, max_ctx_len) — right-padded context tokens
        context_lengths: (B,) — actual context lengths
        all_sids: (N_total, n_layers) — flat packed [chosen_0, rej_0_*, chosen_1, rej_1_*, ...]
        sample_offsets: (B+1,) — sample i's candidates at [off[i]:off[i+1]], first is chosen
    """
    contexts, chosens, rejected_lists = zip(*batch)

    # Pad contexts (same as before — contexts still need alignment)
    ctx_lengths = torch.tensor([len(c) for c in contexts], dtype=torch.long)
    max_ctx = ctx_lengths.max().item()
    ctx_padded = torch.zeros(len(batch), max_ctx, dtype=torch.long)
    for i, c in enumerate(contexts):
        ctx_padded[i, :len(c)] = c

    # Pack all SIDs flat: [chosen_0, rej_0_*, chosen_1, rej_1_*, ...]
    all_sids = []
    sample_offsets = [0]
    for chosen, rejected_list in zip(chosens, rejected_lists):
        all_sids.append(chosen)
        for r in rejected_list:
            all_sids.append(r)
        sample_offsets.append(len(all_sids))

    all_sids = torch.stack(all_sids)  # (N_total, n_layers)
    sample_offsets = torch.tensor(sample_offsets, dtype=torch.long)

    return ctx_padded, ctx_lengths, all_sids, sample_offsets


# ============================================================
# Model loading
# ============================================================

def load_model_from_checkpoint(ckpt_path, device):
    """Load NTPModel/NTPProbe from checkpoint directory.

    Args:
        ckpt_path: directory containing probe.pt
        device: torch device

    Returns:
        (model, config_dict)
    """
    ckpt = torch.load(
        os.path.join(ckpt_path, 'probe.pt'),
        map_location=device, weights_only=False)
    cfg = ckpt['config']
    model_type = cfg.pop('model_type', 'probe')

    if model_type == 's-tier':
        model = NTPModel(**cfg)
    else:
        model = NTPProbe(**cfg)

    model.load_state_dict(ckpt['model_state_dict'])
    cfg['model_type'] = model_type
    return model.to(device), cfg


# ============================================================
# Training loop
# ============================================================

def train_dpo(
    ntp_tokens_list,
    ntp_split_pos_list,
    preference_pairs,
    n_clusters_per_layer,
    n_layers,
    sft_checkpoint,
    local_rank,
    world_size,
    device,
    is_main,
    preprocessed_dir,
    sid_cache_dir,
    difficulty='all',
    dpo_weight=0.1,
    dpo_beta=0.1,
    lr=1e-4,
    batch_size=2048,
    dpo_batch_size=16,
    dpo_n_rejected=20,
    max_steps=None,
    wandb_run=None,
):
    """Joint NTP + DPO training.

    Two DataLoaders alternate: NTP (large batches) and DPO (small batches).
    Each step:
        total_loss = ntp_loss + dpo_weight * dpo_loss
    """
    # ── Load policy model (from SFT checkpoint) ──
    log(is_main, f"  Loading policy model from {sft_checkpoint}...")
    policy_model, cfg = load_model_from_checkpoint(sft_checkpoint, device)
    model_type = cfg.get('model_type', 's-tier')
    n_params = sum(p.numel() for p in policy_model.parameters())
    log(is_main, f"  Policy: {model_type}, {n_params / 1e6:.1f}M params")

    # ── Load reference model (frozen copy) ──
    log(is_main, f"  Loading reference model (frozen)...")
    ref_model, _ = load_model_from_checkpoint(sft_checkpoint, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── No DDP wrapper — manual gradient all-reduce instead ──
    # DDP's automatic sync is incompatible with separate NTP/DPO backward
    # passes through raw model (DPO accesses model internals directly).
    # Manual all-reduce after both backward passes is simpler and correct.
    raw_policy = policy_model

    # ── Auto-cap NTP batch_size based on available GPU memory ──
    #
    # Memory model: total_gpu = static + B × bytes_per_sample
    #
    # 1) Static memory: policy (weights + grads + adam m,v) + ref (weights only, frozen)
    #    45.8M params: policy = 45.8M × (4+4+4+4) = 0.73GB, ref = 45.8M × 4 = 0.18GB
    max_seq_len = max(len(t) for t in ntp_tokens_list) if ntp_tokens_list else 512
    gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    model_mem_gb = n_params * 4 * 4 / (1024 ** 3)
    ref_mem_gb = n_params * 4 / (1024 ** 3)
    avail_gb = gpu_mem_gb * 0.85 - model_mem_gb - ref_mem_gb
    #    A100 40GB: avail = 39.5 × 0.85 - 0.73 - 0.18 = 32.7 GB
    #
    # 2) Per-sample activation memory (all layers' intermediates saved for backward):
    #
    #    Component                              Formula              S=510, H=8, D=256
    #    ─────────────────────────────────────────────────────────────────────────────
    #    Attn weights (pre+post dropout)        H × S² × 8B         8×260K×8  = 16.6MB
    #    Dropout mask                           H × S² × 1B         8×260K×1  =  2.1MB
    #    QKV projections + attn_out + norms     6 × S × D × 4B      6×510×256×4= 3.1MB
    #    FFN intermediate + activation (4×D)    2 × S × 4D × 4B     2×510×1K×4 = 4.2MB
    #    ─────────────────────────────────────────────────────────────────────────────
    #    Per layer subtotal                                          ≈ 26.0 MB
    #    × 6 layers = formula_bytes                                  ≈ 156 MB/sample
    #
    # 3) Safety factor × 1.5 for uncounted overhead:
    #    - Norm layer inputs saved for backward (residual tensors)
    #    - Embedding lookup + position embedding tensors
    #    - PyTorch allocator fragmentation (grows non-linearly with batch size)
    #    formula_bytes × 1.5                                         ≈ 234 MB/sample
    #
    # Empirical validation (A100 40GB, H=8, D=256, L=6):
    #
    #    batch | alloc   | +tried  | OOM point    | per-sample/layer
    #    ──────┼─────────┼─────────┼──────────────┼─────────────────
    #    591   | 35.5 GB | +4.56GB | dropout      |   —
    #    267   | 35.5 GB | +2.06GB | dropout      |   —
    #    224   | 38.3 GB | +1.73GB | softmax      |   —
    #    187   | 38.0 GB | +1.45GB | baddbmm(Q@K) |  ~37 MB
    #    46    |  8.6 GB |   OK    |   —          |  ~28.5 MB
    #
    #    Key finding: per-sample memory is NOT constant — it increases with batch
    #    size due to allocator fragmentation on large tensors (e.g. 187×8×510²×4
    #    = 1.45GB per attention layer allocation). batch=46: 28.5 MB/sample/layer,
    #    batch=187: 37 MB/sample/layer (+30%).
    #
    #    With 1.5× safety: formula 156MB × 1.5 = 234MB → batch ≈ 143
    #    Predicted: 143 × ~34MB/layer × 6 + 0.9 = ~30 GB (leaves ~9.5GB headroom)
    #
    embed_dim = cfg.get('embed_dim', 256)
    n_tf_layers = cfg.get('n_transformer_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    S2 = max_seq_len * max_seq_len
    attn_bytes = n_heads * S2 * 9 * n_tf_layers
    linear_bytes = 6 * max_seq_len * embed_dim * 4 * n_tf_layers
    ffn_bytes = 2 * max_seq_len * embed_dim * 4 * 4 * n_tf_layers
    bytes_per_sample = int((attn_bytes + linear_bytes + ffn_bytes) * 1.5)
    #
    # 4) DPO memory reserve (when dpo_weight > 0):
    #
    #    NTP and DPO run sequentially within each step:
    #      NTP forward → NTP backward (free activations) → DPO forward → DPO backward
    #    NTP activations are freed before DPO starts, so they don't overlap.
    #    However, CUDA's caching allocator retains freed blocks in fragmented chunks.
    #    After NTP backward, ~30 GB of cached blocks exist in varying sizes.
    #    DPO (gradient-checkpointed) then allocates/frees 1 chunk at a time,
    #    further fragmenting the cache. By the time all_reduce needs to allocate
    #    a contiguous flat gradient tensor (~183 MB for 45.8M params), the cache
    #    may not have a contiguous region available → OOM.
    #
    #    Empirical: NTP batch=149 (using ~30 GB) + DPO checkpoint + all_reduce
    #    → OOM at all_reduce (NCCL cuda OOM) on A100 40GB.
    #
    #    Fix: reserve 3 GB from available memory when DPO is active.
    #    This accounts for:
    #      - DPO checkpoint peak: 1 chunk (max_chunk=64) ≈ needs headroom for
    #        allocations that can't reuse fragmented NTP cache blocks
    #      - NCCL internal buffers: ~256 MB (32 MB × 8 connections)
    #      - Flat gradient tensor for all_reduce: ~183 MB
    #      - General fragmentation margin
    #    Result: NTP batch drops from ~149 to ~136, leaving room for DPO.
    #
    #    Validated (8×A100 40GB, 45.8M params, seq_len=510, DPO batch=16, K=21):
    #      GPU util: 79-84% (stable, no wave peaks/valleys)
    #      GPU mem:  37.4-38.9 GB (91-95%), rank variance ~1.4 GB (normal)
    #      No OOM across 1420 steps.
    #
    if dpo_weight > 0:
        dpo_reserve_gb = 3.0
        avail_gb -= dpo_reserve_gb
        log(is_main, f"  DPO active: reserving {dpo_reserve_gb}GB → avail={avail_gb:.1f}GB")
    mem_safe_bs = max(32, int(avail_gb * 1024 ** 3 / bytes_per_sample))
    if batch_size > mem_safe_bs:
        log(is_main, f"  Auto-capping NTP batch_size {batch_size} → {mem_safe_bs} "
                     f"(seq_len={max_seq_len}, avail={avail_gb:.1f}GB)")
        batch_size = mem_safe_bs

    # ── NTP DataLoader ──
    ntp_dataset = UnifiedSequenceDataset(ntp_tokens_list, ntp_split_pos_list)
    ntp_loader = DataLoader(
        ntp_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=unified_collate_fn,
    )

    # ── DPO DataLoader (cyclic) ──
    dpo_dataset = PreferencePairDataset(
        preference_pairs, difficulty=difficulty,
        n_rejected=dpo_n_rejected, n_layers=n_layers)
    log(is_main, f"  DPO dataset: {len(dpo_dataset):,} pairs (difficulty={difficulty})")

    if len(dpo_dataset) == 0:
        log(is_main, "  WARNING: No valid DPO pairs! Training NTP-only.")
        dpo_weight = 0.0

    dpo_loader = DataLoader(
        dpo_dataset,
        batch_size=dpo_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=preference_collate_fn,
    ) if len(dpo_dataset) > 0 else None

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=0.01)
    n_batches = len(ntp_loader)
    if max_steps:
        n_batches = min(n_batches, max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches)

    log(is_main, f"  Training: {n_batches} steps, NTP batch={batch_size}, "
                 f"DPO batch={dpo_batch_size}, n_rej={dpo_n_rejected}, "
                 f"λ={dpo_weight}, β={dpo_beta}, lr={lr}")

    # ── Training loop ──
    policy_model.train()
    total_ntp_loss = 0.0
    total_dpo_loss = 0.0
    total_tokens = 0
    train_log = []
    t0 = time.time()

    # ── Pre-allocate flat gradient buffer for all-reduce ──
    #
    # Why pre-allocate: at the point of all_reduce, CUDA memory is heavily
    # fragmented from NTP forward/backward + DPO checkpoint forward/backward.
    # The caching allocator may hold ~30 GB of freed blocks in varying sizes,
    # but cannot assemble a contiguous ~183 MB region for the flat tensor.
    # By allocating once before training (when memory is clean), we guarantee
    # the buffer exists and avoid fragmentation-induced OOM at all_reduce.
    #
    if world_size > 1:
        total_params = sum(p.numel() for p in policy_model.parameters())
        grad_flat_buffer = torch.zeros(total_params, device=device)
        log(is_main, f"  Pre-allocated grad all-reduce buffer: "
                     f"{total_params * 4 / 1024 ** 2:.0f} MB")
    else:
        grad_flat_buffer = None

    # Cyclic DPO iterator
    dpo_iter = iter(dpo_loader) if dpo_loader else None

    def _next_dpo_batch():
        nonlocal dpo_iter
        try:
            return next(dpo_iter)
        except StopIteration:
            dpo_iter = iter(dpo_loader)
            return next(dpo_iter)

    for step, ntp_batch in enumerate(ntp_loader):
        if max_steps and step >= max_steps:
            break

        optimizer.zero_grad()

        # ── NTP loss (backward immediately to free activations) ──
        padded, lengths, split_positions = ntp_batch
        padded = padded.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        split_positions = split_positions.to(device, non_blocking=True)
        B_ntp, T = padded.shape

        input_tokens = padded[:, :-1]
        target_tokens = padded[:, 1:]

        arange = torch.arange(T - 1, device=device).unsqueeze(0)
        valid_mask = arange < (lengths.unsqueeze(1) - 1)
        train_mask = valid_mask & (arange < (split_positions.unsqueeze(1) - 1))

        ntp_loss = policy_model(
            input_tokens,
            packed_targets=target_tokens,
            packed_mask=train_mask,
        )

        # ── DPO loss (separate backward, gradients accumulate) ──
        #
        # Memory timeline per training step:
        #   1. NTP forward: batch≈136 seqs → activations in GPU memory
        #   2. NTP backward: compute gradients, FREE all NTP activations
        #   3. DPO forward: dpo_batch=16 pairs, packed flat (only valid candidates,
        #      no padding waste). E.g. Hard avg 5.9 rej/pair → ~112 forwards
        #      instead of 336 if padded to max_rej=20.
        #      Uses gradient checkpointing (see rl/dpo.py) so only 1 chunk
        #      (max_chunk=64) of activations exists at any time.
        #   4. DPO backward: recompute + backprop chunk by chunk
        #   5. Optimizer step: update params
        #
        dpo_loss_val = torch.tensor(0.0, device=device)
        ntp_loss.backward()
        del padded, input_tokens, target_tokens, valid_mask, train_mask

        if dpo_weight > 0 and dpo_loader is not None:
          # Freeze MoE expert_bias for the ENTIRE DPO section (forward + backward).
          # Gradient checkpointing recomputes forward during backward(); if
          # expert_bias changed between the original forward and the recompute,
          # MoE router decisions differ → intermediate tensor shapes differ → crash.
          # The freeze must cover backward() too, not just the forward call.
          with _freeze_moe_bias(raw_policy):
            dpo_batch = _next_dpo_batch()
            ctx_padded_dpo, ctx_lengths_dpo, all_sids, sample_offsets = dpo_batch
            ctx_padded_dpo = ctx_padded_dpo.to(device, non_blocking=True)
            ctx_lengths_dpo = ctx_lengths_dpo.to(device, non_blocking=True)
            all_sids = all_sids.to(device, non_blocking=True)
            sample_offsets = sample_offsets.to(device, non_blocking=True)

            # Expand contexts to match flat packed candidates
            counts = sample_offsets[1:] - sample_offsets[:-1]  # (B,)
            ctx_exp = torch.repeat_interleave(ctx_padded_dpo, counts, dim=0)
            len_exp = torch.repeat_interleave(ctx_lengths_dpo, counts, dim=0)

            # Reference model log-probs (no grad)
            with torch.no_grad():
                ref_lp = compute_sid_logprobs_batch(
                    ref_model, ctx_exp, len_exp, all_sids, n_layers)

            # Policy model log-probs (with grad, gradient-checkpointed)
            policy_lp = compute_sid_logprobs_batch(
                raw_policy, ctx_exp, len_exp, all_sids, n_layers)

            dpo_loss_val = softmax_dpo_loss(
                policy_lp, ref_lp, sample_offsets, beta=dpo_beta,
            )

            (dpo_weight * dpo_loss_val).backward()
            del ctx_padded_dpo, ctx_lengths_dpo, all_sids, sample_offsets
            del ctx_exp, len_exp, policy_lp, ref_lp

        # ── Bucketed gradient all-reduce (uses pre-allocated buffer) ──
        if world_size > 1:
            grads = [p.grad for p in policy_model.parameters()
                     if p.grad is not None]
            if grads:
                # Copy gradients into pre-allocated flat buffer instead of
                # torch.cat (which allocates new memory at a fragmented time).
                offset = 0
                for g in grads:
                    numel = g.numel()
                    grad_flat_buffer[offset:offset + numel].copy_(g.reshape(-1))
                    offset += numel
                dist.all_reduce(grad_flat_buffer[:offset], op=dist.ReduceOp.AVG)
                offset = 0
                for g in grads:
                    numel = g.numel()
                    g.copy_(grad_flat_buffer[offset:offset + numel].reshape(g.shape))
                    offset += numel

        # ── Step (gradients from both NTP and DPO are accumulated) ──
        grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()

        step_ntp = ntp_loss.item()
        step_dpo = dpo_loss_val.item()
        total_ntp_loss += step_ntp
        total_dpo_loss += step_dpo
        step_tokens = int(lengths.sum().item()) * world_size
        total_tokens += step_tokens

        if is_main:
            cur_lr = scheduler.get_last_lr()[0]
            train_log.append({
                'step': step,
                'ntp_loss': round(step_ntp, 6),
                'dpo_loss': round(step_dpo, 6),
                'total_loss': round(step_ntp + dpo_weight * step_dpo, 6),
                'lr': round(cur_lr, 8),
                'grad_norm': round(grad_norm, 4),
                'tokens': total_tokens,
                'wall_s': round(time.time() - t0, 2),
            })

        if is_main and (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            toks_per_sec = total_tokens / elapsed
            remaining = (n_batches - step - 1) / ((step + 1) / elapsed)
            eta = format_eta(remaining)
            avg_ntp = total_ntp_loss / (step + 1)
            avg_dpo = total_dpo_loss / (step + 1)
            print(f"    step {step+1}/{n_batches}: "
                  f"ntp={avg_ntp:.4f}, dpo={avg_dpo:.4f}, "
                  f"total={avg_ntp + dpo_weight * avg_dpo:.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.2e}, "
                  f"gnorm={grad_norm:.2f}, "
                  f"{toks_per_sec:.0f} tok/s, ETA {eta}")

    actual_steps = min(step + 1, n_batches) if 'step' in dir() else 0
    avg_ntp = total_ntp_loss / max(actual_steps, 1)
    avg_dpo = total_dpo_loss / max(actual_steps, 1)
    avg_total = avg_ntp + dpo_weight * avg_dpo
    elapsed = time.time() - t0

    log(is_main, f"  Train done: ntp={avg_ntp:.4f}, dpo={avg_dpo:.4f}, "
                 f"total={avg_total:.4f}, {total_tokens:,} tokens ({elapsed:.1f}s)")

    # Move to CPU for saving
    policy_model.cpu()
    ref_model.cpu()

    train_summary = {
        'n_params': n_params,
        'avg_ntp_loss': round(avg_ntp, 6),
        'avg_dpo_loss': round(avg_dpo, 6),
        'avg_total_loss': round(avg_total, 6),
        'dpo_weight': dpo_weight,
        'dpo_beta': dpo_beta,
        'difficulty': difficulty,
        'total_tokens': total_tokens,
        'wall_time_s': round(elapsed, 1),
        'batch_size': batch_size,
        'dpo_batch_size': dpo_batch_size,
        'world_size': world_size,
        'n_steps': actual_steps,
        'n_dpo_pairs': len(dpo_dataset) if dpo_dataset else 0,
    }

    return policy_model, avg_total, n_params, train_log, train_summary


# ============================================================
# CLI entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='SP-DPO Joint NTP+DPO Training')
    parser.add_argument('--sft_checkpoint', type=str, required=True,
                        help='Path to SFT model checkpoint directory')
    parser.add_argument('--preference_dir', type=str, required=True,
                        help='Path to preference pair shards')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                        help='Path to preprocessed NTP data shards')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output checkpoint directory')
    parser.add_argument('--dpo_weight', type=float, default=0.1,
                        help='λ weight for DPO loss (default: 0.1)')
    parser.add_argument('--dpo_beta', type=float, default=0.1,
                        help='β temperature for DPO (default: 0.1)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='NTP batch size (default: 2048)')
    parser.add_argument('--dpo_batch_size', type=int, default=16,
                        help='DPO batch size (default: 16)')
    parser.add_argument('--dpo_n_rejected', type=int, default=20,
                        help='Max rejected candidates per DPO pair (default: 20)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Max training steps (default: full epoch)')
    parser.add_argument('--difficulty', type=str, default='all',
                        choices=['easy', 'medium', 'hard', 'all'],
                        help='Difficulty filter for preference pairs')
    parser.add_argument('--name', type=str, default='sp-dpo',
                        help='Experiment name for logging')
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank, world_size, device, is_main = setup_ddp()

    log(is_main, "=" * 60)
    label = "RF-DPO" if "rf" in args.name.lower() else "SP-DPO"
    log(is_main, f"{label} Training — {args.name}" +
                 (f" (DDP x{world_size})" if world_size > 1 else ""))
    log(is_main, "=" * 60)

    # ── Load NTP data meta ──
    meta_path = os.path.join(args.preprocessed_dir, 'meta.json')
    with open(meta_path) as f:
        prep_meta = json.load(f)

    n_layers = prep_meta['n_layers']
    n_clusters_per_layer = prep_meta['n_clusters_per_layer']
    sid_cache_dir = prep_meta['sid_cache']

    # ── Load NTP shard (this rank) ──
    from gr_demo.ntp.preprocess import load_shard
    shard_path = os.path.join(args.preprocessed_dir, f'train_shard_{local_rank}.npz')
    if not os.path.exists(shard_path):
        shard_path = os.path.join(args.preprocessed_dir, 'train_shard_0.npz')
    shard_data = load_shard(shard_path)
    tokens_list, split_pos_list = shard_data[0], shard_data[1]
    log(is_main, f"  NTP shard: {len(tokens_list):,} seqs (rank {local_rank})")

    # ── Load preference pairs (all shards on each rank for simplicity) ──
    pref_meta_path = os.path.join(args.preference_dir, 'meta.json')
    if os.path.exists(pref_meta_path):
        with open(pref_meta_path) as f:
            pref_meta = json.load(f)
        n_pref_shards = pref_meta.get('n_shards', 1)
    else:
        n_pref_shards = 1

    all_pairs = []
    for si in range(n_pref_shards):
        shard_file = os.path.join(args.preference_dir, f'preference_shard_{si}.npz')
        if os.path.exists(shard_file):
            pairs = load_preference_shard(shard_file)
            all_pairs.extend(pairs)
    log(is_main, f"  Preference pairs: {len(all_pairs):,} total from {n_pref_shards} shards")

    # ── Check if checkpoint already exists ──
    ckpt_path = os.path.join(args.output_dir, 'probe.pt')
    train_meta_path = os.path.join(args.output_dir, 'train_meta.json')
    skip_train = os.path.exists(ckpt_path)

    if skip_train:
        log(is_main, f"\n  Checkpoint found at {args.output_dir}, skipping training.")

        # Check if eval already done
        has_eval = False
        if os.path.exists(train_meta_path):
            with open(train_meta_path) as f:
                existing_meta = json.load(f)
            has_eval = 'eval' in existing_meta

        if has_eval:
            log(is_main, f"  Eval results already present, nothing to do.")
            cleanup_ddp()
            return

        # Load existing checkpoint for eval
        log(is_main, f"  Eval missing — loading checkpoint for eval...")
        model, _ = load_model_from_checkpoint(args.output_dir, device)
        model.eval()
    else:
        # ── Determine difficulty from args or preference meta ──
        difficulty = args.difficulty
        pref_difficulty = pref_meta.get('difficulty', 'all') if os.path.exists(pref_meta_path) else 'all'
        if difficulty == 'all' and pref_difficulty != 'all':
            difficulty = pref_difficulty
            log(is_main, f"  Using difficulty from preference meta: {difficulty}")

        # ── Train ──
        log(is_main, f"\n  Training {label} (difficulty={difficulty})...")
        model, avg_loss, n_params, train_log_data, train_summary = train_dpo(
            ntp_tokens_list=tokens_list,
            ntp_split_pos_list=split_pos_list,
            preference_pairs=all_pairs,
            n_clusters_per_layer=n_clusters_per_layer,
            n_layers=n_layers,
            sft_checkpoint=args.sft_checkpoint,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            is_main=is_main,
            preprocessed_dir=args.preprocessed_dir,
            sid_cache_dir=sid_cache_dir,
            difficulty=difficulty,
            dpo_weight=args.dpo_weight,
            dpo_beta=args.dpo_beta,
            lr=args.lr,
            batch_size=args.batch_size,
            dpo_batch_size=args.dpo_batch_size,
            dpo_n_rejected=args.dpo_n_rejected,
            max_steps=args.max_steps,
        )

        # ── Save checkpoint (rank 0 only) ──
        if is_main:
            log(is_main, f"\n  Saving checkpoint to {args.output_dir}")
            save_checkpoint(
                output_dir=args.output_dir,
                probe=model,
                n_clusters_per_layer=n_clusters_per_layer,
                n_layers=n_layers,
                n_items=prep_meta['n_items'],
                avg_loss=avg_loss,
                n_params=n_params,
                sid_cache_dir=sid_cache_dir,
                preprocessed_dir=args.preprocessed_dir,
                model_type=train_summary.get('model_type', 's-tier'),
                n_train=prep_meta['n_seqs'],
                n_eval=prep_meta['n_eval_items'],
                train_log=train_log_data,
                train_summary=train_summary,
            )

    # ── Inline eval ──
    log(is_main, "\n  Running inline evaluation...")
    from gr_demo.ntp.train import _run_inline_eval
    model.to(device)
    eval_results = _run_inline_eval(
        probe=model,
        sid_cache_dir=sid_cache_dir,
        preprocessed_dir=args.preprocessed_dir,
        n_layers=n_layers,
        n_clusters_per_layer=n_clusters_per_layer,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=is_main,
    )

    # ── Save eval results to train_meta ──
    if is_main and eval_results:
        if os.path.exists(train_meta_path):
            with open(train_meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        meta['eval'] = eval_results
        with open(train_meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        log(is_main, f"  Eval results saved to train_meta.json")

    cleanup_ddp()
    log(is_main, f"\n{label} training complete!")


def eval_main():
    """Standalone eval for an existing checkpoint (no training)."""
    parser = argparse.ArgumentParser(description='Eval NTP/DPO checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory (containing probe.pt)')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                        help='Path to preprocessed NTP data (default: from train_meta)')
    parser.add_argument('--sid_cache', type=str, default=None,
                        help='Path to SID cache (default: from train_meta)')
    parser.add_argument('--n_recall', type=int, default=1000,
                        help='Total beam search recall samples (default: 1000)')
    args = parser.parse_args()

    local_rank, world_size, device, is_main = setup_ddp()

    log(is_main, "=" * 60)
    log(is_main, f"Eval checkpoint: {args.checkpoint}" +
                 (f" (DDP x{world_size})" if world_size > 1 else ""))
    log(is_main, "=" * 60)

    # Load checkpoint
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()

    # Resolve preprocessed_dir and sid_cache from train_meta if not given
    meta_path = os.path.join(args.checkpoint, 'train_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    preprocessed_dir = args.preprocessed_dir or meta.get('preprocessed_dir')
    sid_cache_dir = args.sid_cache or meta.get('sid_cache')
    if not preprocessed_dir:
        raise ValueError("--preprocessed_dir required (not found in train_meta)")
    if not sid_cache_dir:
        raise ValueError("--sid_cache required (not found in train_meta)")

    # Load NTP data meta for n_layers / n_clusters
    ntp_meta_path = os.path.join(preprocessed_dir, 'meta.json')
    with open(ntp_meta_path) as f:
        ntp_meta = json.load(f)
    n_layers = ntp_meta['n_layers']
    n_clusters_per_layer = ntp_meta['n_clusters_per_layer']

    log(is_main, f"  preprocessed_dir: {preprocessed_dir}")
    log(is_main, f"  sid_cache:        {sid_cache_dir}")
    log(is_main, f"  n_recall:         {args.n_recall}")

    # Run eval
    from gr_demo.ntp.train import _run_inline_eval
    eval_results = _run_inline_eval(
        probe=model,
        sid_cache_dir=sid_cache_dir,
        preprocessed_dir=preprocessed_dir,
        n_layers=n_layers,
        n_clusters_per_layer=n_clusters_per_layer,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=is_main,
        n_recall_total=args.n_recall,
    )

    # Save eval results to train_meta
    if is_main and eval_results:
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        meta['eval'] = eval_results
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        log(is_main, f"\n  Eval results saved to {meta_path}")

    cleanup_ddp()
    log(is_main, "Eval complete!")


if __name__ == '__main__':
    main()
