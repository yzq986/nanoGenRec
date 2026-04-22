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

from ntp.model import NTPModel
from ntp.baseline import NTPProbe
from ntp.train import (
    UnifiedSequenceDataset, unified_collate_fn,
    setup_ddp, cleanup_ddp, log, format_eta, save_checkpoint,
)
from rl.dpo import (
    compute_sid_logprobs_batch, softmax_dpo_loss, _freeze_moe_bias,
)
from rl.preference import load_preference_shard


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

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
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
    pure_dpo=False,
    dpo_epochs=1,
    wandb_run=None,
):
    """Joint NTP + DPO training, or pure DPO (when pure_dpo=True).

    Joint mode: NTP (large batches) and DPO (small batches) each step.
        total_loss = ntp_loss + dpo_weight * dpo_loss
    Pure DPO mode: only DPO loss, steps driven by DPO pair epochs.
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
    if not pure_dpo:
        if dpo_weight > 0:
            dpo_reserve_gb = 3.0
            avail_gb -= dpo_reserve_gb
            log(is_main, f"  DPO active: reserving {dpo_reserve_gb}GB → avail={avail_gb:.1f}GB")
        mem_safe_bs = max(32, int(avail_gb * 1024 ** 3 / bytes_per_sample))
        if batch_size > mem_safe_bs:
            log(is_main, f"  Auto-capping NTP batch_size {batch_size} → {mem_safe_bs} "
                         f"(seq_len={max_seq_len}, avail={avail_gb:.1f}GB)")
            batch_size = mem_safe_bs

    # ── DPO DataLoader ──
    dpo_dataset = PreferencePairDataset(
        preference_pairs, difficulty=difficulty,
        n_rejected=dpo_n_rejected, n_layers=n_layers)
    log(is_main, f"  DPO dataset: {len(dpo_dataset):,} pairs (difficulty={difficulty})")

    if len(dpo_dataset) == 0:
        log(is_main, "  WARNING: No valid DPO pairs! Training NTP-only.")
        dpo_weight = 0.0
        pure_dpo = False

    dpo_loader = DataLoader(
        dpo_dataset,
        batch_size=dpo_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=preference_collate_fn,
    ) if len(dpo_dataset) > 0 else None

    # ── NTP DataLoader (skipped in pure_dpo mode) ──
    ntp_loader = None
    if not pure_dpo:
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

    # ── Compute total steps ──
    if pure_dpo:
        n_dpo_batches = len(dpo_loader) if dpo_loader else 0
        n_batches = n_dpo_batches * dpo_epochs
        if max_steps:
            n_batches = min(n_batches, max_steps)
        log(is_main, f"  Pure DPO mode: {n_dpo_batches} batches/epoch × {dpo_epochs} epochs = {n_batches} steps")
    else:
        n_batches = len(ntp_loader)
        if max_steps:
            n_batches = min(n_batches, max_steps)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches)

    if pure_dpo:
        log(is_main, f"  Training: {n_batches} steps (pure DPO), "
                     f"DPO batch={dpo_batch_size}, n_rej={dpo_n_rejected}, "
                     f"β={dpo_beta}, lr={lr}")
    else:
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

    # ── DPO step helper (shared by both modes) ──
    def _dpo_step(dpo_batch, weight=1.0):
        """Compute DPO loss and backward. Returns (unscaled_loss, diagnostics)."""
        with _freeze_moe_bias(raw_policy):
            ctx_padded_dpo, ctx_lengths_dpo, all_sids, sample_offsets = dpo_batch
            ctx_padded_dpo = ctx_padded_dpo.to(device, non_blocking=True)
            ctx_lengths_dpo = ctx_lengths_dpo.to(device, non_blocking=True)
            all_sids = all_sids.to(device, non_blocking=True)
            sample_offsets = sample_offsets.to(device, non_blocking=True)

            counts = sample_offsets[1:] - sample_offsets[:-1]
            ctx_exp = torch.repeat_interleave(ctx_padded_dpo, counts, dim=0)
            len_exp = torch.repeat_interleave(ctx_lengths_dpo, counts, dim=0)

            with torch.no_grad():
                ref_lp = compute_sid_logprobs_batch(
                    ref_model, ctx_exp, len_exp, all_sids, n_layers)

            policy_lp = compute_sid_logprobs_batch(
                raw_policy, ctx_exp, len_exp, all_sids, n_layers)

            dpo_loss, diag = softmax_dpo_loss(
                policy_lp, ref_lp, sample_offsets, beta=dpo_beta,
                return_diagnostics=True,
            )

            (weight * dpo_loss).backward()
            del ctx_padded_dpo, ctx_lengths_dpo, all_sids, sample_offsets
            del ctx_exp, len_exp, policy_lp, ref_lp
        return dpo_loss, diag

    # ── Gradient all-reduce helper ──
    def _allreduce_grads():
        if world_size <= 1:
            return
        grads = [p.grad for p in policy_model.parameters() if p.grad is not None]
        if grads:
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

    # ── Build step iterator ──
    if pure_dpo:
        def _step_iter():
            """Yield (step, dpo_batch) for dpo_epochs over dpo_loader."""
            step = 0
            for _epoch in range(dpo_epochs):
                for batch in dpo_loader:
                    yield step, batch
                    step += 1
        step_iterator = _step_iter()
    else:
        # Cyclic DPO iterator (for joint NTP+DPO)
        dpo_iter = iter(dpo_loader) if dpo_loader else None

        def _next_dpo_batch():
            nonlocal dpo_iter
            try:
                return next(dpo_iter)
            except StopIteration:
                dpo_iter = iter(dpo_loader)
                return next(dpo_iter)

    # ── Alignment metrics accumulators ──
    total_chosen_reward = 0.0
    total_rejected_reward = 0.0
    total_preference_acc = 0.0
    n_diag_steps = 0

    # ── Main training loop ──
    if pure_dpo:
        # Pure DPO: iterate over DPO pairs only
        for step, dpo_batch in step_iterator:
            if max_steps and step >= max_steps:
                break

            optimizer.zero_grad()
            dpo_loss_val, diag = _dpo_step(dpo_batch)
            _allreduce_grads()

            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0).item()
            optimizer.step()
            scheduler.step()

            step_dpo = dpo_loss_val.item()
            total_dpo_loss += step_dpo
            if diag:
                total_chosen_reward += diag['chosen_reward']
                total_rejected_reward += diag['rejected_reward']
                total_preference_acc += diag['preference_acc']
                n_diag_steps += 1

            if is_main:
                cur_lr = scheduler.get_last_lr()[0]
                log_entry = {
                    'step': step,
                    'ntp_loss': 0.0,
                    'dpo_loss': round(step_dpo, 6),
                    'total_loss': round(step_dpo, 6),
                    'lr': round(cur_lr, 8),
                    'grad_norm': round(grad_norm, 4),
                    'tokens': 0,
                    'wall_s': round(time.time() - t0, 2),
                }
                if diag:
                    log_entry.update({
                        'chosen_reward': round(diag['chosen_reward'], 4),
                        'rejected_reward': round(diag['rejected_reward'], 4),
                        'reward_margin': round(diag['reward_margin'], 4),
                        'preference_acc': round(diag['preference_acc'], 4),
                        'kl': round(diag['kl'], 4),
                    })
                train_log.append(log_entry)
                if wandb_run is not None:
                    wb = {
                        'train/dpo_loss': step_dpo,
                        'train/total_loss': step_dpo,
                        'train/lr': cur_lr,
                        'train/grad_norm': grad_norm,
                    }
                    if diag:
                        wb.update({
                            'train/chosen_reward': diag['chosen_reward'],
                            'train/rejected_reward': diag['rejected_reward'],
                            'train/reward_margin': diag['reward_margin'],
                            'train/preference_acc': diag['preference_acc'],
                            'train/kl': diag['kl'],
                        })
                    wandb_run.log(wb, step=step)

            if is_main and (step + 1) % 10 == 0:
                elapsed = time.time() - t0
                remaining = (n_batches - step - 1) / ((step + 1) / elapsed)
                eta = format_eta(remaining)
                avg_dpo = total_dpo_loss / (step + 1)
                margin_str = f", margin={diag['reward_margin']:.2f}, acc={diag['preference_acc']:.0%}" if diag else ""
                print(f"    step {step+1}/{n_batches}: "
                      f"dpo={avg_dpo:.4f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, "
                      f"gnorm={grad_norm:.2f}{margin_str}, ETA {eta}")

    else:
        # Joint NTP + DPO: iterate over NTP loader, sample DPO each step
        for step, ntp_batch in enumerate(ntp_loader):
            if max_steps and step >= max_steps:
                break

            optimizer.zero_grad()

            # NTP loss (backward immediately to free activations)
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

            ntp_loss.backward()
            del padded, input_tokens, target_tokens, valid_mask, train_mask

            # DPO loss (separate backward, gradients accumulate)
            dpo_loss_val = torch.tensor(0.0, device=device)
            diag = {}
            if dpo_weight > 0 and dpo_loader is not None:
                dpo_batch = _next_dpo_batch()
                dpo_loss_val, diag = _dpo_step(dpo_batch, weight=dpo_weight)

            _allreduce_grads()

            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0).item()
            optimizer.step()
            scheduler.step()

            step_ntp = ntp_loss.item()
            step_dpo = dpo_loss_val.item()
            total_ntp_loss += step_ntp
            total_dpo_loss += step_dpo
            step_tokens = int(lengths.sum().item()) * world_size
            total_tokens += step_tokens
            if diag:
                total_chosen_reward += diag['chosen_reward']
                total_rejected_reward += diag['rejected_reward']
                total_preference_acc += diag['preference_acc']
                n_diag_steps += 1

            if is_main:
                cur_lr = scheduler.get_last_lr()[0]
                step_total = step_ntp + dpo_weight * step_dpo
                log_entry = {
                    'step': step,
                    'ntp_loss': round(step_ntp, 6),
                    'dpo_loss': round(step_dpo, 6),
                    'total_loss': round(step_total, 6),
                    'lr': round(cur_lr, 8),
                    'grad_norm': round(grad_norm, 4),
                    'tokens': total_tokens,
                    'wall_s': round(time.time() - t0, 2),
                }
                if diag:
                    log_entry.update({
                        'chosen_reward': round(diag['chosen_reward'], 4),
                        'rejected_reward': round(diag['rejected_reward'], 4),
                        'reward_margin': round(diag['reward_margin'], 4),
                        'preference_acc': round(diag['preference_acc'], 4),
                        'kl': round(diag['kl'], 4),
                    })
                train_log.append(log_entry)
                if wandb_run is not None:
                    wb = {
                        'train/ntp_loss': step_ntp,
                        'train/dpo_loss': step_dpo,
                        'train/total_loss': step_total,
                        'train/lr': cur_lr,
                        'train/grad_norm': grad_norm,
                        'tokens': total_tokens,
                    }
                    if diag:
                        wb.update({
                            'train/chosen_reward': diag['chosen_reward'],
                            'train/rejected_reward': diag['rejected_reward'],
                            'train/reward_margin': diag['reward_margin'],
                            'train/preference_acc': diag['preference_acc'],
                            'train/kl': diag['kl'],
                        })
                    wandb_run.log(wb, step=step)

            if is_main and (step + 1) % 50 == 0:
                elapsed = time.time() - t0
                toks_per_sec = total_tokens / elapsed
                remaining = (n_batches - step - 1) / ((step + 1) / elapsed)
                eta = format_eta(remaining)
                avg_ntp = total_ntp_loss / (step + 1)
                avg_dpo = total_dpo_loss / (step + 1)
                margin_str = f", margin={diag['reward_margin']:.2f}, acc={diag['preference_acc']:.0%}" if diag else ""
                print(f"    step {step+1}/{n_batches}: "
                      f"ntp={avg_ntp:.4f}, dpo={avg_dpo:.4f}, "
                      f"total={avg_ntp + dpo_weight * avg_dpo:.4f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, "
                      f"gnorm={grad_norm:.2f}{margin_str}, "
                      f"{toks_per_sec:.0f} tok/s, ETA {eta}")

    actual_steps = min(step + 1, n_batches) if 'step' in dir() else 0
    avg_ntp = total_ntp_loss / max(actual_steps, 1)
    avg_dpo = total_dpo_loss / max(actual_steps, 1)
    avg_total = avg_dpo if pure_dpo else (avg_ntp + dpo_weight * avg_dpo)
    elapsed = time.time() - t0

    if pure_dpo:
        log(is_main, f"  Train done: dpo={avg_dpo:.4f}, {actual_steps} steps ({elapsed:.1f}s)")
    else:
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
        'dpo_weight': 0.0 if pure_dpo else dpo_weight,
        'dpo_beta': dpo_beta,
        'difficulty': difficulty,
        'pure_dpo': pure_dpo,
        'dpo_epochs': dpo_epochs if pure_dpo else 0,
        'total_tokens': total_tokens,
        'wall_time_s': round(elapsed, 1),
        'batch_size': batch_size,
        'dpo_batch_size': dpo_batch_size,
        'world_size': world_size,
        'n_steps': actual_steps,
        'n_dpo_pairs': len(dpo_dataset) if dpo_dataset else 0,
    }
    if n_diag_steps > 0:
        train_summary.update({
            'avg_chosen_reward': round(total_chosen_reward / n_diag_steps, 4),
            'avg_rejected_reward': round(total_rejected_reward / n_diag_steps, 4),
            'avg_reward_margin': round((total_chosen_reward - total_rejected_reward) / n_diag_steps, 4),
            'avg_preference_acc': round(total_preference_acc / n_diag_steps, 4),
        })

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
    parser.add_argument('--dpo_epochs', type=int, default=1,
                        help='Number of epochs over DPO pairs (pure_dpo mode, default: 1)')
    parser.add_argument('--pure_dpo', action='store_true',
                        help='Pure DPO mode: no NTP loss, steps driven by DPO pairs')
    parser.add_argument('--difficulty', type=str, default='all',
                        choices=['easy', 'medium', 'hard', 'all'],
                        help='Difficulty filter for preference pairs')
    parser.add_argument('--name', type=str, default='sp-dpo',
                        help='Experiment name for logging')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging (rank 0 only)')
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

    # ── Load NTP shard (this rank) — skipped in pure_dpo mode ──
    if args.pure_dpo:
        tokens_list, split_pos_list = [], []
        log(is_main, f"  Pure DPO mode: skipping NTP shard load")
    else:
        from ntp.preprocess import load_shard
        shard_path = os.path.join(args.preprocessed_dir, f'train_shard_{local_rank}.npz')
        if not os.path.exists(shard_path):
            shard_path = os.path.join(args.preprocessed_dir, 'train_shard_0.npz')
        shard_data = load_shard(shard_path)
        tokens_list, split_pos_list = shard_data['tokens_list'], shard_data['split_pos_list']
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

    ckpt_path = os.path.join(args.output_dir, 'probe.pt')
    train_meta_path = os.path.join(args.output_dir, 'train_meta.json')
    skip_train = os.path.exists(ckpt_path)


    # ── Wandb (rank 0 only) ──
    wandb_run = None
    if is_main and args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project='gr-demo',
                name=args.name,
                config={
                    'dpo_weight': args.dpo_weight,
                    'dpo_beta': args.dpo_beta,
                    'lr': args.lr,
                    'max_steps': args.max_steps,
                    'pure_dpo': args.pure_dpo,
                    'dpo_epochs': args.dpo_epochs,
                    'n_preference_pairs': len(all_pairs),
                },
            )
        except Exception:
            wandb_run = None

    if skip_train:
        log(is_main, f"\n  Checkpoint found at {args.output_dir}, skipping training.")

        # Check which evals are already done
        has_eval = False
        has_align_eval = False
        if os.path.exists(train_meta_path):
            with open(train_meta_path) as f:
                existing_meta = json.load(f)
            has_eval = 'eval' in existing_meta
            has_align_eval = 'alignment_eval' in existing_meta

        if has_eval and has_align_eval:
            log(is_main, f"  All evals already present, nothing to do.")
            if wandb_run is not None:
                wandb_run.finish()
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

        if wandb_run is not None:
            wandb_run.config.update({'difficulty': difficulty})

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
            pure_dpo=args.pure_dpo,
            dpo_epochs=args.dpo_epochs,
            wandb_run=wandb_run,
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

    # ── Check existing evals ──
    existing_meta = {}
    if os.path.exists(train_meta_path):
        with open(train_meta_path) as f:
            existing_meta = json.load(f)

    # ── Inline NTP eval ──
    if 'eval' not in existing_meta:
        log(is_main, "\n  Running inline evaluation...")
        from ntp.train import _run_inline_eval
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

        if is_main and eval_results:
            existing_meta['eval'] = eval_results
            with open(train_meta_path, 'w') as f:
                json.dump(existing_meta, f, indent=2)
            log(is_main, f"  Eval results saved to train_meta.json")

        if wandb_run is not None and eval_results:
            for k, v in eval_results.items():
                if isinstance(v, (int, float)):
                    wandb_run.summary[f'eval/{k}'] = v
    else:
        log(is_main, "\n  NTP eval already present, skipping.")

    # ── Inline alignment eval ──
    if 'alignment_eval' not in existing_meta and len(all_pairs) > 0:
        log(is_main, "\n  Running alignment evaluation...")
        model.to(device)

        ref_model_eval, _ = load_model_from_checkpoint(args.sft_checkpoint, device)
        ref_model_eval.eval()

        difficulty = args.difficulty
        if difficulty == 'all' and os.path.exists(pref_meta_path):
            pref_difficulty = pref_meta.get('difficulty', 'all')
            if pref_difficulty != 'all':
                difficulty = pref_difficulty

        align_dataset = PreferencePairDataset(
            all_pairs, difficulty=difficulty,
            n_rejected=args.dpo_n_rejected, n_layers=n_layers)
        align_loader = DataLoader(
            align_dataset, batch_size=args.dpo_batch_size, shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=preference_collate_fn)

        a_chosen, a_rejected, a_margins = [], [], []
        a_wins, a_total = 0, 0

        with torch.no_grad():
            for bi, batch in enumerate(align_loader):
                ctx_p, ctx_l, sids, offsets = batch
                ctx_p = ctx_p.to(device)
                ctx_l = ctx_l.to(device)
                sids = sids.to(device)
                offsets = offsets.to(device)

                counts = offsets[1:] - offsets[:-1]
                ctx_exp = torch.repeat_interleave(ctx_p, counts, dim=0)
                len_exp = torch.repeat_interleave(ctx_l, counts, dim=0)

                ref_lp = compute_sid_logprobs_batch(
                    ref_model_eval, ctx_exp, len_exp, sids, n_layers)
                policy_lp = compute_sid_logprobs_batch(
                    model, ctx_exp, len_exp, sids, n_layers)

                _, diag = softmax_dpo_loss(
                    policy_lp, ref_lp, offsets,
                    beta=args.dpo_beta, return_diagnostics=True)

                if diag:
                    a_chosen.append(diag['chosen_reward'])
                    a_rejected.append(diag['rejected_reward'])
                    a_margins.append(diag['reward_margin'])
                    bs = int((offsets.size(0) - 1))
                    a_wins += int(diag['preference_acc'] * bs)
                    a_total += bs

                if is_main and (bi + 1) % 50 == 0:
                    log(is_main, f"    alignment batch {bi+1}/{len(align_loader)}, "
                        f"margin={diag['reward_margin']:.3f}, acc={diag['preference_acc']:.1%}")

        del ref_model_eval

        if a_total > 0 and is_main:
            align_results = {
                'chosen_reward': round(float(np.mean(a_chosen)), 4),
                'rejected_reward': round(float(np.mean(a_rejected)), 4),
                'reward_margin': round(float(np.mean(a_margins)), 4),
                'preference_acc': round(a_wins / a_total, 4),
                'n_pairs': a_total,
                'difficulty': difficulty,
                'dpo_beta': args.dpo_beta,
            }
            existing_meta['alignment_eval'] = align_results
            with open(train_meta_path, 'w') as f:
                json.dump(existing_meta, f, indent=2)
            log(is_main, f"  Alignment eval: margin={align_results['reward_margin']:.4f}, "
                         f"acc={align_results['preference_acc']:.2%}")
            log(is_main, f"  Saved to train_meta.json ['alignment_eval']")

            if wandb_run is not None:
                for k, v in align_results.items():
                    if isinstance(v, (int, float)):
                        wandb_run.summary[f'alignment/{k}'] = v
    else:
        if 'alignment_eval' in existing_meta:
            log(is_main, "\n  Alignment eval already present, skipping.")

    if wandb_run is not None:
        wandb_run.finish()

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
    from ntp.train import _run_inline_eval
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


def alignment_eval_main():
    """Evaluate alignment metrics on existing checkpoint + preference pairs.

    Runs forward-only (no training) to compute implicit reward, preference
    accuracy, and reward margin on all preference pairs.

    Usage:
        python run.py alignment-eval \
            --checkpoint experiments/ntp_checkpoints/exp019-joint-hard-lam10 \
            --reference experiments/ntp_checkpoints/exp017-fixed-medium \
            --preference_dir experiments/rf_dpo_data/exp018/hard
    """
    parser = argparse.ArgumentParser(description='Evaluate DPO alignment metrics')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Policy model checkpoint directory')
    parser.add_argument('--reference', type=str, required=True,
                        help='Reference model checkpoint directory (the SFT/SP-DPO base)')
    parser.add_argument('--preference_dir', type=str, required=True,
                        help='Path to preference pair shards')
    parser.add_argument('--difficulty', type=str, default='all',
                        choices=['easy', 'medium', 'hard', 'all'])
    parser.add_argument('--dpo_beta', type=float, default=0.1)
    parser.add_argument('--dpo_batch_size', type=int, default=16)
    parser.add_argument('--dpo_n_rejected', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_main = True

    log(is_main, "=" * 60)
    log(is_main, "Alignment Evaluation (forward-only)")
    log(is_main, f"  Policy:    {args.checkpoint}")
    log(is_main, f"  Reference: {args.reference}")
    log(is_main, f"  Pref data: {args.preference_dir}")
    log(is_main, "=" * 60)

    # Load models
    log(is_main, "  Loading policy model...")
    policy_model, cfg = load_model_from_checkpoint(args.checkpoint, device)
    policy_model.eval()

    log(is_main, "  Loading reference model...")
    ref_model, _ = load_model_from_checkpoint(args.reference, device)
    ref_model.eval()

    # Load NTP meta for n_layers
    meta_path = os.path.join(args.checkpoint, 'train_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    preprocessed_dir = meta.get('preprocessed_dir')
    with open(os.path.join(preprocessed_dir, 'meta.json')) as f:
        prep_meta = json.load(f)
    n_layers = prep_meta['n_layers']

    # Load preference pairs
    pref_meta_path = os.path.join(args.preference_dir, 'meta.json')
    n_pref_shards = 1
    if os.path.exists(pref_meta_path):
        with open(pref_meta_path) as f:
            pref_meta = json.load(f)
        n_pref_shards = pref_meta.get('n_shards', 1)

    all_pairs = []
    for si in range(n_pref_shards):
        shard_file = os.path.join(args.preference_dir, f'preference_shard_{si}.npz')
        if os.path.exists(shard_file):
            pairs = load_preference_shard(shard_file)
            all_pairs.extend(pairs)
    log(is_main, f"  Loaded {len(all_pairs):,} preference pairs")

    # Build dataset + loader
    difficulty = args.difficulty
    if difficulty == 'all' and os.path.exists(pref_meta_path):
        pref_difficulty = pref_meta.get('difficulty', 'all')
        if pref_difficulty != 'all':
            difficulty = pref_difficulty

    dataset = PreferencePairDataset(
        all_pairs, difficulty=difficulty,
        n_rejected=args.dpo_n_rejected, n_layers=n_layers)
    log(is_main, f"  Dataset: {len(dataset):,} pairs (difficulty={difficulty})")

    loader = DataLoader(
        dataset, batch_size=args.dpo_batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=preference_collate_fn)

    # Forward pass over all pairs
    all_chosen = []
    all_rejected = []
    all_margins = []
    wins = 0
    total_pairs = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            ctx_padded, ctx_lengths, all_sids, sample_offsets = batch
            ctx_padded = ctx_padded.to(device)
            ctx_lengths = ctx_lengths.to(device)
            all_sids = all_sids.to(device)
            sample_offsets = sample_offsets.to(device)

            counts = sample_offsets[1:] - sample_offsets[:-1]
            ctx_exp = torch.repeat_interleave(ctx_padded, counts, dim=0)
            len_exp = torch.repeat_interleave(ctx_lengths, counts, dim=0)

            ref_lp = compute_sid_logprobs_batch(
                ref_model, ctx_exp, len_exp, all_sids, n_layers)
            policy_lp = compute_sid_logprobs_batch(
                policy_model, ctx_exp, len_exp, all_sids, n_layers)

            _, diag = softmax_dpo_loss(
                policy_lp, ref_lp, sample_offsets,
                beta=args.dpo_beta, return_diagnostics=True)

            if diag:
                all_chosen.append(diag['chosen_reward'])
                all_rejected.append(diag['rejected_reward'])
                all_margins.append(diag['reward_margin'])
                wins += int(diag['preference_acc'] * args.dpo_batch_size)
                total_pairs += args.dpo_batch_size

            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                log(is_main, f"    batch {batch_idx+1}/{len(loader)}, "
                    f"margin={diag['reward_margin']:.3f}, acc={diag['preference_acc']:.1%} "
                    f"({elapsed:.1f}s)")

    elapsed = time.time() - t0

    if not all_chosen:
        log(is_main, "  No valid pairs evaluated!")
        return

    avg_chosen = np.mean(all_chosen)
    avg_rejected = np.mean(all_rejected)
    avg_margin = np.mean(all_margins)
    pref_acc = wins / total_pairs if total_pairs > 0 else 0

    log(is_main, "")
    log(is_main, "=" * 60)
    log(is_main, "Alignment Evaluation Results")
    log(is_main, "=" * 60)
    log(is_main, f"  Pairs evaluated:   {total_pairs:,}")
    log(is_main, f"  Chosen reward:     {avg_chosen:.4f}")
    log(is_main, f"  Rejected reward:   {avg_rejected:.4f}")
    log(is_main, f"  Reward margin:     {avg_margin:.4f}")
    log(is_main, f"  Preference acc:    {pref_acc:.2%}")
    log(is_main, f"  Wall time:         {elapsed:.1f}s")

    # Save to train_meta.json
    alignment_results = {
        'chosen_reward': round(avg_chosen, 4),
        'rejected_reward': round(avg_rejected, 4),
        'reward_margin': round(avg_margin, 4),
        'preference_acc': round(pref_acc, 4),
        'n_pairs': total_pairs,
        'reference': args.reference,
        'preference_dir': args.preference_dir,
        'difficulty': difficulty,
        'dpo_beta': args.dpo_beta,
    }
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    meta['alignment_eval'] = alignment_results
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log(is_main, f"\n  Saved to {meta_path} ['alignment_eval']")


if __name__ == '__main__':
    main()
