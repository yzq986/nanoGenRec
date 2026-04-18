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
    compute_sid_logprobs_batch, softmax_dpo_loss,
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
    """Collate preference pairs with right-padded contexts and rejected mask.

    Returns:
        context_padded: (B, max_ctx_len) — right-padded context tokens
        context_lengths: (B,) — actual context lengths
        chosen_sids: (B, n_layers) — ground truth SIDs
        rejected_sids: (B, max_n_rej, n_layers) — rejected SIDs, 0-padded
        rejected_mask: (B, max_n_rej) — True for valid rejected entries
    """
    contexts, chosens, rejected_lists = zip(*batch)

    # Pad contexts
    ctx_lengths = torch.tensor([len(c) for c in contexts], dtype=torch.long)
    max_ctx = ctx_lengths.max().item()
    ctx_padded = torch.zeros(len(batch), max_ctx, dtype=torch.long)
    for i, c in enumerate(contexts):
        ctx_padded[i, :len(c)] = c

    # Stack chosen
    chosen_sids = torch.stack(chosens)  # (B, n_layers)

    # Pad rejected
    max_rej = max(len(rl) for rl in rejected_lists)
    n_layers = chosen_sids.size(1)
    rej_padded = torch.zeros(len(batch), max_rej, n_layers, dtype=torch.long)
    rej_mask = torch.zeros(len(batch), max_rej, dtype=torch.bool)
    for i, rl in enumerate(rejected_lists):
        for j, r in enumerate(rl):
            rej_padded[i, j] = r
            rej_mask[i, j] = True

    return ctx_padded, ctx_lengths, chosen_sids, rej_padded, rej_mask


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
    dpo_batch_size=32,
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

    # ── DDP wrap policy only (ref is inference-only, no gradient sync needed) ──
    if world_size > 1:
        ddp_kwargs = {}
        if model_type == 's-tier' and hasattr(policy_model, 'layers'):
            ffn0 = policy_model.layers[0].ffn
            if hasattr(ffn0, 'n_experts') and ffn0.n_experts >= 2:
                ddp_kwargs['find_unused_parameters'] = True
                import warnings
                warnings.filterwarnings('ignore', message='.*find_unused_parameters.*')
        policy_model = DDP(policy_model, device_ids=[local_rank], **ddp_kwargs)

    raw_policy = policy_model.module if isinstance(policy_model, DDP) else policy_model

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
        n_rejected=20, n_layers=n_layers)
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
                 f"DPO batch={dpo_batch_size}, λ={dpo_weight}, β={dpo_beta}, lr={lr}")

    # ── Training loop ──
    policy_model.train()
    total_ntp_loss = 0.0
    total_dpo_loss = 0.0
    total_tokens = 0
    train_log = []
    t0 = time.time()

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
        ntp_loss.backward()  # free NTP activations before DPO forward
        del padded, input_tokens, target_tokens, valid_mask, train_mask

        # ── DPO loss (separate backward, gradients accumulate) ──
        dpo_loss_val = torch.tensor(0.0, device=device)
        if dpo_weight > 0 and dpo_loader is not None:
            dpo_batch = _next_dpo_batch()
            ctx_padded, ctx_lengths, chosen_sids, rej_sids, rej_mask = dpo_batch
            ctx_padded = ctx_padded.to(device, non_blocking=True)
            ctx_lengths = ctx_lengths.to(device, non_blocking=True)
            chosen_sids = chosen_sids.to(device, non_blocking=True)
            rej_sids = rej_sids.to(device, non_blocking=True)
            rej_mask = rej_mask.to(device, non_blocking=True)

            B_dpo = ctx_padded.size(0)
            N_rej = rej_sids.size(1)

            # All SIDs: chosen (col 0) + rejected (cols 1..N_rej)
            # Shape: (B, 1+N_rej, n_layers)
            all_sids = torch.cat([chosen_sids.unsqueeze(1), rej_sids], dim=1)

            # Reference model log-probs (no grad)
            with torch.no_grad():
                ref_lp = compute_sid_logprobs_batch(
                    ref_model, ctx_padded, ctx_lengths, all_sids, n_layers)
            ref_chosen_lp = ref_lp[:, 0]       # (B,)
            ref_rejected_lp = ref_lp[:, 1:]    # (B, N_rej)

            # Policy model log-probs (with grad, micro-batched)
            policy_lp = compute_sid_logprobs_batch(
                raw_policy, ctx_padded, ctx_lengths, all_sids, n_layers)
            policy_chosen_lp = policy_lp[:, 0]
            policy_rejected_lp = policy_lp[:, 1:]

            dpo_loss_val = softmax_dpo_loss(
                policy_chosen_lp, policy_rejected_lp,
                ref_chosen_lp, ref_rejected_lp,
                rej_mask, beta=dpo_beta,
            )

            (dpo_weight * dpo_loss_val).backward()
            del ctx_padded, all_sids, ref_lp, policy_lp

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

    # Extract raw model for saving
    raw_model = policy_model.module if isinstance(policy_model, DDP) else policy_model
    raw_model.cpu()
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

    return raw_model, avg_total, n_params, train_log, train_summary


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
    parser.add_argument('--dpo_batch_size', type=int, default=32,
                        help='DPO batch size (default: 32)')
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
    log(is_main, f"SP-DPO Training — {args.name}" +
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

    # ── Determine difficulty from args or preference meta ──
    difficulty = args.difficulty
    pref_difficulty = pref_meta.get('difficulty', 'all') if os.path.exists(pref_meta_path) else 'all'
    if difficulty == 'all' and pref_difficulty != 'all':
        difficulty = pref_difficulty
        log(is_main, f"  Using difficulty from preference meta: {difficulty}")

    # ── Train ──
    log(is_main, f"\n  Training SP-DPO (difficulty={difficulty})...")
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
        meta_path = os.path.join(args.output_dir, 'train_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta['eval'] = eval_results
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            log(is_main, f"  Eval results saved to train_meta.json")

    cleanup_ddp()
    log(is_main, "\nSP-DPO training complete!")


if __name__ == '__main__':
    main()
