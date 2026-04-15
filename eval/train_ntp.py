"""NTP Probe 训练 — DDP 支持。

将 NTP probe 的数据准备 + 训练从 metric 中拆出，支持多卡 DDP。
训练产物保存到 checkpoint 目录，eval 阶段只加载 checkpoint 做推理。

Usage:
    # 单卡
    python run.py train-ntp --sid_cache experiments/sid_cache/qwen3-0.6b

    # 8卡 DDP
    torchrun --nproc_per_node=8 run.py train-ntp --sid_cache experiments/sid_cache/qwen3-0.6b

输出目录: {output_dir}/
    - probe.pt          NTPProbe state_dict + model config
    - eval_data.pt      eval sequences + eval_cids + sid_to_items
    - train_meta.json   训练元信息 (loss, n_train, n_eval, etc.)
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gr_demo.config import MODEL_CONFIGS, EFS_EMBEDDING_CACHE
from gr_demo.metrics.sid_prediction import NTPProbe, SIDSequenceDataset


# ============================================================
# DDP helpers (borrowed from contrastive_finetune.py)
# ============================================================

def setup_ddp():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main = (local_rank == 0)
    return local_rank, world_size, device, is_main


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def log(is_main, msg):
    if is_main:
        print(msg)


# ============================================================
# Data preparation
# ============================================================

def build_sequences(sid_dict, behavior_data, n_items=10, verbose_fn=print):
    """Build (input, target) sequences from SID assignments + behavior data.

    Returns:
        train_data: list of (input_tokens, target_tokens)
        eval_data:  list of (input_tokens, target_tokens)
        eval_cids:  list of target content_ids (aligned with eval_data)
        sid_to_items: dict {sid_string: set of content_ids}
        n_layers: int
        n_clusters: int
    """
    # content_id -> token list
    content_to_tokens = {}
    for cid, sid_str in sid_dict.items():
        if isinstance(sid_str, str):
            content_to_tokens[cid] = [int(t) for t in sid_str.split('_')]
        else:
            # Already a list/tuple
            content_to_tokens[cid] = [int(t) for t in sid_str]

    n_layers = len(next(iter(content_to_tokens.values())))
    n_clusters = max(max(t) for t in content_to_tokens.values()) + 1

    # SID -> items reverse mapping (for recall eval)
    sid_to_items = defaultdict(set)
    for cid, tokens in content_to_tokens.items():
        sid_to_items['_'.join(str(t) for t in tokens)].add(cid)

    # Build per-user interaction lists
    uids = behavior_data['uid']
    iids = behavior_data['iid']
    actions = behavior_data['action_bitmap']
    timestamps = behavior_data.get('first_ts')

    user_items = defaultdict(list)
    for i in range(len(uids)):
        uid, iid, action = uids[i], iids[i], actions[i]
        if action > 0 and iid in content_to_tokens:
            ts = timestamps[i] if timestamps is not None else i
            user_items[uid].append((ts, content_to_tokens[iid], iid))

    verbose_fn(f"  Users with interactions: {len(user_items):,}")

    # Sliding window samples
    all_samples = []
    for uid, items in user_items.items():
        if len(items) < n_items + 1:
            continue
        items.sort(key=lambda x: x[0])
        for i in range(len(items) - n_items):
            input_tokens = []
            for j in range(n_items):
                input_tokens.extend(items[i + j][1])
            target_tokens = items[i + n_items][1]
            target_cid = items[i + n_items][2]
            target_ts = items[i + n_items][0]
            all_samples.append((input_tokens, target_tokens, target_cid, target_ts))

    if not all_samples:
        raise ValueError("No valid sequences generated")

    # Time-sorted 80/20 split
    all_samples.sort(key=lambda x: x[3])
    split_idx = int(len(all_samples) * 0.8)

    train_data = [(s[0], s[1]) for s in all_samples[:split_idx]]
    eval_data = [(s[0], s[1]) for s in all_samples[split_idx:]]
    eval_cids = [s[2] for s in all_samples[split_idx:]]

    verbose_fn(f"  Total samples: {len(all_samples):,} "
               f"(train={len(train_data):,}, eval={len(eval_data):,})")
    verbose_fn(f"  SID: {n_layers} layers x {n_clusters} clusters")
    verbose_fn(f"  Unique SIDs with items: {len(sid_to_items):,}")

    return train_data, eval_data, eval_cids, sid_to_items, n_layers, n_clusters


# ============================================================
# Training
# ============================================================

def train_probe(
    train_data,
    n_clusters,
    n_layers,
    n_items,
    local_rank,
    world_size,
    device,
    is_main,
    batch_size=4096,
    lr=3e-3,
    embed_dim=256,
    n_heads=4,
    n_transformer_layers=2,
    ffn_dim=512,
):
    """Train NTPProbe with optional DDP. Returns trained probe (unwrapped) on CPU."""

    use_parallel = n_layers >= 5

    probe = NTPProbe(
        n_clusters=n_clusters,
        n_sid_layers=n_layers,
        n_items=n_items,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        ffn_dim=ffn_dim,
        parallel=use_parallel,
    ).to(device)

    n_params = sum(p.numel() for p in probe.parameters())
    mode_str = "parallel (MTP)" if use_parallel else "autoregressive"
    log(is_main, f"  NTPProbe: {n_params / 1e6:.1f}M params, {mode_str}")

    # DDP wrap
    if world_size > 1:
        probe = DDP(probe, device_ids=[local_rank])

    dataset = SIDSequenceDataset(train_data)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
    n_batches = len(train_loader)

    log(is_main, f"  Training: {len(train_data):,} samples, "
                 f"{n_batches} batches/epoch, batch_size={batch_size}, "
                 f"world_size={world_size}")

    probe.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (input_batch, target_batch) in enumerate(train_loader):
        input_batch = input_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)

        if use_parallel:
            logits = probe(input_batch)
            loss = F.cross_entropy(
                logits.reshape(-1, n_clusters), target_batch.reshape(-1)
            )
        else:
            teacher_input = torch.cat([input_batch, target_batch[:, :-1]], dim=1)
            logits = probe(teacher_input, return_last_n=n_layers)
            loss = F.cross_entropy(
                logits.reshape(-1, n_clusters), target_batch.reshape(-1)
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if is_main and (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            samples_per_sec = (step + 1) * batch_size * world_size / elapsed
            print(f"    step {step+1}/{n_batches}: "
                  f"loss={total_loss/(step+1):.4f}, "
                  f"{samples_per_sec:.0f} samples/s")

    avg_loss = total_loss / n_batches
    elapsed = time.time() - t0
    log(is_main, f"  Train done: loss={avg_loss:.4f} ({elapsed:.1f}s)")

    # Unwrap DDP
    raw_probe = probe.module if isinstance(probe, DDP) else probe
    return raw_probe.cpu(), avg_loss, n_params


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(output_dir, probe, train_data, eval_data, eval_cids,
                    sid_to_items, n_clusters, n_layers, n_items,
                    avg_loss, n_params, sid_cache_dir):
    """Save probe checkpoint + eval data (rank 0 only)."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Probe checkpoint
    probe_config = {
        'n_clusters': n_clusters,
        'n_sid_layers': n_layers,
        'n_items': n_items,
        'embed_dim': probe.embed_dim,
        'n_heads': probe.decoder.layers[0].self_attn.num_heads,
        'n_transformer_layers': len(probe.decoder.layers),
        'ffn_dim': probe.decoder.layers[0].linear1.out_features,
        'parallel': probe.parallel,
    }
    torch.save({
        'model_state_dict': probe.state_dict(),
        'config': probe_config,
    }, os.path.join(output_dir, 'probe.pt'))

    # 2. Eval data
    torch.save({
        'eval_data': eval_data,
        'eval_cids': eval_cids,
        'sid_to_items': dict(sid_to_items),  # defaultdict -> dict for serialization
    }, os.path.join(output_dir, 'eval_data.pt'))

    # 3. Train metadata
    meta = {
        'n_train': len(train_data),
        'n_eval': len(eval_data),
        'n_clusters': n_clusters,
        'n_layers': n_layers,
        'n_items': n_items,
        'n_params': n_params,
        'train_loss': round(avg_loss, 6),
        'sid_cache': sid_cache_dir,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(output_dir, 'train_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"    probe.pt      ({os.path.getsize(os.path.join(output_dir, 'probe.pt')) / 1e6:.1f}MB)")
    print(f"    eval_data.pt  ({os.path.getsize(os.path.join(output_dir, 'eval_data.pt')) / 1e6:.1f}MB)")
    print(f"    train_meta.json")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train NTP Probe (DDP)')
    parser.add_argument('--sid_cache', type=str, required=True,
                        help='Path to preprocess-sid cache dir')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir (default: experiments/ntp_checkpoints/{name})')
    parser.add_argument('--name', type=str, default='default',
                        help='Experiment name for output subdir')
    parser.add_argument('--n_items', type=int, default=10,
                        help='Number of history items per sequence')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_transformer_layers', type=int, default=2)
    parser.add_argument('--ffn_dim', type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank, world_size, device, is_main = setup_ddp()

    log(is_main, "=" * 60)
    log(is_main, "NTP Probe Training (DDP)" if world_size > 1 else "NTP Probe Training")
    log(is_main, "=" * 60)

    # ── Load SID cache ──
    log(is_main, f"\nStep 1: Loading SID cache from {args.sid_cache}")
    cache_config_path = os.path.join(args.sid_cache, 'config.json')
    with open(cache_config_path) as f:
        cache_config = json.load(f)

    sid_dict = np.load(
        os.path.join(args.sid_cache, 'semantic_ids.npy'), allow_pickle=True
    ).item()
    log(is_main, f"  SID assignments: {len(sid_dict):,}")

    # ── Load behavior data ──
    log(is_main, "\nStep 2: Loading behavior data")
    from gr_demo.eval.batch import load_all_behavior_data
    behavior_data = load_all_behavior_data()
    log(is_main, f"  Interactions: {len(behavior_data['uid']):,}")

    # ── Build sequences (all ranks do this identically) ──
    log(is_main, "\nStep 3: Building user sequences")
    verbose_fn = (lambda msg: print(msg)) if is_main else (lambda msg: None)
    train_data, eval_data, eval_cids, sid_to_items, n_layers, n_clusters = \
        build_sequences(sid_dict, behavior_data, n_items=args.n_items,
                        verbose_fn=verbose_fn)

    # ── Train ──
    log(is_main, f"\nStep 4: Training")
    probe, avg_loss, n_params = train_probe(
        train_data=train_data,
        n_clusters=n_clusters,
        n_layers=n_layers,
        n_items=args.n_items,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=is_main,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_transformer_layers,
        ffn_dim=args.ffn_dim,
    )

    # ── Save (rank 0 only) ──
    if is_main:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = args.output_dir or os.path.join(
            repo_root, 'experiments', 'ntp_checkpoints', args.name)

        log(is_main, f"\nStep 5: Saving checkpoint")
        save_checkpoint(
            output_dir=output_dir,
            probe=probe,
            train_data=train_data,
            eval_data=eval_data,
            eval_cids=eval_cids,
            sid_to_items=sid_to_items,
            n_clusters=n_clusters,
            n_layers=n_layers,
            n_items=args.n_items,
            avg_loss=avg_loss,
            n_params=n_params,
            sid_cache_dir=args.sid_cache,
        )

        log(is_main, f"\n{'=' * 60}")
        log(is_main, "Training complete!")
        log(is_main, f"{'=' * 60}")
        log(is_main, f"\nNext: python run.py hyperparam --sid_cache {args.sid_cache} "
                      f"--ntp_checkpoint {output_dir} --run_ntp")

    cleanup_ddp()


if __name__ == '__main__':
    main()
