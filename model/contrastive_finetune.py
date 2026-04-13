"""
EXP-007: Contrastive fine-tune Qwen3-0.6B with collaborative signals.

I2I InfoNCE: 同一用户正向行为的 item pair 做正样本，in-batch negatives 做负样本。
全量参数更新, FP16, DDP (torchrun).

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        model/contrastive_finetune.py \
        --temperature 0.05 --epochs 3 --batch_size 512 --lr 1e-5 \
        --output_dir experiments/hyperparam/xxx/config_a
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))


# ============================================================
# Dataset: I2I pairs from user behavior
# ============================================================

class I2IPairDataset(Dataset):
    """Item-Item pairs from user co-interaction.

    For each user with >= 2 positive items, sample pairs of
    co-interacted items as positive pairs for contrastive learning.
    """

    def __init__(self, content_id_to_text: dict, behavior_data: dict, max_pairs: int = 5_000_000):
        """
        Args:
            content_id_to_text: {content_id: text_string}
            behavior_data: {'uid': [...], 'iid': [...], 'action_bitmap': [...]}
            max_pairs: cap total pairs to limit memory
        """
        t0 = time.time()

        # Build user -> positive items
        user_items = defaultdict(set)
        valid_cids = set(content_id_to_text.keys())

        for uid, iid, action in zip(
            behavior_data['uid'], behavior_data['iid'], behavior_data['action_bitmap']
        ):
            if action > 0 and iid in valid_cids:
                user_items[uid].add(iid)

        # Generate pairs: for each user, all combinations of positive items
        pairs = []
        for uid, items in user_items.items():
            items = list(items)
            if len(items) < 2:
                continue
            # Sample up to 10 pairs per user to avoid mega-user domination
            n = min(len(items), 10)
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((items[i], items[j]))
                    if len(pairs) >= max_pairs:
                        break
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break

        self.pairs = pairs
        self.content_id_to_text = content_id_to_text
        print(f"I2IPairDataset: {len(pairs):,} pairs from {len(user_items):,} users "
              f"({time.time() - t0:.1f}s)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cid_a, cid_b = self.pairs[idx]
        return self.content_id_to_text[cid_a], self.content_id_to_text[cid_b]


# ============================================================
# InfoNCE Loss
# ============================================================

def info_nce_loss(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor,
                  temperature: float = 0.05) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives.

    Args:
        embeddings_a: (B, D) normalized embeddings of anchor items
        embeddings_b: (B, D) normalized embeddings of positive items
        temperature: softmax temperature
    Returns:
        scalar loss
    """
    # If DDP, gather all embeddings across GPUs for larger negative pool
    if dist.is_initialized():
        all_a = [torch.zeros_like(embeddings_a) for _ in range(dist.get_world_size())]
        all_b = [torch.zeros_like(embeddings_b) for _ in range(dist.get_world_size())]
        dist.all_gather(all_a, embeddings_a)
        dist.all_gather(all_b, embeddings_b)
        # Replace own shard with original (preserves gradients)
        rank = dist.get_rank()
        all_a[rank] = embeddings_a
        all_b[rank] = embeddings_b
        all_a = torch.cat(all_a, dim=0)
        all_b = torch.cat(all_b, dim=0)
    else:
        all_a, all_b = embeddings_a, embeddings_b

    # Similarity matrix: (B_total, B_total)
    logits_ab = all_a @ all_b.t() / temperature  # a -> b
    logits_ba = all_b @ all_a.t() / temperature  # b -> a

    B = all_a.shape[0]
    labels = torch.arange(B, device=all_a.device)

    loss = (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels)) / 2
    return loss


# ============================================================
# Training loop
# ============================================================

def train(args):
    # DDP setup
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)

    device = torch.device(f'cuda:{local_rank}')
    is_main = (local_rank == 0)

    if is_main:
        print(f"Config: τ={args.temperature}, epochs={args.epochs}, "
              f"bs={args.batch_size}, lr={args.lr}, world_size={world_size}")
        print(f"Output: {args.output_dir}")

    # ── Load model ──
    from transformers import AutoModel, AutoTokenizer

    model_name = args.model_name
    if is_main:
        print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    model.train()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # ── Load data ──
    if is_main:
        print("Loading data...")

    # Load content texts
    from gr_demo.data.loaders import load_content_texts
    content_id_to_text = load_content_texts()

    # Load behavior data
    from gr_demo.eval.batch import load_all_behavior_data
    behavior_data = load_all_behavior_data()

    # Convert behavior iids to strings to match content_id_to_text keys
    behavior_data_str = {
        'uid': behavior_data['uid'],
        'iid': np.array([str(x) for x in behavior_data['iid']]),
        'action_bitmap': behavior_data['action_bitmap'],
    }

    dataset = I2IPairDataset(content_id_to_text, behavior_data_str, max_pairs=args.max_pairs)

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    # ── Training ──
    if is_main:
        print(f"Training: {len(dataset):,} pairs, {len(dataloader)} steps/epoch, "
              f"{total_steps} total steps")

    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        t_epoch = time.time()

        for batch_idx, (texts_a, texts_b) in enumerate(dataloader):
            # Tokenize
            inputs_a = tokenizer(
                list(texts_a), padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            )
            inputs_b = tokenizer(
                list(texts_b), padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            )
            inputs_a = {k: v.to(device) for k, v in inputs_a.items()}
            inputs_b = {k: v.to(device) for k, v in inputs_b.items()}

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # Forward: last hidden state, last token
                out_a = (model.module if world_size > 1 else model)(**inputs_a)
                emb_a = F.normalize(out_a.last_hidden_state[:, -1, :], dim=-1)

                out_b = (model.module if world_size > 1 else model)(**inputs_b)
                emb_b = F.normalize(out_b.last_hidden_state[:, -1, :], dim=-1)

                loss = info_nce_loss(emb_a, emb_b, temperature=args.temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if is_main and batch_idx % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  [Epoch {epoch+1}/{args.epochs}] "
                      f"Step {batch_idx}/{len(dataloader)} | "
                      f"loss={loss.item():.4f} | lr={lr_now:.2e}")

        epoch_loss /= len(dataloader)
        if is_main:
            print(f"  Epoch {epoch+1} done: avg_loss={epoch_loss:.4f} "
                  f"({time.time() - t_epoch:.0f}s)")

    # ── Save model ──
    if is_main:
        output_model_dir = os.path.join(args.output_dir, 'model')
        os.makedirs(output_model_dir, exist_ok=True)
        raw_model = model.module if world_size > 1 else model
        raw_model.save_pretrained(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)
        print(f"Model saved to {output_model_dir}")

    if world_size > 1:
        dist.destroy_process_group()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='EXP-007: Contrastive fine-tune Qwen3-0.6B')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-Embedding-0.6B',
                        help='HuggingFace model name')
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Per-GPU batch size')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_pairs', type=int, default=5_000_000,
                        help='Max I2I pairs to generate')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, default='default')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
