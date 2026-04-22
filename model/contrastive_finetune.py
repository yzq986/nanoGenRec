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
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

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
    """Item-Item pairs from collaborative signals.

    Two pair construction methods (combined):

    方式 1 — Adjacent Positive Pairs:
        Per user, sort positive interactions by first_ts.
        Pair each item with the immediately preceding positive item.
        Captures temporal co-interest within a session.

    方式 2 — Swing I2I High-Score Pairs:
        Swing(i,j) = Σ_{u∈U(i)∩U(j)} 1 / (α + |I(u)|)
        Select top-scoring pairs as high-quality collaborative signals.
    """

    def __init__(
        self,
        content_id_to_text: dict,
        behavior_data: dict,
        max_pairs: int = 5_000_000,
        swing_top_k: int = 10,
        swing_alpha: float = 5.0,
        swing_max_items: int = 500_000,
    ):
        """
        Args:
            content_id_to_text: {content_id: text_string}
            behavior_data: {'uid': [...], 'iid': [...], 'action_bitmap': [...], 'first_ts': [...]}
            max_pairs: cap total pairs
            swing_top_k: per-item top-K swing neighbors to keep
            swing_alpha: swing smoothing factor
            swing_max_items: max items for swing computation (memory bound)
        """
        t0 = time.time()
        valid_cids = set(content_id_to_text.keys())

        uids = behavior_data['uid']
        iids = behavior_data['iid']
        actions = behavior_data['action_bitmap']
        timestamps = behavior_data.get('first_ts')

        # ── Build user -> sorted positive items ──
        user_items_ts = defaultdict(list)  # uid -> [(ts, iid), ...]
        item_users = defaultdict(set)      # iid -> {uid, ...}

        for i in range(len(uids)):
            uid, iid, action = uids[i], iids[i], actions[i]
            if action > 0 and iid in valid_cids:
                ts = timestamps[i] if timestamps is not None else 0
                user_items_ts[uid].append((ts, iid))
                item_users[iid].add(uid)

        # ── 方式 1: Adjacent Positive Pairs ──
        adjacent_pairs = set()
        for uid, ts_items in user_items_ts.items():
            if len(ts_items) < 2:
                continue
            # Sort by timestamp
            sorted_items = sorted(ts_items, key=lambda x: x[0])
            # Deduplicate consecutive items
            prev_iid = sorted_items[0][1]
            for _, iid in sorted_items[1:]:
                if iid != prev_iid:
                    pair = (prev_iid, iid) if prev_iid < iid else (iid, prev_iid)
                    adjacent_pairs.add(pair)
                prev_iid = iid

        print(f"  方式 1 (adjacent): {len(adjacent_pairs):,} unique pairs "
              f"from {len(user_items_ts):,} users")

        # ── 方式 2: Swing I2I ──
        # Filter to items with enough co-occurrence potential
        active_items = [iid for iid, users in item_users.items() if len(users) >= 2]
        if len(active_items) > swing_max_items:
            import random
            random.seed(42)
            active_items = random.sample(active_items, swing_max_items)
        active_set = set(active_items)

        # Build inverted index: user -> items (filtered)
        user_active_items = defaultdict(set)
        for iid in active_items:
            for uid in item_users[iid]:
                user_active_items[uid].add(iid)

        # Compute swing scores via co-occurrence users
        swing_pairs = defaultdict(float)  # (i,j) -> score
        for uid, items in user_active_items.items():
            items = list(items)
            user_weight = 1.0 / (swing_alpha + len(items))
            # Only compute for users with manageable item count
            if len(items) > 200:
                continue
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    pair = (items[a], items[b]) if items[a] < items[b] else (items[b], items[a])
                    swing_pairs[pair] += user_weight

        # Select top pairs globally
        swing_pair_list = sorted(swing_pairs.items(), key=lambda x: -x[1])
        swing_top_pairs = set()
        for (i, j), score in swing_pair_list:
            swing_top_pairs.add((i, j))
            if len(swing_top_pairs) >= max_pairs // 2:
                break

        print(f"  方式 2 (swing): {len(swing_top_pairs):,} pairs "
              f"(from {len(swing_pairs):,} total co-occurrence pairs)")

        # ── Merge ──
        all_pairs = list(adjacent_pairs | swing_top_pairs)
        import random
        random.seed(42)
        random.shuffle(all_pairs)
        if len(all_pairs) > max_pairs:
            all_pairs = all_pairs[:max_pairs]

        self.pairs = all_pairs
        self.content_id_to_text = content_id_to_text
        print(f"I2IPairDataset: {len(self.pairs):,} total pairs "
              f"(adjacent={len(adjacent_pairs):,}, swing={len(swing_top_pairs):,}) "
              f"({time.time() - t0:.1f}s)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cid_a, cid_b = self.pairs[idx]
        return self.content_id_to_text[cid_a], self.content_id_to_text[cid_b], cid_a, cid_b


# ============================================================
# Inline HR@50 Monitor (zero-overhead embedding reuse)
# ============================================================

class InlineHRMonitor:
    """Accumulate embeddings from training batches, compute HR@50 periodically.

    Key insight: training forward pass already computes item embeddings.
    We just detach + move to CPU — zero extra GPU compute.
    When enough unique items accumulate, run FAISS HR@50 on CPU.
    """

    def __init__(self, behavior_data: dict, eval_interval: int = 2000, min_items: int = 10000):
        """
        Args:
            behavior_data: {'uid': [...], 'iid': [...], 'action_bitmap': [...]}
            eval_interval: compute HR@50 every N micro-steps
            min_items: minimum unique items before first eval
        """
        self.eval_interval = eval_interval
        self.min_items = min_items
        self.embedding_buffer = {}  # cid -> embedding (numpy, float32)

        # Pre-compute co-occurrence map from behavior data
        t0 = time.time()
        user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        for uid, iid, action in zip(
            behavior_data['uid'], behavior_data['iid'], behavior_data['action_bitmap']
        ):
            if action > 0:
                user_items[uid].add(iid)
                self.item_users[iid].add(uid)
        self.user_items = user_items
        print(f"  InlineHRMonitor: pre-computed co-occurrence "
              f"({len(self.item_users):,} items, {len(user_items):,} users, "
              f"{time.time() - t0:.1f}s)")

    def update(self, cids: list, embeddings: torch.Tensor):
        """Cache embeddings from a training batch. Called every step on rank 0.

        Args:
            cids: list of content_id strings
            embeddings: (N, D) tensor, already normalized, on GPU
        """
        embs_cpu = embeddings.detach().cpu().numpy()
        for i, cid in enumerate(cids):
            self.embedding_buffer[cid] = embs_cpu[i]

    def maybe_eval(self, step: int) -> Optional[float]:
        """Compute HR@50 if interval reached and enough items. Returns HR@50 or None."""
        if step % self.eval_interval != 0 or step == 0:
            return None
        if len(self.embedding_buffer) < self.min_items:
            return None
        return self._compute_hr50()

    def _compute_hr50(self, top_k: int = 50) -> float:
        """Compute HR@50 on buffered embeddings using FAISS."""
        import faiss

        cids = list(self.embedding_buffer.keys())
        embs = np.array([self.embedding_buffer[c] for c in cids], dtype=np.float32)
        N, D = embs.shape

        # Build FAISS index
        index = faiss.IndexFlatIP(D)
        index.add(embs)
        _, I = index.search(embs, top_k + 1)  # +1 to exclude self

        # Compute HR@50
        hit_rates = []
        for i, cid in enumerate(cids):
            if cid not in self.item_users:
                continue
            # Co-occurrence items for this cid
            cooccur = set()
            for uid in self.item_users[cid]:
                cooccur.update(self.user_items[uid])
            cooccur.discard(cid)
            if not cooccur:
                continue

            # Top-K neighbors (exclude self)
            neighbors = [cids[j] for j in I[i] if j != i][:top_k]
            hits = sum(1 for n in neighbors if n in cooccur)
            hit_rates.append(hits / len(cooccur) if cooccur else 0.0)

        return float(np.mean(hit_rates)) if hit_rates else 0.0


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

    # ── Dry run: override to minimal config ──
    if args.dry_run:
        args.max_pairs = 10_000
        args.epochs = 1
        if is_main:
            print("=" * 40)
            print("DRY RUN: 1% data, 1 epoch, 10 steps")
            print("=" * 40)

    if is_main:
        print(f"Config: τ={args.temperature}, epochs={args.epochs}, "
              f"bs={args.batch_size}, lr={args.lr}, world_size={world_size}")
        print(f"Output: {args.output_dir}")

        # W&B init (rank 0 only)
        try:
            import wandb
            wandb_config = {
                    'temperature': args.temperature,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'grad_accum': args.grad_accum,
                    'lr': args.lr,
                    'max_pairs': args.max_pairs,
                    'world_size': world_size,
                    'effective_batch': args.batch_size * args.grad_accum * world_size,
                    'model_name': args.model_name,
                    'dry_run': args.dry_run,
                    'use_qformer': args.use_qformer,
            }
            if args.use_qformer:
                wandb_config.update({
                    'qformer_layers': args.qformer_layers,
                    'qformer_queries': args.qformer_queries,
                    'qformer_heads': args.qformer_heads,
                })
            wandb.init(
                project="gr-demo",
                name=args.experiment_name,
                config=wandb_config,
            )
            print("W&B initialized")
        except Exception as e:
            wandb = None
            print(f"W&B not available: {e}")
    else:
        wandb = None

    # ── Load model ──
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

    model_name = args.model_name
    if is_main:
        print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    # ── QFormer: freeze base model, train cross-attention compressor ──
    qformer = None
    if args.use_qformer:
        from model.qformer import QFormer as QFormerModule

        model.requires_grad_(False)
        model.eval()  # frozen encoder stays in eval mode (no dropout)
        # Disable gradient checkpointing for frozen encoder (not needed)
        model.gradient_checkpointing_disable()

        # Detect encoder hidden dim from model config
        d_enc = model.config.hidden_size  # 1024 for Qwen3-0.6B

        qformer = QFormerModule(
            num_queries=args.qformer_queries,
            num_layers=args.qformer_layers,
            d_model=d_enc,  # match encoder dim for simplicity
            d_enc=d_enc,
            n_heads=args.qformer_heads,
            dropout=args.qformer_dropout,
        ).to(device)
        qformer.train()

        if is_main:
            print(f"QFormer enabled: {qformer.param_count()} "
                  f"(queries={args.qformer_queries}, layers={args.qformer_layers}, "
                  f"d={d_enc}, heads={args.qformer_heads})")
            base_params = sum(p.numel() for p in model.parameters())
            print(f"  Base model frozen: {base_params/1e6:.1f}M params (no grad)")

    # ── LoRA: freeze base model, inject trainable adapters ──
    elif args.lora:
        from peft import LoraConfig, get_peft_model
        model.requires_grad_(False)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if is_main:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"LoRA enabled: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total "
                  f"({trainable/total*100:.1f}%)")

    # ── LM head for caption reconstruction loss (frozen probe; --cap_loss_weight>0 enables BP) ──
    lm_head = None
    LM_MODEL_NAME = "Qwen/Qwen3-0.6B"
    if is_main:
        cap_mode = "training" if args.cap_loss_weight > 0 else "monitor only"
        print(f"Loading LM head from {LM_MODEL_NAME} (frozen, {cap_mode})...")
    try:
        causal_model = AutoModelForCausalLM.from_pretrained(
            LM_MODEL_NAME, torch_dtype=torch.bfloat16
        )
        lm_head = causal_model.lm_head.to(device)
        lm_head.requires_grad_(False)
        lm_head.eval()
        del causal_model
        torch.cuda.empty_cache()
        if is_main:
            print(f"  LM head loaded: {lm_head.in_features}→{lm_head.out_features} (frozen)")
    except Exception as e:
        if is_main:
            print(f"  LM head not available, skipping caption loss: {e}")
        lm_head = None

    if world_size > 1:
        if args.use_qformer:
            # Only QFormer needs DDP (base model is frozen, no gradient sync)
            qformer = DDP(qformer, device_ids=[local_rank])
        else:
            model = DDP(model, device_ids=[local_rank])

    # ── Load data (rank 0 builds dataset, then broadcast to all ranks) ──
    if is_main:
        print("Loading data...")
        from data.loaders import load_content_texts
        content_id_to_text = load_content_texts()

        from eval.batch import load_all_behavior_data
        behavior_data = load_all_behavior_data()

        behavior_data_str = {
            'uid': behavior_data['uid'],
            'iid': np.array([str(x) for x in behavior_data['iid']]),
            'action_bitmap': behavior_data['action_bitmap'],
            'first_ts': behavior_data.get('first_ts', np.zeros(len(behavior_data['uid']), dtype=np.int64)),
        }

        dataset = I2IPairDataset(content_id_to_text, behavior_data_str, max_pairs=args.max_pairs)
        # Share pairs + text mapping to other ranks via broadcast
        shared = (dataset.pairs, dataset.content_id_to_text)
    else:
        shared = None

    if world_size > 1:
        import pickle
        if is_main:
            data_bytes = pickle.dumps(shared)
            size_tensor = torch.tensor([len(data_bytes)], dtype=torch.long, device=device)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(size_tensor, src=0)

        if is_main:
            data_tensor = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).to(device)
        else:
            data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device=device)
        dist.broadcast(data_tensor, src=0)

        if not is_main:
            shared = pickle.loads(data_tensor.cpu().numpy().tobytes())

        pairs, content_id_to_text = shared
        dataset = I2IPairDataset.__new__(I2IPairDataset)
        dataset.pairs = pairs
        dataset.content_id_to_text = content_id_to_text
        print(f"[Rank {local_rank}] Received {len(dataset.pairs):,} pairs")

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer ──
    # Gradient accumulation: effective_batch = batch_size * grad_accum * world_size
    grad_accum = args.grad_accum
    effective_batch = args.batch_size * grad_accum * world_size
    if is_main:
        print(f"Batch: {args.batch_size}/GPU × {grad_accum} accum × {world_size} GPUs = {effective_batch} effective")

    train_params = qformer.parameters() if args.use_qformer else model.parameters()
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.01)
    total_steps = (len(dataloader) // grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # BF16 on A100: no GradScaler needed (sufficient dynamic range)

    # ── Training ──
    if is_main:
        print(f"Training: {len(dataset):,} pairs, {len(dataloader)} steps/epoch, "
              f"{total_steps} total steps")

    # ── Inline HR@50 monitor (rank 0 only, zero GPU overhead) ──
    hr_monitor = None
    if is_main:
        hr_monitor = InlineHRMonitor(behavior_data_str, eval_interval=2000, min_items=10000)

    # ── Eval-only mode: no training, just compute baseline HR@50 ──
    if args.eval_only and is_main:
        print("Eval-only mode: computing baseline HR@50 (no training)")
        model.eval()
        if qformer is not None:
            qformer.eval()
        with torch.no_grad():
            for batch_idx, (texts_a, texts_b, cids_a, cids_b) in enumerate(dataloader):
                all_texts = list(texts_a) + list(texts_b)
                inputs = tokenizer(all_texts, padding=True, truncation=True,
                                   max_length=256, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    raw_model = model.module if (world_size > 1 and not args.use_qformer) else model
                    out = raw_model(**inputs)
                    if args.use_qformer:
                        enc_mask = inputs['attention_mask'].bool()
                        raw_qf = qformer.module if (world_size > 1 and isinstance(qformer, DDP)) else qformer
                        pooled = raw_qf(out.last_hidden_state, enc_mask)
                        embeddings = F.normalize(pooled.float(), dim=-1)
                    else:
                        embeddings = F.normalize(out.last_hidden_state[:, -1, :].float(), dim=-1)
                all_cids = list(cids_a) + list(cids_b)
                hr_monitor.update(all_cids, embeddings.detach())
                if (batch_idx + 1) % 100 == 0:
                    print(f"  Encoded {len(hr_monitor.embedding_buffer):,} items ({batch_idx+1} batches)")
                if len(hr_monitor.embedding_buffer) >= 50000:
                    break
        hr50 = hr_monitor._compute_hr50()
        n_items = len(hr_monitor.embedding_buffer)
        print(f"Baseline HR@50 = {hr50:.4f} ({n_items:,} items)")
        os.makedirs(args.output_dir, exist_ok=True)
        results = {
            'experiment_name': args.experiment_name,
            'model_name': args.model_name,
            'lora': False, 'cap_loss_weight': 0,
            'final_hr50': round(hr50, 4), 'hr50_items': n_items,
            'eval_only': True,
        }
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output_dir}/results.json")
        if world_size > 1:
            dist.destroy_process_group()
        return

    global_step = 0
    log_entries = []  # accumulate per-print-step metrics for results.json
    t_train_start = time.time()
    total_micro_steps = len(dataloader) * args.epochs

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        t_epoch = time.time()

        for batch_idx, (texts_a, texts_b, cids_a, cids_b) in enumerate(dataloader):
            # Tokenize both batches together → single forward pass
            all_texts = list(texts_a) + list(texts_b)
            inputs = tokenizer(
                all_texts, padding=True, truncation=True,
                max_length=256, return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            B = len(texts_a)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                raw_model = model.module if (world_size > 1 and not args.use_qformer) else model

                if args.use_qformer:
                    # Frozen encoder → full hidden states → QFormer → pooled embedding
                    with torch.no_grad():
                        out = raw_model(**inputs)
                        encoder_hidden = out.last_hidden_state  # (2B, S, D_enc)
                    # attention_mask as bool for QFormer cross-attention masking
                    enc_mask = inputs['attention_mask'].bool()  # (2B, S)
                    pooled = qformer(encoder_hidden.detach(), enc_mask)  # (2B, d_model)
                    embeddings = F.normalize(pooled.float(), dim=-1)
                else:
                    out = raw_model(**inputs)
                    embeddings = F.normalize(out.last_hidden_state[:, -1, :].float(), dim=-1)

                emb_a, emb_b = embeddings[:B], embeddings[B:]

                contrastive_loss = info_nce_loss(emb_a, emb_b, temperature=args.temperature)
                loss = contrastive_loss / grad_accum

                # Caption reconstruction loss
                # Note: with QFormer, cap_loss uses encoder hidden states (not QFormer output)
                # to monitor semantic preservation of the frozen base model.
                caption_loss_val = None
                # Caption loss: lm_head(hidden) produces (N, S, vocab~150K) — OOM risk.
                # Monitor-only paths subsample to CAP_MONITOR_N samples.
                # Training path chunks across batch dim.
                CAP_MONITOR_N = 4
                if lm_head is not None and args.cap_loss_weight > 0:
                    if args.use_qformer:
                        # Encoder is frozen → cap_loss is monitor-only in QFormer mode
                        with torch.no_grad():
                            h = out.last_hidden_state[:CAP_MONITOR_N]
                            l = inputs['input_ids'][:CAP_MONITOR_N]
                            logits = lm_head(h)
                            sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                            st = l[:, 1:].contiguous().view(-1)
                            caption_loss_val = F.cross_entropy(
                                sl, st, ignore_index=tokenizer.pad_token_id).item()
                    else:
                        # Gradient flows through hidden_states → model (lm_head stays frozen)
                        # Chunk to avoid OOM on (2B, S, vocab)
                        hidden = out.last_hidden_state  # (2B, S, D)
                        labels = inputs['input_ids']    # (2B, S)
                        cap_losses = []
                        for chunk_h, chunk_l in zip(hidden.chunk(4), labels.chunk(4)):
                            logits = lm_head(chunk_h)
                            sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                            st = chunk_l[:, 1:].contiguous().view(-1)
                            cap_losses.append(F.cross_entropy(
                                sl, st, ignore_index=tokenizer.pad_token_id))
                        cap_loss = torch.stack(cap_losses).mean()
                        loss = loss + args.cap_loss_weight * cap_loss / grad_accum
                        caption_loss_val = cap_loss.item()
                elif lm_head is not None and is_main and batch_idx % (50 * grad_accum) == 0:
                    # Monitor only (no BP), subsample to avoid OOM
                    with torch.no_grad():
                        h = (out.last_hidden_state[:CAP_MONITOR_N].detach()
                             if not args.use_qformer
                             else out.last_hidden_state[:CAP_MONITOR_N])
                        l = inputs['input_ids'][:CAP_MONITOR_N]
                        logits = lm_head(h)
                        sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                        st = l[:, 1:].contiguous().view(-1)
                        caption_loss_val = F.cross_entropy(
                            sl, st, ignore_index=tokenizer.pad_token_id).item()

            loss.backward()

            # Cache embeddings for inline HR@50 (rank 0 only, detach→CPU, zero GPU cost)
            if hr_monitor is not None:
                all_cids = list(cids_a) + list(cids_b)
                hr_monitor.update(all_cids, embeddings.detach())

            if (batch_idx + 1) % grad_accum == 0:
                clip_params = qformer.parameters() if args.use_qformer else model.parameters()
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * grad_accum  # unscale for logging

            # Dry run: stop after 10 steps
            if args.dry_run and batch_idx >= 10:
                if is_main:
                    print(f"  DRY RUN: stopped after {batch_idx} steps, loss={loss.item() * grad_accum:.4f}")
                break

            if is_main and batch_idx % (50 * grad_accum) == 0:
                global_micro = epoch * len(dataloader) + batch_idx
                elapsed = time.time() - t_train_start
                if global_micro > 0:
                    speed = global_micro / elapsed  # steps/sec
                    remaining = (total_micro_steps - global_micro) / speed
                    eta_m, eta_s = divmod(int(remaining), 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"
                else:
                    eta_str = "..."
                lr_now = scheduler.get_last_lr()[0]

                # Inline HR@50: compute every print step
                hr_str = ""
                hr50 = None
                if hr_monitor is not None and len(hr_monitor.embedding_buffer) >= hr_monitor.min_items:
                    hr50 = hr_monitor._compute_hr50()
                    hr_str = f" | HR@50={hr50:.4f} ({len(hr_monitor.embedding_buffer):,} items)"

                pairs_seen = global_micro * args.batch_size * world_size
                cur_loss = round(loss.item() * grad_accum, 4)
                cap_str = f" | cap_loss={caption_loss_val:.4f}" if caption_loss_val is not None else ""
                print(f"  [Epoch {epoch+1}/{args.epochs}] "
                      f"Step {batch_idx}/{len(dataloader)} ({pairs_seen/1e6:.2f}M pairs) | "
                      f"loss={cur_loss:.4f} | lr={lr_now:.2e} | "
                      f"ETA {eta_str}{hr_str}{cap_str}")

                entry = {'step': batch_idx, 'pairs_seen': pairs_seen,
                         'loss': cur_loss, 'lr': lr_now, 'epoch': epoch + 1}
                if hr50 is not None:
                    entry['hr50'] = round(hr50, 4)
                log_entries.append(entry)

                # W&B log — x-axis = pairs_seen (cross-experiment comparable)
                if wandb is not None:
                    log_dict = {
                        'loss': loss.item() * grad_accum,
                        'lr': lr_now,
                        'throughput': speed if global_micro > 0 else 0,
                        'pairs_seen': pairs_seen,
                        'epoch': epoch + batch_idx / len(dataloader),
                        'buffer_items': len(hr_monitor.embedding_buffer) if hr_monitor else 0,
                    }
                    if hr50 is not None:
                        log_dict['HR@50'] = hr50
                    if caption_loss_val is not None:
                        log_dict['caption_loss'] = caption_loss_val
                    wandb.log(log_dict, step=pairs_seen)

        epoch_loss /= len(dataloader)
        if is_main:
            elapsed_total = time.time() - t_train_start
            epochs_done = epoch + 1
            eta_remaining = elapsed_total / epochs_done * (args.epochs - epochs_done)
            eta_m, eta_s = divmod(int(eta_remaining), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_s:02d}s"
            # Epoch-end HR@50
            hr_epoch_str = ""
            if hr_monitor is not None:
                hr50 = hr_monitor._compute_hr50()
                hr_epoch_str = f" | HR@50={hr50:.4f} ({len(hr_monitor.embedding_buffer):,} items)"
            print(f"  Epoch {epoch+1} done: avg_loss={epoch_loss:.4f} "
                  f"({time.time() - t_epoch:.0f}s) | ETA remaining: {eta_str}{hr_epoch_str}")

    # ── Save model + results ──
    if is_main:
        output_model_dir = os.path.join(args.output_dir, 'model')
        os.makedirs(output_model_dir, exist_ok=True)

        if args.use_qformer:
            # Save QFormer weights + config (base model is unchanged, no need to save)
            raw_qf = qformer.module if isinstance(qformer, DDP) else qformer
            qformer_path = os.path.join(output_model_dir, 'qformer.pt')
            torch.save({
                'state_dict': raw_qf.state_dict(),
                'config': {
                    'num_queries': args.qformer_queries,
                    'num_layers': args.qformer_layers,
                    'd_model': raw_qf.d_model,
                    'd_enc': model.config.hidden_size,
                    'n_heads': args.qformer_heads,
                    'dropout': args.qformer_dropout,
                    'base_model': args.model_name,
                },
            }, qformer_path)
            tokenizer.save_pretrained(output_model_dir)
            print(f"QFormer saved to {qformer_path} ({raw_qf.param_count()})")
        else:
            raw_model = model.module if world_size > 1 else model
            if args.lora:
                raw_model = raw_model.merge_and_unload()
                print("LoRA merged into base model")
            raw_model.save_pretrained(output_model_dir)
            tokenizer.save_pretrained(output_model_dir)
            print(f"Model saved to {output_model_dir}")

        # Save training results summary
        final_hr50 = None
        if hr_monitor is not None and len(hr_monitor.embedding_buffer) >= hr_monitor.min_items:
            final_hr50 = hr_monitor._compute_hr50()
        results = {
            'experiment_name': args.experiment_name,
            'model_name': args.model_name,
            'temperature': args.temperature,
            'epochs': args.epochs,
            'lr': args.lr,
            'lora': args.lora,
            'lora_rank': args.lora_rank if args.lora else None,
            'use_qformer': args.use_qformer,
            'qformer_layers': args.qformer_layers if args.use_qformer else None,
            'qformer_queries': args.qformer_queries if args.use_qformer else None,
            'cap_loss_weight': args.cap_loss_weight,
            'batch_size': args.batch_size,
            'grad_accum': args.grad_accum,
            'max_pairs': args.max_pairs,
            'actual_pairs': len(dataset),
            'world_size': world_size,
            'effective_batch_size': args.batch_size * args.grad_accum * world_size,
            'final_avg_loss': round(epoch_loss, 4),
            'final_hr50': round(final_hr50, 4) if final_hr50 is not None else None,
            'hr50_items': len(hr_monitor.embedding_buffer) if hr_monitor else 0,
            'training_time_s': round(time.time() - t_train_start, 1),
            'log': log_entries,
        }
        results_path = os.path.join(args.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

        if wandb is not None:
            wandb.finish()

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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Per-GPU batch size (reduce to 16 if OOM)')
    parser.add_argument('--grad_accum', type=int, default=8,
                        help='Gradient accumulation steps (effective_batch = batch_size * grad_accum * n_gpus)')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_pairs', type=int, default=500_000,
                        help='Max I2I pairs to generate (500K ≈ 15min on 8xA100)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Smoke test: 1%% data, 1 epoch, 10 steps, verify full pipeline')
    parser.add_argument('--eval_only', action='store_true',
                        help='No training — forward pass to compute baseline HR@50')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, default='default')
    parser.add_argument('--cap_loss_weight', type=float, default=0.0,
                        help='Caption reconstruction loss weight (0=monitor only, >0=train with it)')
    parser.add_argument('--lora', action='store_true',
                        help='Freeze base model, train LoRA adapters only (requires peft)')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank (default: 16, ~10M trainable params)')
    parser.add_argument('--use_qformer', action='store_true',
                        help='Freeze Qwen3, train QFormer cross-attention compressor on top')
    parser.add_argument('--qformer_layers', type=int, default=2,
                        help='Number of QFormer layers (default: 2)')
    parser.add_argument('--qformer_queries', type=int, default=4,
                        help='Number of learnable query tokens M (default: 4)')
    parser.add_argument('--qformer_heads', type=int, default=16,
                        help='QFormer attention heads (default: 16)')
    parser.add_argument('--qformer_dropout', type=float, default=0.1,
                        help='QFormer dropout rate')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
