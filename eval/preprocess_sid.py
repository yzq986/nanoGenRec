"""SID 预处理 — 训练 tokenizer 并缓存 SID assignments。

固定 tokenizer config 后一次性跑完，后续 NTP 实验直接加载缓存。

Usage:
    python run.py preprocess-sid --model qwen3-0.6b --behavior_path auto

输出目录: experiments/sid_cache/{model_key}/
    - quantizer.pt       训练好的 ResKmeansFSQ 模型
    - semantic_ids.npy    content_id → SID 映射 (dict)
    - config.json         量化参数 + embedding 指纹
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from gr_demo.config import MODEL_CONFIGS, EFS_EMBEDDING_CACHE
from gr_demo.data.loaders import load_exposed_iids
from gr_demo.model.rkmeans_fsq import ResKmeansFSQ, generate_semantic_ids_fsq
from gr_demo.model.fsq import FSQ_LEVEL_CONFIGS


# ============================================================
# 固定 tokenizer 参数 (EXP-008 winner)
# ============================================================

TOKENIZER_CONFIG = {
    'num_clusters': 1024,
    'niter': 25,
    'nredo': 3,
    'num_kmeans_layers': 2,
    'normalize_residuals': True,
    'fsq_levels_key': '6d_4096',
    'fsq_projection': 'mlp',
    'fsq_mlp_hidden': 64,
    'fsq_epochs': 50,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess SID: train tokenizer + cache assignments')
    parser.add_argument('--model', type=str, default='qwen3-0.6b',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Embedding model key')
    parser.add_argument('--behavior_path', type=str, default='auto',
                        help='Behavior data path for exposure filter ("auto" = S3 date range)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: experiments/sid_cache/{model})')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    model_key = args.model
    _, embedding_dim, _, _ = MODEL_CONFIGS[model_key]

    # ── Output dir ──
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(repo_root, 'experiments', 'sid_cache', model_key)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("SID Preprocessing")
    print("=" * 60)
    print(f"  Model:      {model_key} (dim={embedding_dim})")
    print(f"  Tokenizer:  KMeans {TOKENIZER_CONFIG['num_clusters']}x2 + MLP-FSQ h={TOKENIZER_CONFIG['fsq_mlp_hidden']}")
    print(f"  Output:     {output_dir}")
    print()

    # ── Step 1: Load embeddings ──
    print("Step 1: Loading embeddings...")
    t0 = time.time()

    embedding_cache_dir = f'{EFS_EMBEDDING_CACHE}/{model_key}'
    incremental_cache_path = f'{embedding_cache_dir}/incremental_cache.npy'
    embedding_cache_path = f'{embedding_cache_dir}/embeddings.npy'
    content_ids_cache_path = f'{embedding_cache_dir}/content_ids.npy'

    if os.path.exists(incremental_cache_path):
        cache_dict = np.load(incremental_cache_path, allow_pickle=True).item()
        content_ids = np.array(list(cache_dict.keys()))
        embeddings = np.array(list(cache_dict.values()), dtype=np.float32)
        print(f"  Loaded {len(content_ids):,} embeddings from incremental_cache, dim={embeddings.shape[1]}")
    elif os.path.exists(embedding_cache_path) and os.path.exists(content_ids_cache_path):
        embeddings = np.load(embedding_cache_path, allow_pickle=True)
        content_ids = np.load(content_ids_cache_path, allow_pickle=True)
        print(f"  Loaded {len(content_ids):,} embeddings from cache, dim={embeddings.shape[1]}")
    else:
        raise FileNotFoundError(f"Cache not found at {embedding_cache_dir}. Run encode first.")

    # ── Step 2: Exposure filter ──
    if args.behavior_path:
        print("\nStep 2: Filtering by exposed IIDs...")
        exposed_iids = load_exposed_iids(args.behavior_path)
        cid_str = np.array([str(cid) for cid in content_ids])
        mask = np.isin(cid_str, list(exposed_iids))
        embeddings = embeddings[mask]
        content_ids = content_ids[mask]
        print(f"  Exposed items: {len(embeddings):,} / {len(cid_str):,}")
    else:
        print("\nStep 2: Skipped (no behavior filter)")

    embed_tensor = torch.tensor(embeddings, dtype=torch.float32)
    n_features = embed_tensor.shape[1]
    n_items = len(content_ids)
    load_time = time.time() - t0
    print(f"  Done ({load_time:.1f}s)")

    # ── Step 3: Train tokenizer ──
    print(f"\nStep 3: Training tokenizer ({n_items:,} items)...")
    t1 = time.time()

    cfg = TOKENIZER_CONFIG
    fsq_levels = FSQ_LEVEL_CONFIGS[cfg['fsq_levels_key']]
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    model = ResKmeansFSQ(
        n_kmeans_clusters=cfg['num_clusters'],
        fsq_levels=fsq_levels,
        n_features=n_features,
        normalize_residuals=cfg['normalize_residuals'],
        num_gpus=num_gpus,
        fsq_projection=cfg['fsq_projection'],
        fsq_mlp_hidden=cfg['fsq_mlp_hidden'],
        fsq_epochs=cfg['fsq_epochs'],
    )
    model.train(embed_tensor, niter=cfg['niter'], nredo=cfg['nredo'])
    train_time = time.time() - t1
    print(f"  Tokenizer trained ({train_time:.1f}s)")

    # ── Step 4: Generate SIDs ──
    print("\nStep 4: Generating SID assignments...")
    t2 = time.time()
    semantic_ids = generate_semantic_ids_fsq(model, embed_tensor, cfg['normalize_residuals'])
    gen_time = time.time() - t2
    print(f"  Generated {len(semantic_ids):,} SIDs ({gen_time:.1f}s)")

    # Quick stats
    unique_sids = len(set(semantic_ids))
    collision = 1.0 - unique_sids / len(semantic_ids)
    print(f"  Unique SIDs: {unique_sids:,} / {len(semantic_ids):,} (collision={collision:.4f})")

    # ── Step 5: Save ──
    print(f"\nStep 5: Saving to {output_dir}")

    # 5a. Quantizer model
    model_path = os.path.join(output_dir, 'quantizer.pt')
    model.save(model_path)

    # 5b. SID assignments (content_id → SID mapping as dict)
    sid_dict = {str(cid): sid for cid, sid in zip(content_ids, semantic_ids)}
    sid_path = os.path.join(output_dir, 'semantic_ids.npy')
    np.save(sid_path, sid_dict)
    print(f"  Saved {len(sid_dict):,} SID assignments to {sid_path}")

    # 5c. Config metadata
    config = {
        **cfg,
        'model_key': model_key,
        'embedding_dim': n_features,
        'n_items': n_items,
        'n_unique_sids': unique_sids,
        'collision_rate': round(collision, 6),
        'train_time_seconds': round(train_time, 1),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config to {config_path}")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SID preprocessing complete! ({total_time:.1f}s)")
    print(f"{'='*60}")
    print(f"  Quantizer:     {model_path}")
    print(f"  SID cache:     {sid_path}")
    print(f"  Config:        {config_path}")
    print(f"\nUsage: python run.py hyperparam --sid_cache {output_dir} --run_ntp")


if __name__ == '__main__':
    main()
