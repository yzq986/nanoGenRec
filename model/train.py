"""训练主流程 + CLI 参数。"""

import argparse
import os
from typing import List

import numpy as np
import torch

from gr_demo.config import (
    MODEL_CONFIGS, Config, EFS_EMBEDDING_CACHE, DEFAULT_DATE,
)
from gr_demo.s3_utils import upload_to_s3
from gr_demo.data.loaders import (
    load_text_from_s3, load_old_embeddings_from_s3,
    load_exposed_iids, export_results_to_s3, load_model_from_s3,
)
from gr_demo.model.embedders import Qwen3VLEmbedder, Qwen3TextEmbedder
from gr_demo.model.rkmeans import ResidualQuantizationMultiGPU
from gr_demo.model.semantic_ids import generate_semantic_ids
from gr_demo.model.encode import encode_with_qwen3, encode_with_qwen3_text


def _run_intrinsic_eval(
    embeddings_tensor: torch.Tensor,
    semantic_ids: List[str],
    rkmeans_model: ResidualQuantizationMultiGPU,
    config: Config,
):
    """运行 intrinsic metrics (不需要 behavior 数据)"""
    from gr_demo.metrics import INTRINSIC_METRICS
    from gr_demo.eval.wrapper import RKMeansModelWrapper

    # 构造 RKMeansModelWrapper 兼容的 model_data dict
    model_data = {
        'centroids_list': rkmeans_model.get_centroids_list(),
        'normalize_residuals': config.NORMALIZE_RESIDUALS,
        'n_layers': rkmeans_model.n_layers,
        'n_clusters': rkmeans_model.n_clusters,
        'n_features': rkmeans_model.n_features,
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_wrapper = RKMeansModelWrapper(model_data, device=device)

    results = {}
    for name, MetricClass in INTRINSIC_METRICS.items():
        print(f"\n  Computing {name}...")
        metric = MetricClass()
        try:
            result = metric.compute(
                embeddings=embeddings_tensor,
                model=model_wrapper,
                semantic_ids=semantic_ids,
                normalize_residuals=config.NORMALIZE_RESIDUALS,
            )
            results[name] = result
            print(f"    {name}: {result.value:.4f} ({result.status})")
            if result.layer_values:
                for i, lv in enumerate(result.layer_values):
                    print(f"      depth {i+1}: {lv:.4f}")
            for key in ('depth_stats',):
                if key in result.details:
                    for d in result.details[key]:
                        depth = d.get('depth', '?')
                        parts = [f"{k}={v}" for k, v in d.items() if k != 'depth']
                        print(f"      depth {depth}: {', '.join(str(p) for p in parts)}")
            for key in ('n_unique_sids',):
                if key in result.details:
                    print(f"      {key}: {result.details[key]}")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    # 打印汇总表
    print(f"\n{'='*60}")
    print("Intrinsic Metrics Summary")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Value':>10} {'Status':>12}")
    print(f"  {'-'*52}")
    for name, result in results.items():
        print(f"  {name:<30} {result.value:>10.4f} {result.status:>12}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description='RKMeans Training with Qwen3 Embedding')
    parser.add_argument('--model', type=str, default='qwen3-vl-8b',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Embedding model: qwen3-vl-8b, qwen3-vl-2b, qwen3-8b, qwen3-2b')
    parser.add_argument('--input_path', type=str, default=Config.INPUT_PATH,
                        help='S3 path to input text parquet files')
    parser.add_argument('--input_path_old', type=str, default=Config.INPUT_PATH_OLD_EMB,
                        help='S3 path to old Sentence-BERT embeddings')
    parser.add_argument('--output_path', type=str, default=None,
                        help='S3 path for output (auto-generated if not specified)')
    parser.add_argument('--num_layers', type=int, default=Config.NUM_LAYERS)
    parser.add_argument('--num_clusters', type=int, default=Config.NUM_CLUSTERS)
    parser.add_argument('--niter', type=int, default=Config.NITER,
                        help='FAISS KMeans iterations per layer (default: 25)')
    parser.add_argument('--nredo', type=int, default=Config.NREDO,
                        help='FAISS KMeans restarts, keep best (default: 1)')
    parser.add_argument('--embedding_batch_size', type=int, default=None,
                        help='Embedding batch size，不指定则根据模型自动选择')
    parser.add_argument('--max_images', type=int, default=0,
                        help='Max images per sample, 0 means text-only (ignored for text-only models)')
    parser.add_argument('--force', action='store_true',
                        help='强制重新生成 embedding，忽略缓存')
    parser.add_argument('--use_old_embedding', action='store_true',
                        help='Use old Sentence-BERT embeddings instead of Qwen3')
    parser.add_argument('--max_partitions', type=int, default=0,
                        help='只读取前 N 个 partition，0 表示读取全部')
    parser.add_argument('--skip_embedding', action='store_true',
                        help='跳过 embedding 生成，直接从缓存加载，只跑 rkmeans + SID 生成')
    parser.add_argument('--behavior_path', type=str, default=None,
                        help='S3 path to behavior parquet，用曝光 iid 过滤训练集')
    parser.add_argument('--eval_intrinsic', action='store_true',
                        help='生成 SID 后自动跑 intrinsic metrics 评估')
    parser.add_argument('--skip_rkmeans', action='store_true',
                        help='跳过 rkmeans 训练，从 S3 加载已有模型，只跑 SID 生成 + eval')
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()

    # 解析模型配置
    model_name, embedding_dim, is_multimodal, default_batch_size = MODEL_CONFIGS[args.model]

    # 如果未指定 embedding_batch_size，使用模型默认值
    if args.embedding_batch_size is None:
        args.embedding_batch_size = default_batch_size

    # 自动生成输出路径
    if args.output_path is None:
        args.output_path = f"{config.OUTPUT_PATH_BASE}/{args.model}/{DEFAULT_DATE}"

    # 缓存路径按模型区分
    embedding_cache_dir = f'{EFS_EMBEDDING_CACHE}/{args.model}'
    os.makedirs(embedding_cache_dir, exist_ok=True)
    embedding_cache_path = f'{embedding_cache_dir}/embeddings.npy'
    content_ids_cache_path = f'{embedding_cache_dir}/content_ids.npy'

    # 根据 embedding 类型设置输出路径后缀
    embedding_type = "sentence_bert" if args.use_old_embedding else args.model
    model_filename = f"rkmeans_{embedding_type}.pt"

    # 非多模态模型强制 max_images=0
    if not is_multimodal:
        args.max_images = 0

    print("="*60)
    if args.use_old_embedding:
        print("RKMeans Training with Sentence-BERT (384d)")
    else:
        modal_type = "图文多模态" if is_multimodal else "纯文本"
        print(f"RKMeans Training with {model_name} ({embedding_dim}d, {modal_type})")
    print("="*60)
    print(f"Model: {args.model} -> {model_name}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Multimodal: {is_multimodal}, max_images: {args.max_images}")
    print(f"Embedding batch_size: {args.embedding_batch_size}")
    print(f"Input: {args.input_path_old if args.use_old_embedding else args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Cache: {embedding_cache_dir}")
    print(f"RKMeans: {args.num_layers} layers x {args.num_clusters} clusters, niter={args.niter}, nredo={args.nredo}")
    print(f"Max partitions: {args.max_partitions}" if args.max_partitions > 0 else "Max partitions: all")
    print(f"Skip embedding: {args.skip_embedding}")
    print(f"Skip rkmeans: {args.skip_rkmeans}")
    print(f"Behavior filter: {args.behavior_path or 'None (use all IIDs)'}")
    print(f"Eval intrinsic: {args.eval_intrinsic}")
    print(f"Device: {config.DEVICE}, GPUs: {config.NUM_GPUS}")
    print("="*60)

    # Step 1 & 2: Load embeddings
    if args.skip_embedding:
        # --skip_embedding: 直接从缓存加载，跳过 embedding 生成
        print(f"\n{'='*60}")
        print("Loading embeddings from cache (--skip_embedding)...")
        print(f"{'='*60}")

        # 优先读 incremental_cache.npy (encode_multiprocess.py 产出的 dict 格式)
        incremental_cache_path = f'{embedding_cache_dir}/incremental_cache.npy'
        if os.path.exists(incremental_cache_path):
            cache_dict = np.load(incremental_cache_path, allow_pickle=True).item()
            content_ids = np.array(list(cache_dict.keys()))
            embeddings = np.array(list(cache_dict.values()), dtype=np.float32)
            print(f"Loaded {len(content_ids):,} embeddings from incremental_cache, dim={embeddings.shape[1]}")
        elif os.path.exists(embedding_cache_path) and os.path.exists(content_ids_cache_path):
            embeddings = np.load(embedding_cache_path, allow_pickle=True)
            content_ids = np.load(content_ids_cache_path, allow_pickle=True)
            print(f"Loaded {len(content_ids):,} embeddings from cache, dim={embeddings.shape[1]}")
        else:
            print(f"ERROR: Cache not found at {embedding_cache_dir}")
            print(f"  Run encode_multiprocess.py first to generate embeddings")
            return
    elif args.use_old_embedding:
        # 使用旧的 Sentence-BERT embedding
        print(f"\n{'='*60}")
        print("Loading OLD Sentence-BERT embeddings...")
        print(f"{'='*60}")
        content_ids, embeddings = load_old_embeddings_from_s3(args.input_path_old, max_partitions=args.max_partitions)
    else:
        # 生成新的 Qwen3 embedding (增量缓存模式)
        content_ids, texts, images_list, languages = load_text_from_s3(args.input_path, max_partitions=args.max_partitions)
        _ = languages  # unused

        print(f"\n{'='*60}")
        modal_type = "图文多模态" if is_multimodal else "纯文本"
        print(f"Generating {model_name} ({modal_type}) with incremental cache...")
        print(f"{'='*60}")

        if is_multimodal:
            # 使用 Qwen3VLEmbedder (图文多模态)
            embedder = Qwen3VLEmbedder(
                model_name_or_path=model_name,
                torch_dtype=torch.bfloat16,
            )
            content_ids, embeddings = encode_with_qwen3(
                embedder,
                content_ids,
                texts,
                images_list=images_list,
                batch_size=args.embedding_batch_size,
                max_images=args.max_images,
                cache_dir=embedding_cache_dir,
            )
        else:
            # 使用 Qwen3TextEmbedder (纯文本)
            embedder = Qwen3TextEmbedder(
                model_name_or_path=model_name,
                torch_dtype=torch.bfloat16,
            )
            content_ids, embeddings = encode_with_qwen3_text(
                embedder,
                content_ids,
                texts,
                batch_size=args.embedding_batch_size,
                cache_dir=embedding_cache_dir,
            )
        print(f"Embeddings shape: {embeddings.shape}")

    # Step 2.5: 用曝光 iid 过滤训练集（只保留被曝光过的 content）
    # 全量 embedding 保留用于最后的 SID 生成（所有 content 都需要 SID）
    all_content_ids = content_ids
    all_embeddings = embeddings

    if args.behavior_path:
        print(f"\n{'='*60}")
        print("Filtering by exposed IIDs from behavior data...")
        print(f"{'='*60}")
        exposed_iids = load_exposed_iids(args.behavior_path)
        print(f"Exposed IIDs: {len(exposed_iids):,}")

        # content_ids 转成 str 集合做交集
        cid_str = np.array([str(cid) for cid in content_ids])
        mask = np.isin(cid_str, list(exposed_iids))
        train_content_ids = content_ids[mask]
        train_embeddings = embeddings[mask]
        print(f"Before filter: {len(content_ids):,}, After filter: {len(train_content_ids):,} "
              f"({len(train_content_ids)/len(content_ids):.1%})")
    else:
        train_content_ids = content_ids
        train_embeddings = embeddings

    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32)

    # Step 3: Train or load RKMeans
    if args.skip_rkmeans:
        # 从 S3 加载已有模型
        print(f"\n{'='*60}")
        print("Loading RKMeans from S3 (--skip_rkmeans)...")
        print(f"{'='*60}")
        model_s3_path = f"{args.output_path}/{model_filename}"
        model_data = load_model_from_s3(model_s3_path)
        # 用 ResidualQuantizationMultiGPU 包装，保持接口一致
        rkmeans_model = ResidualQuantizationMultiGPU(
            n_layers=model_data['n_layers'],
            n_clusters=model_data['n_clusters'],
            n_features=model_data.get('n_features', model_data['centroids_list'][0].shape[1]),
            normalize_residuals=model_data.get('normalize_residuals', True),
            num_gpus=config.NUM_GPUS,
        )
        for i, centroids in enumerate(model_data['centroids_list']):
            rkmeans_model.kmeans_layers[i].centroids = centroids.to(rkmeans_model.primary_device)
    else:
        # 训练 RKMeans（只用曝光过的 content 训练）
        print(f"\n{'='*60}")
        print("Training RKMeans...")
        print(f"{'='*60}")

        rkmeans_model = ResidualQuantizationMultiGPU(
            n_layers=args.num_layers,
            n_clusters=args.num_clusters,
            n_features=train_embeddings.shape[1],
            normalize_residuals=config.NORMALIZE_RESIDUALS,
            num_gpus=config.NUM_GPUS,
        )

        rkmeans_model.train(
            train_embeddings_tensor,
            niter=args.niter,
            nredo=args.nredo,
        )

        # Save model
        local_path = f'/tmp/{model_filename}'
        rkmeans_model.save(local_path)
        upload_to_s3(local_path, f"{args.output_path}/{model_filename}")

    # Step 5: 对所有 content 生成 semantic_id（不只是训练集）
    print(f"\n{'='*60}")
    print("Generating semantic_ids and exporting to S3...")
    print(f"{'='*60}")

    all_embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float32)
    semantic_ids = generate_semantic_ids(rkmeans_model, all_embeddings_tensor, config.NORMALIZE_RESIDUALS)
    print(f"Generated {len(semantic_ids):,} semantic_ids")

    # 导出为 parquet
    export_results_to_s3(
        content_ids=all_content_ids,
        semantic_ids=semantic_ids,
        embeddings=all_embeddings,
        output_path=args.output_path,
        embedding_type=embedding_type
    )

    # Step 6: 可选的 intrinsic metrics 评估（在曝光 item 上跑）
    if args.eval_intrinsic:
        print(f"\n{'='*60}")
        print("Running Intrinsic Metrics Evaluation (on exposed items)...")
        print(f"{'='*60}")

        if args.behavior_path:
            # 用曝光子集的 embedding + SID 做评估
            eval_embeddings = train_embeddings_tensor
            # 从全量 semantic_ids 中按 mask 取出曝光子集的 SID
            cid_str = np.array([str(cid) for cid in all_content_ids])
            eval_mask = np.isin(cid_str, list(exposed_iids))
            eval_sids = [sid for sid, m in zip(semantic_ids, eval_mask) if m]
            print(f"  Eval on {len(eval_sids):,} exposed items (out of {len(semantic_ids):,} total)")
        else:
            eval_embeddings = all_embeddings_tensor
            eval_sids = semantic_ids

        _run_intrinsic_eval(eval_embeddings, eval_sids, rkmeans_model, config)

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Embedding type: {embedding_type}")
    print(f"Total samples: {len(all_content_ids):,}")
    if args.behavior_path:
        print(f"RKMeans training samples (exposed): {len(train_content_ids):,}")
    print(f"Embedding dim: {all_embeddings.shape[1]}")
    print(f"RKMeans: {args.num_layers} layers x {args.num_clusters} clusters (FAISS)")
    print(f"Model path: {args.output_path}/{model_filename}")
    print(f"Results path: {args.output_path}/results_{embedding_type}.parquet")
    print("="*60)


if __name__ == "__main__":
    main()
