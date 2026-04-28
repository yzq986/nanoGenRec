"""批量评测编排（原 demo_eval_all.py）。"""

import argparse
import os
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import torch

from metrics import INTRINSIC_METRICS, BEHAVIOR_METRICS, AVAILABLE_METRICS, ReportGenerator
from data.loaders import load_results_from_s3, load_model_from_s3
from eval.wrapper import RKMeansModelWrapper
from eval.behavior import BehaviorMetricsEvaluator
from eval.compare import load_model_results, generate_comparison_report

from config import S3_RKMEANS_BASE
from config import DEFAULT_DATE
from data.loaders import resolve_behavior_paths


# ============================================================
# Configuration
# ============================================================

S3_BASE = S3_RKMEANS_BASE
DATE = DEFAULT_DATE

OUTPUT_BASE = "experiments/eval"
COMPARISON_OUTPUT = "comparison_report"

ALL_MODELS = {
    "qwen3-0.6b": ("qwen3-0.6b", 1024),
    "qwen3-4b": ("qwen3-4b", 2560),
    "qwen3-8b": ("qwen3-8b", 4096),
    "qwen3-vl-2b": ("qwen3-vl-2b", 2048),
    # "qwen3-vl-8b": ("qwen3-vl-8b", 4096),
}


# ============================================================
# Evaluation
# ============================================================

def run_model_eval(
    model_name: str,
    behavior_data: Optional[Dict[str, np.ndarray]],
    sample_size: int = 0,
    device: str = "cuda",
    run_intrinsic: bool = True,
    run_behavior: bool = True,
    only_sid: bool = False,
    recall_beam_size: int = 50,
    eval_sample_size: int = 50000,
) -> Optional[Dict]:
    """评估单个模型的所有指标"""

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    emb_type = ALL_MODELS[model_name][0]
    results_path = f"{S3_BASE}/{model_name}/{DATE}/results_{emb_type}.parquet"
    model_path = f"{S3_BASE}/{model_name}/{DATE}/rkmeans_{emb_type}.pt"
    output_dir = f"{OUTPUT_BASE}/{date.today().isoformat()}_{model_name}"

    # Load results
    try:
        content_ids, embeddings, semantic_ids = load_results_from_s3(results_path, sample_size)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

    if embeddings is None:
        print("No embeddings found")
        return None

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Load model
    rkmeans_model = None
    try:
        model_data = load_model_from_s3(model_path)
        rkmeans_model = RKMeansModelWrapper(model_data, device=device)
    except Exception as e:
        print(f"Warning: Could not load model: {e}")

    # Filter behavior data for this model (不过滤，让 metric 自己处理)
    filtered_behavior = behavior_data
    if behavior_data is not None:
        print(f"Behavior data: {len(behavior_data['uid']):,} total interactions")

    # Create evaluator
    print("Creating evaluator...")
    evaluator = BehaviorMetricsEvaluator(
        embeddings=embeddings_tensor,
        content_ids=content_ids,
        semantic_ids=semantic_ids,
        model=rkmeans_model,
        behavior_data=filtered_behavior,
        device=device,
    )

    # Register metrics
    if run_intrinsic:
        print(f"Registering intrinsic metrics: {list(INTRINSIC_METRICS.keys())}")
        evaluator.register_intrinsic_metrics()

    if run_behavior and filtered_behavior is not None:
        if only_sid:
            # 只注册 semantic_id_prediction
            print("Registering: semantic_id_prediction only")
            evaluator.register_metrics(['semantic_id_prediction'])
        else:
            print(f"Registering behavior metrics: {list(BEHAVIOR_METRICS.keys())}")
            evaluator.register_behavior_metrics()

    # Run evaluation with extra kwargs for SID prediction
    sid_kwargs = {
        'semantic_id_prediction': {
            'device': device,
            'recall_beam_size': recall_beam_size,
            'eval_sample_size': eval_sample_size,
        }
    }
    evaluator.evaluate(metric_kwargs=sid_kwargs)

    # Generate reports
    metadata = {
        'n_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'n_unique_semantic_ids': len(set(semantic_ids)),
        'results_path': results_path,
        'model_path': model_path,
    }
    if filtered_behavior is not None:
        metadata['n_behavior_interactions'] = len(filtered_behavior['uid'])

    generator = ReportGenerator(
        model_name=model_name,
        output_dir=output_dir,
        metadata=metadata,
    )
    generator.add_results(evaluator.results)
    generator.generate_all()

    print(f"\nDone: {model_name}")
    return evaluator.results


def run_comparison():
    """生成对比报告"""
    print(f"\n{'='*60}")
    print("Generating Comparison Report")
    print(f"{'='*60}")

    results = load_model_results(OUTPUT_BASE)
    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} models: {list(results.keys())}")
    generate_comparison_report(results, COMPARISON_OUTPUT)


def resolve_exposure_files(date_start: str = None, date_end: str = None) -> List[str]:
    """Resolve S3 exposure parquet file paths for date range.

    Returns list of S3 paths (without s3:// prefix, for use with s3fs).
    """
    import s3fs

    from config import S3_USER_BEHAVIOR
    from config import DEFAULT_DATE_START, DEFAULT_DATE_END
    from datetime import datetime, timedelta

    s3_exposure_base = S3_USER_BEHAVIOR.rsplit("/", 1)[0] + "/feed_user_exposure"
    ds = date_start or DEFAULT_DATE_START
    de = date_end or DEFAULT_DATE_END
    start = datetime.strptime(ds, "%Y-%m-%d")
    end = datetime.strptime(de, "%Y-%m-%d")

    fs = s3fs.S3FileSystem()
    files = []
    d = start
    while d <= end:
        path_clean = f"{s3_exposure_base}/{d.strftime('%Y-%m-%d')}".replace('s3://', '')
        files.extend(fs.glob(f"{path_clean}/*.parquet"))
        d += timedelta(days=1)
    print(f"  Resolved {ds} ~ {de} → {len(files)} exposure files")
    return files


def load_exposure_neg_data(date_start: str = None, date_end: str = None,
                           local_path: str = None) -> Dict:
    """Load compact ENTP negative-sample parquet (PySpark 产出).

    Schema: uid STRING, iid STRING, first_ts LONG, neg_iids ARRAY<STRING>

    Args:
        local_path: 本地目录路径（优先），如 /mnt/workspace/gr-demo-exposure-neg/2026-03-01_2026-03-31
                    为 None 时走 S3（需要 s3fs 权限）。

    Returns:
        dict with uid, iid, first_ts (np arrays) and neg_iids (list of lists).
    """
    import glob as glob_module
    import pandas as pd

    from config import DEFAULT_DATE_START, DEFAULT_DATE_END

    ds = date_start or DEFAULT_DATE_START
    de = date_end or DEFAULT_DATE_END

    print(f"\n{'='*60}")
    print("Loading ENTP Negative Data")
    print(f"{'='*60}")

    if local_path is not None:
        # 本地路径模式
        files = sorted(glob_module.glob(f"{local_path}/*.parquet"))
        print(f"  Local path: {local_path}")
        print(f"  Found {len(files)} parquet files")
        if not files:
            raise FileNotFoundError(f"No parquet files found at {local_path}")
        dfs = []
        for i, f in enumerate(files):
            dfs.append(pd.read_parquet(f))
            if (i + 1) % 5 == 0:
                print(f"  Loaded {i + 1}/{len(files)}")
    else:
        # S3 模式（fallback）
        import s3fs
        from config import S3_USER_BEHAVIOR
        s3_neg_base = S3_USER_BEHAVIOR.rsplit("/", 1)[0] + "/feed_user_exposure_neg"
        neg_path = f"{s3_neg_base}/{ds}_{de}".replace('s3://', '')
        print(f"  S3 path: {neg_path}")
        fs = s3fs.S3FileSystem()
        files = fs.glob(f"{neg_path}/*.parquet")
        print(f"  Found {len(files)} parquet files")
        dfs = []
        for i, f in enumerate(files):
            with fs.open(f, 'rb') as fh:
                dfs.append(pd.read_parquet(fh))
            if (i + 1) % 5 == 0:
                print(f"  Loaded {i + 1}/{len(files)}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(df):,} positive interactions with negatives")

    first_ts = df['first_ts'].fillna(0).values.astype(np.int64)
    neg_iids = df['neg_iids'].tolist()  # list of lists of str

    return {
        'uid': df['uid'].values,
        'iid': df['iid'].values,
        'first_ts': first_ts,
        'neg_iids': neg_iids,
    }


def load_all_behavior_data(date_start: str = None, date_end: str = None,
                           behavior_path: str = "auto") -> Dict[str, np.ndarray]:
    """加载全部行为数据。支持本地路径和 S3。

    Args:
        date_start: 起始日期 (YYYY-MM-DD)
        date_end: 结束日期 (YYYY-MM-DD)
        behavior_path: 本地目录或 "auto"（走 S3）
    """
    import os
    import glob as glob_module
    import pandas as pd

    print(f"\n{'='*60}")
    print("Loading Behavior Data")
    print(f"{'='*60}")

    paths = resolve_behavior_paths(behavior_path, date_start=date_start, date_end=date_end)
    is_local = all(not p.startswith('s3://') for p in paths)

    files = []
    if is_local:
        for bp in paths:
            files.extend(sorted(glob_module.glob(os.path.join(bp, '*.parquet'))))
    else:
        import s3fs
        fs = s3fs.S3FileSystem()
        for bp in paths:
            path_clean = bp.replace('s3://', '')
            files.extend(fs.glob(f"{path_clean}/*.parquet"))

    print(f"Found {len(files)} files")

    dfs = []
    for i, f in enumerate(files):
        if is_local:
            dfs.append(pd.read_parquet(f))
        else:
            with fs.open(f, 'rb') as fh:
                dfs.append(pd.read_parquet(fh))
        if (i + 1) % 5 == 0:
            print(f"  Loaded {i + 1}/{len(files)}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(df):,} interactions")

    # 填充 NaN 的 first_ts 为 0
    first_ts = df['first_ts'].fillna(0).values.astype(np.int64)

    return {
        'uid': df['uid'].values,
        'iid': df['iid'].values,
        'action_bitmap': df['action_bitmap'].values.astype(np.int32),
        'first_ts': first_ts,
    }


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Complete Evaluation Pipeline')
    parser.add_argument('--models', type=str, nargs='+', default=None)
    parser.add_argument('--sample_size', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    # SID prediction settings
    parser.add_argument('--recall_beam_size', type=int, default=50,
                        help='Beam size for item recall in NTP eval (default: 50)')
    parser.add_argument('--eval_sample_size', type=int, default=50000,
                        help='Max eval samples for NTP (0=all, default: 50000)')

    # Quick mode
    parser.add_argument('--quick', action='store_true')

    # Phase control
    parser.add_argument('--only-intrinsic', action='store_true')
    parser.add_argument('--only-behavior', action='store_true')
    parser.add_argument('--skip-intrinsic', action='store_true')
    parser.add_argument('--skip-behavior', action='store_true')
    parser.add_argument('--skip-compare', action='store_true')
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--only-sid', action='store_true',
                        help='Only run semantic_id_prediction metric')

    return parser.parse_args()


def main():
    args = parse_args()

    # Quick mode
    if args.quick:
        args.sample_size = 50000
        args.eval_sample_size = 10000

    # Handle --only flags
    run_intrinsic = True
    run_behavior = True

    # --only-sid: 只运行 semantic_id_prediction
    only_sid = args.only_sid

    if args.only_intrinsic:
        run_behavior = False
    elif args.only_behavior:
        run_intrinsic = False
    elif only_sid:
        run_intrinsic = False
        run_behavior = True  # SID prediction 在 behavior metrics 里

    if args.skip_intrinsic:
        run_intrinsic = False
    if args.skip_behavior:
        run_behavior = False

    # Models
    models = args.models if args.models else list(ALL_MODELS.keys())
    models = [m for m in models if m in ALL_MODELS]

    print("=" * 60)
    print("Complete Evaluation Pipeline")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {models}")
    print(f"Sample size: {args.sample_size or 'all'}")
    print(f"Run intrinsic: {run_intrinsic}")
    print(f"Run behavior: {run_behavior}")
    print("=" * 60)

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(COMPARISON_OUTPUT, exist_ok=True)

    if args.compare_only:
        run_comparison()
        return

    # Load behavior data
    behavior_data = None
    if run_behavior:
        try:
            behavior_data = load_all_behavior_data()
        except Exception as e:
            print(f"Warning: Could not load behavior data: {e}")
            print("Skipping behavior metrics")
            run_behavior = False

    # Evaluate each model
    for model_name in models:
        try:
            run_model_eval(
                model_name=model_name,
                behavior_data=behavior_data,
                sample_size=args.sample_size,
                device=args.device,
                run_intrinsic=run_intrinsic,
                run_behavior=run_behavior,
                only_sid=only_sid,
                recall_beam_size=args.recall_beam_size,
                eval_sample_size=args.eval_sample_size,
            )
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Comparison
    if not args.skip_compare:
        run_comparison()

    print("\n" + "=" * 60)
    print("All Done!")
    print("=" * 60)
    print(f"Results: {OUTPUT_BASE}/")
    print(f"Comparison: {COMPARISON_OUTPUT}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
