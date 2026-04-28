"""检查 S3_CONTENT_TEXT_EXPOSED 下 parquet 是否有 images 列。

用法:
    python experiments/scripts/check_parquet_images.py
    python experiments/scripts/check_parquet_images.py --date 2026-04-20
"""

import argparse
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

import s3fs
import pandas as pd
from config import S3_CONTENT_TEXT_EXPOSED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2026-04-20',
                        help='YYYY-MM-DD 检查哪天的 parquet')
    parser.add_argument('--n_rows', type=int, default=5,
                        help='sample 几行看内容')
    args = parser.parse_args()

    fs = s3fs.S3FileSystem()
    path_clean = S3_CONTENT_TEXT_EXPOSED.replace('s3://', '')
    files = sorted(fs.glob(f"{path_clean}/{args.date}/*.parquet"))

    if not files:
        print(f"No parquet found at {S3_CONTENT_TEXT_EXPOSED}/{args.date}")
        return

    print(f"Found {len(files)} parquet files for {args.date}")
    print(f"Reading first file: {files[0]}")

    with fs.open(files[0], 'rb') as f:
        df = pd.read_parquet(f)

    print(f"\n=== Schema ({len(df):,} rows) ===")
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].iloc[0] if len(df) > 0 else None
        sample_str = repr(sample)[:100]
        print(f"  {col:30s}  {dtype:20s}  sample: {sample_str}")

    print(f"\n=== Image column candidates ===")
    image_cols = [c for c in df.columns
                  if any(k in c.lower() for k in ['image', 'img', 'pic', 'photo', 'media'])]
    if image_cols:
        for col in image_cols:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null:,}/{len(df):,} non-null")
            print(f"  samples: {df[col].head(args.n_rows).tolist()}")
    else:
        print("  (none found — check schema above)")

    print(f"\n=== 'images' column present? ===")
    print(f"  {'YES' if 'images' in df.columns else 'NO'}")


if __name__ == '__main__':
    main()
