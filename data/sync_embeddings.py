"""同步本地 embedding cache 到 S3 备份。

用法:
    python run.py sync-embeddings                      # 同步所有模型
    python run.py sync-embeddings --model qwen3-vl-2b  # 只同步指定模型
    python run.py sync-embeddings --dry-run             # 预览不执行
"""

import argparse
import os
import subprocess
import sys

from config import EFS_EMBEDDING_CACHE, S3_EMBEDDING_CACHE_BACKUP


def main():
    parser = argparse.ArgumentParser(description='Sync embedding cache to S3')
    parser.add_argument('--model', type=str, default=None,
                        help='Only sync this model (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without uploading')
    parser.add_argument('--delete', action='store_true',
                        help='Delete S3 files not in local (mirror mode)')
    args = parser.parse_args()

    if args.model:
        local = f"{EFS_EMBEDDING_CACHE}/{args.model}"
        remote = f"{S3_EMBEDDING_CACHE_BACKUP}/{args.model}"
        if not os.path.isdir(local):
            print(f"Local dir not found: {local}")
            sys.exit(1)
        pairs = [(local, remote)]
    else:
        if not os.path.isdir(EFS_EMBEDDING_CACHE):
            print(f"Local cache not found: {EFS_EMBEDDING_CACHE}")
            sys.exit(1)
        pairs = []
        for name in sorted(os.listdir(EFS_EMBEDDING_CACHE)):
            d = f"{EFS_EMBEDDING_CACHE}/{name}"
            if os.path.isdir(d):
                pairs.append((d, f"{S3_EMBEDDING_CACHE_BACKUP}/{name}"))

    if not pairs:
        print("Nothing to sync.")
        return

    for local, remote in pairs:
        n_files = sum(1 for f in os.listdir(local) if f.endswith('.npy'))
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}{local}  →  {remote}  ({n_files} .npy files)")
        cmd = ['aws', 's3', 'sync', local, remote, '--exclude', '*', '--include', '*.npy', '--include', 'cached_ids.txt']
        if args.dry_run:
            cmd.append('--dryrun')
        if args.delete:
            cmd.append('--delete')
        subprocess.run(cmd, check=True)

    print("\nDone.")


if __name__ == '__main__':
    main()
