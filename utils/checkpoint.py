"""Checkpoint directory utilities."""

import os
import time


def archive_if_exists(output_dir):
    """If output_dir has existing results, rename it with a timestamp suffix.

    Prevents silent overwrites — old results are always preserved.
    Returns the archive path if renamed, None if directory was empty/new.
    """
    meta_path = os.path.join(output_dir, 'train_meta.json')
    if not os.path.exists(meta_path):
        return None

    ts = time.strftime('%Y%m%d_%H%M%S')
    archive_dir = f"{output_dir}___{ts}"
    os.rename(output_dir, archive_dir)
    print(f"  Archived existing results: {output_dir} → {archive_dir}")
    return archive_dir
