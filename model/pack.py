"""打包 model.tar.gz + model registry 上传。"""

import argparse
import json
import os
import shutil
import tarfile
import tempfile

from s3_utils import download_from_s3
from config import RAVEN_ENDPOINT, EFS_MODEL_CACHE, EFS_DEFAULT_OUTPUT


def download_qwen_model(model_name: str, local_dir: str):
    """Download Qwen model from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_name} from HuggingFace...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"  Done: {local_dir}")


def upload_model_to_raven(model_name: str, model_path: str, report: dict = None):
    """Upload model.tar.gz to model registry."""
    from raven.core.Provider import model registry
    from raven.core.data.Models import Application, TrainingStatus, ModelDataType

    endpoint = RAVEN_ENDPOINT
    client = model registry().with_endpoint(endpoint).create_raven(application=Application.MODEL)

    print(f"\nUploading to model registry: {model_name}")
    client.manuly_upload_training_result(
        model_name,
        model_path,
        model_data_type=ModelDataType.TAR_FILE,
        status=TrainingStatus.SUCCESS,
        evaluate_result=json.dumps(report or {})
    )
    print(f"  Upload complete!")
    return True


def pack_model_tarball(
    qwen_model_name: str,
    rkmeans_s3_path: str,
    output_path: str,
    qwen_local_cache: str = None,
):
    """Pack model.tar.gz for cloud notebook."""
    # 默认缓存目录
    default_cache = EFS_MODEL_CACHE

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nWorking directory: {tmpdir}")

        # 1. Qwen model - 优先用缓存
        qwen_dir = os.path.join(tmpdir, "qwen3_emb")
        cache_path = qwen_local_cache or default_cache

        if os.path.exists(cache_path) and os.listdir(cache_path):
            print(f"\n[Step 1/3] Using cached Qwen model: {cache_path}")
            shutil.copytree(cache_path, qwen_dir)
        else:
            print(f"\n[Step 1/3] Downloading Qwen model (first time, will cache)...")
            download_qwen_model(qwen_model_name, qwen_dir)
            # 保存到缓存
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if not os.path.exists(cache_path):
                shutil.copytree(qwen_dir, cache_path)
                print(f"  Cached to: {cache_path}")

        # 2. RKMeans model
        print(f"\n[Step 2/3] Downloading RKMeans model...")
        rkmeans_dir = os.path.join(tmpdir, "rkmeans")
        os.makedirs(rkmeans_dir, exist_ok=True)
        rkmeans_local = os.path.join(rkmeans_dir, "rkmeans.pt")
        download_from_s3(rkmeans_s3_path, rkmeans_local)

        # 3. Pack tar (不压缩，速度更快；cloud notebook 支持 .tar)
        print(f"\n[Step 3/3] Creating {output_path}...")
        # 如果是 .tar.gz 就压缩，否则不压缩
        mode = "w:gz" if output_path.endswith(".gz") else "w"
        with tarfile.open(output_path, mode) as tar:
            tar.add(qwen_dir, arcname="qwen3_emb")
            tar.add(rkmeans_dir, arcname="rkmeans")

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\nCreated: {output_path} ({size_mb:.1f} MB)")

        # Print contents
        print("\nContents:")
        with tarfile.open(output_path, "r:*") as tar:
            for member in tar.getmembers()[:20]:
                print(f"  {member.name}")
            if len(tar.getmembers()) > 20:
                print(f"  ... and {len(tar.getmembers()) - 20} more files")


def parse_args():
    parser = argparse.ArgumentParser(description="Pack model.tar.gz for cloud notebook")
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Qwen model name on HuggingFace",
    )
    parser.add_argument(
        "--rkmeans_s3_path",
        type=str,
        required=True,
        help="S3 path to rkmeans .pt file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=EFS_DEFAULT_OUTPUT,
        help="Output path for model.tar.gz",
    )
    parser.add_argument(
        "--qwen_local_cache",
        type=str,
        default=None,
        help="Local path to cached Qwen model (skip download)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="feed_content_embedding_v4",
        help="Model name for model registry upload",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to model registry after packing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Pack model.tar.gz for cloud notebook")
    print("=" * 60)
    print(f"Qwen model: {args.qwen_model}")
    print(f"RKMeans: {args.rkmeans_s3_path}")
    print(f"Output: {args.output_path}")
    print(f"Model name: {args.model_name}")
    print(f"Upload to model registry: {args.upload}")
    print("=" * 60)

    pack_model_tarball(
        qwen_model_name=args.qwen_model,
        rkmeans_s3_path=args.rkmeans_s3_path,
        output_path=args.output_path,
        qwen_local_cache=args.qwen_local_cache,
    )

    if args.upload:
        report = {
            "qwen_model": args.qwen_model,
            "rkmeans_s3_path": args.rkmeans_s3_path,
        }
        upload_model_to_raven(args.model_name, args.output_path, report)

    print("\n" + "=" * 60)
    print("Done!")
    if not args.upload:
        print(f"\nTo upload manually:")
        print(f"  from model.pack import upload_model_to_raven")
        print(f"  upload_model_to_raven('{args.model_name}', '{args.output_path}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
