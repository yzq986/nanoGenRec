"""预下载 HF 模型权重 (避免 torchrun 里 rank 0 阻塞其他 rank + NCCL barrier 超时)。

用法:
    python run.py download-model --model qwen3-vl-2b
    python run.py download-model --model qwen3-8b

会自动启用 hf_transfer (如已安装) 以加速大文件下载。
"""

import argparse
import os

from data.encode_distributed import DISTRIBUTED_MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(description='Pre-download HF model weights')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(DISTRIBUTED_MODEL_CONFIGS.keys()),
                        help='Model key (see DISTRIBUTED_MODEL_CONFIGS)')
    parser.add_argument('--hf_name', type=str, default=None,
                        help='Override HF repo id (default: from config)')
    args = parser.parse_args()

    hf_name = args.hf_name or DISTRIBUTED_MODEL_CONFIGS[args.model][0]

    # 尝试启用 hf_transfer (rust 后端, 快 & 更稳)
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
        print('[hf_transfer] enabled')
    except ImportError:
        print('[hf_transfer] not installed; falling back to python downloader')
        print('              pip install hf_transfer  # 推荐安装')

    from huggingface_hub import snapshot_download
    print(f'Downloading {hf_name} ...')
    path = snapshot_download(hf_name)
    print(f'Done: {path}')


if __name__ == '__main__':
    main()
