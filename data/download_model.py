"""预下载 HF 模型权重 (避免 torchrun 里 rank 0 阻塞其他 rank + NCCL barrier 超时)。

用法:
    python run.py download-model --model qwen3-vl-2b
    python run.py download-model --model qwen3-8b

hf_transfer 默认禁用 (某些环境下会卡死)，使用普通 HTTP 下载。
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

    # 禁用 hf_transfer (rust 后端在某些环境下会卡死)
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

    from huggingface_hub import snapshot_download
    print(f'Downloading {hf_name} ...')
    path = snapshot_download(hf_name)
    print(f'Done: {path}')


if __name__ == '__main__':
    main()
