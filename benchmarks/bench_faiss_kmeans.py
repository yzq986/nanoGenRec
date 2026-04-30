"""Benchmark: faiss KMeans GPU methods.

Compares three approaches for KMeans training on 1.1M x 1024 float32 embeddings:
  1. faiss.Kmeans(gpu=True) + numpy input  (current approach — requires CPU transfer)
  2. swig_ptr direct                        (bypass numpy wrapper — segfaults on this faiss build)
  3. DatasetAssignGPU + contrib kmeans      (recommended — fully GPU, 2.3x faster)

Usage:
    python benchmarks/bench_faiss_kmeans.py [--d D] [--n N] [--k K] [--niter NITER]
"""

import argparse
import time

import faiss
import faiss.contrib.torch_utils  # noqa: F401 — patches Index classes
import numpy as np
import torch
from faiss.contrib.clustering import kmeans as faiss_kmeans_contrib
from faiss.contrib.torch.clustering import DatasetAssignGPU


def method1_faiss_kmeans_numpy(data_gpu, d, n, k, niter):
    """Current approach: faiss.Kmeans(gpu=True) with numpy input.

    Requires a full CPU copy of data before training.
    faiss then uploads it internally to GPU.
    """
    data_np = data_gpu.cpu().numpy().astype(np.float32)
    t0 = time.time()
    km = faiss.Kmeans(d, k, niter=niter, gpu=True, seed=42, verbose=False, nredo=1)
    km.train(data_np)
    elapsed = time.time() - t0
    centroids = torch.tensor(km.centroids, device='cuda:0')
    inertia = km.obj[-1]
    return centroids, elapsed, inertia


def method3_dataset_assign_gpu(data_gpu, d, n, k, niter):
    """Recommended: DatasetAssignGPU + faiss.contrib.clustering.kmeans.

    Data stays on GPU throughout — no CPU transfer.
    Note: nredo not implemented in pure-python kmeans; run multiple seeds manually if needed.
    """
    res = faiss.StandardGpuResources()
    dataset = DatasetAssignGPU(res, data_gpu)
    t0 = time.time()
    centroids = faiss_kmeans_contrib(k=k, data=dataset, niter=niter, seed=42, verbose=False)
    elapsed = time.time() - t0
    return centroids, elapsed, None  # inertia not directly exposed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=1024)
    parser.add_argument('--n', type=int, default=1_100_000)
    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--niter', type=int, default=25)
    args = parser.parse_args()

    d, n, k, niter = args.d, args.n, args.k, args.niter
    print(f'Benchmark: {n:,} x {d}  k={k}  niter={niter}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print()

    data_gpu = torch.randn(n, d, dtype=torch.float32, device='cuda:0').contiguous()

    print('--- Method 1: faiss.Kmeans(gpu=True) + numpy (CPU transfer) ---')
    c1, t1, inertia1 = method1_faiss_kmeans_numpy(data_gpu, d, n, k, niter)
    print(f'Time: {t1:.1f}s  Inertia: {inertia1:.2f}  Centroids: {c1.shape} on {c1.device}')
    del c1
    torch.cuda.empty_cache()
    print()

    print('--- Method 3: DatasetAssignGPU + contrib kmeans (fully GPU) ---')
    c3, t3, _ = method3_dataset_assign_gpu(data_gpu, d, n, k, niter)
    print(f'Time: {t3:.1f}s  Centroids: {c3.shape} on {c3.device}')
    del c3
    torch.cuda.empty_cache()
    print()

    print(f'Speedup (M3 vs M1): {t1/t3:.2f}x')


if __name__ == '__main__':
    main()
