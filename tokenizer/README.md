# tokenizer/ — Semantic ID Tokenizer

将 item embedding 量化为离散 3-token Semantic ID（SID），供 NTP 模型使用。

## 架构

```
item embedding (1024D / 2560D)
  → L2 normalize
  → KMeans L1  (faiss GPU, nc=4096~8192)
  → 残差
  → KMeans L2  (faiss GPU, nc=2048~8192)
  → 残差
  → FSQ MLP    (D → h → 12D binary, h=64~128)
  → SID: "c1_c2_c3"  (3-token, codebook=4096³)
```

## 文件

| 文件 | 说明 |
|------|------|
| `rkmeans.py` | `FaissKMeansLayer` — 单层 KMeans，GPU-native via `DatasetAssignGPU` |
| `fsq.py` | `FSQLayer`（PCA）、`LearnedFSQLayer`（MLP+STE）、`FSQ_LEVEL_CONFIGS` |
| `rkmeans_fsq.py` | `ResKmeansFSQ` — 2×KMeans + 1×FSQ 组合量化器 |
| `preprocess_sid.py` | CLI 入口：加载 embedding → 训练量化器 → 生成 SID → 保存 cache |

## GPU Pipeline

数据搬到 GPU 一次，全程不落 CPU：

| 环节 | 实现 |
|------|------|
| Normalize | `F.normalize(data_gpu)` |
| KMeans | `DatasetAssignGPU` + `faiss.contrib.clustering.kmeans` |
| 残差 | `data_gpu - centroids[assignments]`（GPU） |
| FSQ MLP | residuals 整体 pin GPU，`torch.randperm(N, device=device)` |
| SID 构建 | 三列一次 `.cpu()` |

**注意**：禁止用 `faiss.Kmeans(gpu=True)` — 内部有 `np.ascontiguousarray` 强制 CPU，
即使传 CUDA tensor 也会先搬回 CPU。必须用 `DatasetAssignGPU`。

### KMeans 性能（1.1M × 1024，k=4096，niter=25）

| 方式 | 时间 |
|------|------|
| `faiss.Kmeans` + numpy | 14.6s |
| `DatasetAssignGPU` | **6.4s（2.3×）** |

Benchmark：`python benchmarks/bench_faiss_kmeans.py`

## KMeans Cache

相同 `(n_kmeans_clusters, n_features, n_samples, niter, nredo, normalize_residuals)` 的
KMeans 结果缓存在 `experiments/sid_cache/_kmeans_cache/<hash>/`，
FSQ 不同的 variants 可直接跳过 KMeans 重训。

## 运行

```bash
# 单进程
python run.py preprocess-sid \
    --model qwen3-0.6b \
    --output_dir experiments/sid_cache/my-exp \
    --num_clusters 4096 \
    --fsq_levels 12d_4096 \
    --fsq_mlp_hidden 128

# 通过 run_exp.py（推荐）
python experiments/run_exp.py experiments/configs/exp-049.yaml --no-smoke --commit
```

## 向后兼容

原 `model/rkmeans.py`、`model/fsq.py`、`model/rkmeans_fsq.py`、`eval/preprocess_sid.py`
均保留为 shim，`from model.xxx import ...` 仍可用。

## 实验记录

见 `experiments/logs/tokenizer/README.md`。
