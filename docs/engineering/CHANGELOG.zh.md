# Engineering Changelog

[English](CHANGELOG.md) | [中文](CHANGELOG.zh.md)

代码/基础设施变更记录，按时间倒序。实验结果见 `experiments/logs/`。

---

## 2026-04-30

### tokenizer/ 目录重构
- 新建 `tokenizer/` 目录，将 tokenizer 相关代码从 `model/` 和 `eval/` 迁移
- `model/rkmeans.py`、`model/fsq.py`、`model/rkmeans_fsq.py`、`eval/preprocess_sid.py` 保留为 shim（`from tokenizer.xxx import *`）

### GPU Pipeline 全面优化（tokenizer）
- **KMeans**：`faiss.Kmeans(gpu=True)` 换为 `DatasetAssignGPU` + `faiss.contrib.clustering.kmeans`
  - 数据全程在 GPU，无 CPU 中转
  - 1.1M × 1024，k=4096，niter=25：14.6s → **6.4s（2.3×）**
  - 原因：`faiss.Kmeans.train()` 内部有 `np.ascontiguousarray` 强制 CPU，即使传 CUDA tensor 也一样；`torch_utils` 只 patch Index 类，不 patch Kmeans/Clustering
  - Benchmark：`benchmarks/bench_faiss_kmeans.py`
- **Normalize**：chunked CPU bounce → 单次 `F.normalize(data_gpu)`
- **残差计算**：`assigned_centroids.cpu()` → 全 GPU 直接相减
- **FSQ MLP**：整体 `residuals.to(device)` + `torch.randperm(N, device=device)`，消除 per-batch 传输
- **KMeans cache load**：`map_location="cpu"` → `map_location=primary_device`
- **SID 构建**：三列统一一次 `.cpu()`
- `np.bincount` → `torch.bincount`，GPU path 完全移除 numpy

### KMeans Layer Cache
- 相同 `(n_kmeans_clusters, n_features, n_samples, niter, nredo)` 的 KMeans 结果缓存到 `experiments/sid_cache/_kmeans_cache/<hash>/`
- FSQ 不同的 variants 共享 KMeans，直接跳到 FSQ 训练

### EXP-049 基础设施
- `num_clusters` 支持 comma-string 格式（`"4096,2048"`），L1/L2 可设不同 cluster 数（MGMR）
- `config/config.py`：`EFS_BASE` 默认值从 cloud notebook 路径改为 `/mnt/workspace`
- `run_config.sh`：加 `LD_LIBRARY_PATH` export，修复 faiss-gpu 因找不到 libcuda.so 静默 fallback CPU 的问题
- `rkmeans.py`：移除 `use_gpu = self.gpu and self.n_features <= 2048` 的错误 CPU fallback（根因是 LD_LIBRARY_PATH 缺失，非数值问题）

---

## 2026-04-29

### run_exp.py variants 支持
- YAML 支持 `variants:` 列表，`base_config + shared_keys + variant_overrides` 自动展开
- `--only NAME` 断点续跑单个 variant
- `ntp_data_name` 优先取 YAML 显式值，fallback `sid_cache_name`
- phase 从 resolved config 读取（修复从 raw YAML 读时漏掉 `_base_tokenizer.yaml` 里 `phase: tokenizer` 的 bug）

### NTP 模型
- TO-RoPE：`use_rope` + `rope_dims` 支持 2-dim（时间+位置）和 3-dim（时间+位置+层级）
- `load_model_from_checkpoint`：strip `use_rope`，`rope_dims` dict → `RopeDimSpec` 反序列化

### 实验基础设施
- `run_config.sh`：使用 `gr` conda 环境（faiss-gpu、torch cu128）
- 新增 `experiments/logs/tokenizer/`、`ntp/`、`rl/` 阶段目录

---

## 2026-04-28 及更早

### GRPO/ECPO 工程修复
- SIDTrie 构建：iterate `.values()` 而非 `.keys()`
- reward std≈0 保护：`std<1e-6` group skip + `adv.clamp(-5,5)` + `log_rho.clamp(-10,10)`
- BehaviorReward prefix cascade fallback（L0 覆盖率 0.16% → ~24%）
- on-policy beam search（EXP-029，修复 off-policy ratio 爆炸）

### Side Features 全链路
- train-eval 一致性：beam search incremental path 正确 carry-forward features
- EXP-044B bug fix：`constrained_beam_search` 补传 `step_timestamp`；`eval_items` 构建循环不再过滤 `inject='torope'` 的 timestamps
