# Engineering Changelog

[English](CHANGELOG.md) | [Chinese](CHANGELOG.zh.md)

Code and infrastructure changes in reverse chronological order. Experiment results live in `experiments/logs/`.

## 2026-04-30

### `tokenizer/` Directory Refactor

- Created `tokenizer/` and moved tokenizer-related code out of `model/` and `eval/`.
- Kept `model/rkmeans.py`, `model/fsq.py`, `model/rkmeans_fsq.py`, and `eval/preprocess_sid.py` as compatibility shims using `from tokenizer.xxx import *`.

### Full GPU Pipeline Optimization for Tokenizers

- **KMeans**: replaced `faiss.Kmeans(gpu=True)` with `DatasetAssignGPU` plus `faiss.contrib.clustering.kmeans`.
  - Data stays on GPU throughout the path.
  - 1.1M x 1024, k=4096, niter=25: 14.6s -> **6.4s (2.3x)**.
  - Root cause: `faiss.Kmeans.train()` internally calls `np.ascontiguousarray`, forcing CPU transfer even for CUDA tensors. `torch_utils` patches Index classes, not Kmeans/Clustering.
  - Benchmark: `benchmarks/bench_faiss_kmeans.py`.
- **Normalize**: replaced chunked CPU bounce with one `F.normalize(data_gpu)` call.
- **Residual computation**: replaced `assigned_centroids.cpu()` with direct GPU subtraction.
- **FSQ MLP**: moved full residual tensors to device and used `torch.randperm(N, device=device)`, removing per-batch transfer.
- **KMeans cache loading**: changed `map_location="cpu"` to `map_location=primary_device`.
- **SID construction**: moved all three columns to CPU in one unified step.
- Replaced `np.bincount` with `torch.bincount`; the GPU path no longer depends on NumPy.

### KMeans Layer Cache

- Cache KMeans results for identical `(n_kmeans_clusters, n_features, n_samples, niter, nredo)` under `experiments/sid_cache/_kmeans_cache/<hash>/`.
- FSQ variants can share the same KMeans result and jump directly to FSQ training.

### EXP-049 Infrastructure

- `num_clusters` now supports comma-separated strings such as `"4096,2048"`, allowing different L1/L2 cluster counts for MGMR.
- `config.py`: changed the default `EFS_BASE` to `/mnt/workspace`.
- `run_config.sh`: exports `LD_LIBRARY_PATH`, fixing silent CPU fallback in faiss-gpu when `libcuda.so` was missing.
- `rkmeans.py`: removed the incorrect `use_gpu = self.gpu and self.n_features <= 2048` CPU fallback. The real cause was missing `LD_LIBRARY_PATH`, not numerical instability.

## 2026-04-29

### `run_exp.py` Variant Support

- YAML configs now support a `variants:` list; `base_config + shared_keys + variant_overrides` expands automatically.
- Added `--only NAME` for resuming or running one variant.
- `ntp_data_name` now prefers the explicit YAML value, falling back to `sid_cache_name`.
- Fixed phase detection to read from resolved config instead of raw YAML, so `_base_tokenizer.yaml` can supply `phase: tokenizer`.

### NTP Model

- TO-RoPE: `use_rope` and `rope_dims` now support 2D time+position and 3D time+position+level variants.
- `load_model_from_checkpoint`: strips `use_rope` and deserializes `rope_dims` dicts into `RopeDimSpec`.

### Experiment Infrastructure

- `run_config.sh`: uses the `gr` conda environment with faiss-gpu and torch cu128.
- Added stage directories under `experiments/logs/tokenizer/`, `experiments/logs/ntp/`, and `experiments/logs/rl/`.

## 2026-04-28 and Earlier

### GRPO/ECPO Engineering Fixes

- SIDTrie construction: iterate `.values()` instead of `.keys()`.
- Reward std near zero protection: `std<1e-6` group skip, `adv.clamp(-5,5)`, and `log_rho.clamp(-10,10)`.
- BehaviorReward prefix cascade fallback, increasing L0 coverage from 0.16% to about 24%.
- On-policy beam search for EXP-029, fixing off-policy ratio explosion.

### Side Features Pipeline

- Train/eval consistency: the beam-search incremental path now carries forward features correctly.
- EXP-044B bug fix: `constrained_beam_search` passes `step_timestamp`; the `eval_items` construction loop no longer filters out timestamps with `inject='torope'`.
