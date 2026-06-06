# tokenizer/

[English](README.md) | [Chinese](README.zh.md)

Semantic ID tokenizers for converting dense item embeddings into discrete item IDs.

The tokenizer is the representation layer of the project. It takes Qwen3 item embeddings, quantizes them into a compact 3-token Semantic ID, and writes SID caches consumed by the NTP recommender.

## What It Produces

```text
item embedding
  -> L2 normalization
  -> residual KMeans layer 1
  -> residual KMeans layer 2
  -> FSQ MLP residual quantizer
  -> Semantic ID: c1_c2_c3
```

The current recommended family is a 4096x3 binary `[2]x12` SID, with `num_clusters=8192` selected by EXP-049.

## Files

| File | Purpose |
|------|---------|
| `rkmeans.py` | GPU-native `FaissKMeansLayer` using `DatasetAssignGPU`. |
| `fsq.py` | FSQ layers, learned FSQ, and `FSQ_LEVEL_CONFIGS`. |
| `rkmeans_fsq.py` | `ResKmeansFSQ`, the 2xKMeans + 1xFSQ tokenizer. |
| `preprocess_sid.py` | CLI implementation for training tokenizers and writing SID caches. |

## Usage

```bash
# Train a SID cache directly
python run.py preprocess-sid \
    --model qwen3-0.6b \
    --output_dir experiments/sid_cache/my-exp \
    --num_clusters 8192 \
    --fsq_levels 12d_4096 \
    --fsq_mlp_hidden 128

# Preferred experiment path
python experiments/run_exp.py experiments/configs/exp-049.yaml --no-smoke --commit
```

## Implementation Notes

- KMeans assignment stays on GPU through `DatasetAssignGPU`.
- Residuals are computed on GPU as `data_gpu - centroids[assignments]`.
- KMeans results are cached under `experiments/sid_cache/_kmeans_cache/<hash>/` for reuse across FSQ variants.
- Avoid `faiss.Kmeans(gpu=True)` for this path; it can force CPU copies through numpy conversion.

## Data Contract

A SID cache should provide:

| Artifact | Meaning |
|----------|---------|
| `semantic_ids.npy` | Mapping from item ID to SID string. |
| tokenizer weights/config | Quantizer state used to reproduce assignment. |
| metadata | Model name, cluster counts, FSQ config, and date/data window. |

Downstream NTP preprocessing assumes every behavior item can be resolved to a SID. When building new caches, verify item coverage against the target behavior data window.

## Compatibility

Legacy imports from `model/rkmeans.py`, `model/fsq.py`, `model/rkmeans_fsq.py`, and `eval/preprocess_sid.py` are kept as shims. New code should import from `tokenizer/`.

## Results

Tokenizer experiments and current recommendations are tracked in [experiments/logs/tokenizer/README.md](../experiments/logs/tokenizer/README.md).
