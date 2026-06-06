# model/

[English](README.md) | [Chinese](README.zh.md)

Embedding wrappers, orchestration helpers, packaging utilities, and compatibility shims.

Tokenizer implementations were moved to [tokenizer/](../tokenizer/). The old tokenizer files in this directory remain only to preserve legacy imports.

## Files

| File | Purpose |
|------|---------|
| `embedders.py` | Qwen3 text and vision-language embedding wrappers. |
| `encode.py` | Batch encoding pipeline with caching, OOM retry, and optional remote backup. |
| `train.py` | End-to-end CLI orchestration for embedding, tokenizer training, and SID export. |
| `semantic_ids.py` | Semantic ID utility functions. |
| `pack.py` | Packaging entry point for deployment artifacts. |
| `rkmeans.py` | Compatibility shim to `tokenizer/rkmeans.py`. |
| `fsq.py` | Compatibility shim to `tokenizer/fsq.py`. |
| `rkmeans_fsq.py` | Compatibility shim to `tokenizer/rkmeans_fsq.py`. |

## Typical Flow

```text
raw item text/image
  -> embedders.py
  -> encode.py cache
  -> tokenizer/
  -> train.py orchestration
  -> pack.py deployment artifact
```

## Usage

```bash
# End-to-end tokenizer path
python run.py train --model qwen3-0.6b

# Reuse embeddings
python run.py train --model qwen3-0.6b --skip_embedding

# Package an artifact
python run.py pack --rkmeans_s3_path s3://example-bucket/path/to/rkmeans.pt
```

## Implementation Notes

- Pass `torch_dtype` explicitly in embedder wrappers. Silent fp32 fallback can cause large OOMs.
- Avoid `output_hidden_states=True` unless every layer output is required.
- OOM retry paths must delete live tensor references before calling `empty_cache()`.
- Do not reuse text-only caches for vision-language inputs; identical text with different images can produce different embeddings.

## Related Docs

- Tokenizer implementation: [tokenizer/README.md](../tokenizer/README.md)
- Tokenizer experiment results: [experiments/logs/tokenizer/README.md](../experiments/logs/tokenizer/README.md)
