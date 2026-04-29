#!/bin/bash
set -euo pipefail

# EXP-045: FSQ Hidden Dim 经验公式 — 扫参拟合 collision_rate ∼ f(h, dim)
# Date: 2026-04-29
#
# 目标: 找到 "给定 embedding_dim，选多大 h 能让 collision_rate < 1%" 的经验公式，
#       避免每次换 embedding model 都要重新 sweep。
#
# 设计:
#   固定 fsq_levels=12d_4096, projection=mlp, epochs=50
#   扫 h × model:
#     0.6b (dim=1024):  h ∈ {32, 64*, 128, 256}
#     4b   (dim=2560):  h ∈ {64*, 128, 256, 512, 1024}
#     8b   (dim=4096):  h ∈ {64*, 128, 256, 512, 1024, 2048}
#   * = exp026 已有结果，直接复用，不重训
#
#   最终拟合 collision_rate = a * (h/dim)^b，
#   推导 h_min(dim) = dim * (target_cr/a)^(1/b)
#   结论写入 CLAUDE.md
#
# 不跑 NTP — collision_rate 作为 proxy metric
#
# 前置条件:
#   export EFS_BASE=/mnt/workspace
#   embedding cache 存在于 ${EFS_BASE}/embedding_cache/qwen3-{0.6b,4b,8b}/

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

FORCE=false
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
    esac
done

echo "=========================================="
echo "EXP-045: FSQ Hidden Dim Sweep"
echo "=========================================="
echo "  EFS_BASE: ${EFS_BASE}"
echo ""

# Sanity check
for model in 0.6b 4b 8b; do
    cache="${EFS_BASE}/embedding_cache/qwen3-${model}"
    if [ ! -f "${cache}/content_ids.npy" ]; then
        echo "ERROR: Embedding cache not found: ${cache}/content_ids.npy"
        exit 1
    fi
done

# ── Helper: run one preprocess-sid config ────────────────────
run_fsq() {
    local MODEL_KEY=$1   # qwen3-0.6b / qwen3-4b / qwen3-8b
    local H=$2
    local SID_DIR="experiments/sid_cache/exp045-${MODEL_KEY#qwen3-}-h${H}"

    if [ -f "${SID_DIR}/config.json" ] && [ "${FORCE}" != true ]; then
        python3 -c "
import json
c = json.load(open('${SID_DIR}/config.json'))
print(f'  [skip] {\"${MODEL_KEY}\":12s} h={\"${H}\":4s}  dim={c[\"embedding_dim\"]}  collision={c[\"collision_rate\"]:.4%}  ({c[\"n_items\"]:,} items)')
" 2>/dev/null || echo "  [skip] ${MODEL_KEY} h=${H} (already exists)"
        return
    fi

    echo ">>> ${MODEL_KEY} h=${H}..."
    python run.py preprocess-sid \
        --model "${MODEL_KEY}" \
        --output_dir "${SID_DIR}" \
        --fsq_mlp_hidden "${H}" \
        --fsq_levels 12d_4096 \
        --fsq_projection mlp \
        --behavior_path ""

    python3 -c "
import json
c = json.load(open('${SID_DIR}/config.json'))
print(f'  done: collision={c[\"collision_rate\"]:.4%}  n_unique={c[\"n_unique_sids\"]:,}  time={c[\"train_time_seconds\"]:.0f}s')
" 2>/dev/null || true
    echo ""
}

# ── Step 1: 0.6b sweep (skip h=64, already in exp026) ────────
echo "=== Qwen3-0.6b (dim=1024) ==="
run_fsq "qwen3-0.6b" 32
# h=64: 复用 exp026-0.6b-14d
run_fsq "qwen3-0.6b" 128
run_fsq "qwen3-0.6b" 256
echo ""

# ── Step 2: 4b sweep (skip h=64, already in exp026) ──────────
echo "=== Qwen3-4b (dim=2560) ==="
# h=64: 复用 exp026-4b-14d
run_fsq "qwen3-4b" 128
run_fsq "qwen3-4b" 256
run_fsq "qwen3-4b" 512
run_fsq "qwen3-4b" 1024
echo ""

# ── Step 3: 8b sweep (skip h=64, already in exp026) ──────────
echo "=== Qwen3-8b (dim=4096) ==="
# h=64: 复用 exp026-8b-14d
run_fsq "qwen3-8b" 128
run_fsq "qwen3-8b" 256
run_fsq "qwen3-8b" 512
run_fsq "qwen3-8b" 1024
run_fsq "qwen3-8b" 2048
echo ""

# ── Step 4: 收集结果 + 拟合公式 ──────────────────────────────
echo ">>> Fitting empirical formula..."
python3 - <<'PYEOF'
import json, os, sys
import numpy as np

# (model_key, dim, h, sid_dir)
CONFIGS = [
    # 0.6b
    ("qwen3-0.6b",  1024,   32, "experiments/sid_cache/exp045-0.6b-h32"),
    ("qwen3-0.6b",  1024,   64, "experiments/sid_cache/exp026-0.6b-14d"),   # reuse
    ("qwen3-0.6b",  1024,  128, "experiments/sid_cache/exp045-0.6b-h128"),
    ("qwen3-0.6b",  1024,  256, "experiments/sid_cache/exp045-0.6b-h256"),
    # 4b
    ("qwen3-4b",   2560,   64, "experiments/sid_cache/exp026-4b-14d"),      # reuse
    ("qwen3-4b",   2560,  128, "experiments/sid_cache/exp045-4b-h128"),
    ("qwen3-4b",   2560,  256, "experiments/sid_cache/exp045-4b-h256"),
    ("qwen3-4b",   2560,  512, "experiments/sid_cache/exp045-4b-h512"),
    ("qwen3-4b",   2560, 1024, "experiments/sid_cache/exp045-4b-h1024"),
    # 8b
    ("qwen3-8b",   4096,   64, "experiments/sid_cache/exp026-8b-14d"),      # reuse
    ("qwen3-8b",   4096,  128, "experiments/sid_cache/exp045-8b-h128"),
    ("qwen3-8b",   4096,  256, "experiments/sid_cache/exp045-8b-h256"),
    ("qwen3-8b",   4096,  512, "experiments/sid_cache/exp045-8b-h512"),
    ("qwen3-8b",   4096, 1024, "experiments/sid_cache/exp045-8b-h1024"),
    ("qwen3-8b",   4096, 2048, "experiments/sid_cache/exp045-8b-h2048"),
]

rows = []
print(f"  {'Model':<12} {'dim':>5} {'h':>5} {'h/dim':>7} {'collision':>11} {'n_items':>9}")
print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*7} {'-'*11} {'-'*9}")
for model, dim, h, path in CONFIGS:
    cfg_path = f"{path}/config.json"
    if not os.path.exists(cfg_path):
        print(f"  {model:<12} {dim:>5} {h:>5}   -- MISSING --")
        continue
    c = json.load(open(cfg_path))
    cr = c["collision_rate"]
    n  = c["n_items"]
    print(f"  {model:<12} {dim:>5} {h:>5} {h/dim:>7.4f} {cr:>11.4%} {n:>9,}")
    rows.append((dim, h, cr))

if len(rows) < 4:
    print("\n  Not enough data points to fit formula.")
    sys.exit(0)

# Fit: log(cr) = log(a) + b * log(h/dim)
# i.e. cr = a * (h/dim)^b
dims = np.array([r[0] for r in rows], dtype=float)
hs   = np.array([r[1] for r in rows], dtype=float)
crs  = np.array([r[2] for r in rows], dtype=float)

# Remove zeros (log undefined)
mask = crs > 0
x = np.log(hs[mask] / dims[mask])
y = np.log(crs[mask])

# Linear regression in log-log space
A = np.vstack([np.ones_like(x), x]).T
result = np.linalg.lstsq(A, y, rcond=None)
log_a, b = result[0]
a = np.exp(log_a)

r2 = 1 - np.sum((y - (log_a + b * x))**2) / np.sum((y - y.mean())**2)

print(f"\n  === Fit: collision_rate = {a:.4f} * (h/dim)^{b:.3f}  (R²={r2:.3f}) ===")
print()

# Derive h_min for target collision rates
print(f"  h_min to achieve target collision rate:")
print(f"  {'target_cr':>10}  {'0.6b (1024)':>12}  {'4b (2560)':>10}  {'8b (4096)':>10}  {'16b (8192?)':>12}")
print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*12}")
for target in [0.02, 0.01, 0.005, 0.001]:
    row = f"  {target:>10.1%}"
    for dim in [1024, 2560, 4096, 8192]:
        # cr = a*(h/dim)^b  =>  h = dim * (cr/a)^(1/b)
        if a > 0 and b < 0:
            h_min = dim * (target / a) ** (1.0 / b)
            row += f"  {int(h_min):>12,}" if dim == 1024 else f"  {int(h_min):>10,}"
        else:
            row += f"  {'N/A':>12}"
    print(row)

print()
print(f"  Recommendation: h = ceil(dim * ({a:.4f} * target_cr)^(1/{b:.3f}))")
print(f"  Rule of thumb (cr<1%): h ≈ {int(1024*(0.01/a)**(1/b))} for 0.6b, "
      f"{int(2560*(0.01/a)**(1/b))} for 4b, "
      f"{int(4096*(0.01/a)**(1/b))} for 8b")

# Save results JSON for CLAUDE.md update
import json as _json
result_data = {
    "formula": f"collision_rate = {a:.6f} * (h/dim)^{b:.6f}",
    "a": a, "b": b, "r2": r2,
    "h_min_1pct": {
        "dim_1024": int(1024*(0.01/a)**(1/b)),
        "dim_2560": int(2560*(0.01/a)**(1/b)),
        "dim_4096": int(4096*(0.01/a)**(1/b)),
    },
    "data_points": [{"dim": int(r[0]), "h": int(r[1]), "collision_rate": r[2]} for r in rows],
}
with open("experiments/fsq_formula.json", "w") as f:
    _json.dump(result_data, f, indent=2)
print(f"\n  Saved to experiments/fsq_formula.json")
PYEOF

# ── Step 5: Update CLAUDE.md ──────────────────────────────────
echo ""
echo ">>> Updating CLAUDE.md with FSQ formula..."
python3 - <<'PYEOF'
import json, os

fpath = "experiments/fsq_formula.json"
if not os.path.exists(fpath):
    print("  No formula file, skipping CLAUDE.md update.")
    exit(0)

f = json.load(open(fpath))
a, b, r2 = f["a"], f["b"], f["r2"]
h_1024 = f["h_min_1pct"]["dim_1024"]
h_2560 = f["h_min_1pct"]["dim_2560"]
h_4096 = f["h_min_1pct"]["dim_4096"]

section = f"""
## FSQ Hidden Dim 选取规则（EXP-045 实验结论）

经验公式（拟合自 {len(f["data_points"])} 个数据点，R²={r2:.3f}）：

```
collision_rate ≈ {a:.4f} × (h / embedding_dim) ^ {b:.3f}
```

**h_min（collision_rate < 1%）**：

| Embedding Model | dim | h_min |
|----------------|-----|-------|
| Qwen3-0.6B | 1024 | {h_1024} |
| Qwen3-4B   | 2560 | {h_2560} |
| Qwen3-8B   | 4096 | {h_4096} |

通用公式：`h_min = ceil(embedding_dim × ({a:.4f} × 0.01) ^ (1 / {b:.3f}))`

写实验脚本时直接查上表，不需要 sweep。
"""

claude_md = open("CLAUDE.md").read()

marker = "## FSQ Hidden Dim 选取规则"
if marker in claude_md:
    # Replace existing section
    start = claude_md.index(marker)
    # Find next ## section
    next_sec = claude_md.find("\n## ", start + 1)
    if next_sec == -1:
        claude_md = claude_md[:start] + section.strip() + "\n"
    else:
        claude_md = claude_md[:start] + section.strip() + "\n\n" + claude_md[next_sec+1:]
else:
    # Append before "## Code quality"
    insert_at = claude_md.find("## Code quality")
    if insert_at == -1:
        claude_md += "\n" + section.strip() + "\n"
    else:
        claude_md = claude_md[:insert_at] + section.strip() + "\n\n" + claude_md[insert_at:]

open("CLAUDE.md", "w").write(claude_md)
print("  CLAUDE.md updated.")
PYEOF

# ── Commit ────────────────────────────────────────────────────
echo ""
echo ">>> Committing results..."
git add experiments/sid_cache/ experiments/fsq_formula.json CLAUDE.md 2>/dev/null || true
git commit -m "EXP-045 complete: FSQ h sweep, formula collision_rate=a*(h/dim)^b" || echo "Nothing to commit"
./push.sh

echo ""
echo "EXP-045 complete!"
