#!/usr/bin/env bash
# ============================================================
# EXP-016 Prep: Extend SID cache to cover 01-25 ~ 03-31
#
# Reuses exp013 quantizer (4096×3 FSQ [2]×12 binary),
# incrementally predicts SIDs for items not yet in cache.
#
# Usage:
#   bash experiments/scripts/exp016_prepare_sid.sh
# ============================================================
set -euo pipefail

N_GPUS="${N_GPUS:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
SID_CACHE="experiments/sid_cache/exp013-4096x3-12d-binary"

echo "============================================================"
echo "EXP-016 Prep: Extend SID cache"
echo "  SID cache: ${SID_CACHE}"
echo "  Date range: 2026-01-25 ~ 2026-03-31"
echo "  GPUs: ${N_GPUS}"
echo "============================================================"

# ── Verify existing cache ──
if [ ! -f "${SID_CACHE}/quantizer.pt" ] || [ ! -f "${SID_CACHE}/semantic_ids.npy" ]; then
    echo "ERROR: exp013 SID cache not found at ${SID_CACHE}"
    echo "Run exp-013.sh first."
    exit 1
fi

echo ""
echo ">>> Before: SID cache item count"
python -c "
import numpy as np
sid = np.load('${SID_CACHE}/semantic_ids.npy', allow_pickle=True).item()
print(f'  Items in cache: {len(sid):,}')
"

# ── Incremental SID update ──
echo ""
echo ">>> Running incremental preprocess-sid (01-25 ~ 03-31)..."
if [ "${N_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${N_GPUS}" -m gr_demo.eval.preprocess_sid \
        --model qwen3-0.6b \
        --output_dir "${SID_CACHE}" \
        --incremental \
        --behavior_path auto \
        --date_start 2026-01-25 --date_end 2026-03-31
else
    python run.py preprocess-sid \
        --model qwen3-0.6b \
        --output_dir "${SID_CACHE}" \
        --incremental \
        --behavior_path auto \
        --date_start 2026-01-25 --date_end 2026-03-31
fi

# ── Verify ──
echo ""
echo ">>> After: SID cache stats"
python -c "
import numpy as np
sid = np.load('${SID_CACHE}/semantic_ids.npy', allow_pickle=True).item()
sids = list(sid.values())
unique = len(set(sids))
collision = 1.0 - unique / len(sids)
print(f'  Items in cache: {len(sid):,}')
print(f'  Unique SIDs:    {unique:,}')
print(f'  Collision rate:  {collision:.4f}')
"

echo ""
echo "============================================================"
echo "SID cache ready for EXP-016!"
echo "  Next: python experiments/scripts/analyze_data_distribution.py"
echo "  Then: bash experiments/scripts/exp-016.sh"
echo "============================================================"
