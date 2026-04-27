#!/bin/bash
# Upload key experiment artifacts to S3
# Usage: S3_BUCKET=s3://your-bucket bash experiments/scripts/upload_to_s3.sh
#
# Uploads:
#   experiments/ntp_checkpoints/exp020-hard-lam03/     (175M) — SFT baseline, GRPO起点
#   experiments/ntp_checkpoints/exp025-beam-passes/    (log only, probe.pt missing)
#   experiments/ntp_data/exp023-14d-features/          (223M) — 当前GRPO训练数据
#   experiments/sid_cache/exp013-4096x3-12d-binary/    (84M)  — SID tokenizer
#   experiments/ntp_checkpoints/exp028-*/              (训练中，完成后上传)
#
# NOTE: exp025-beam-passes 的 probe.pt 已丢失，只能上传 log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

S3_BUCKET="${S3_BUCKET:-}"
if [ -z "${S3_BUCKET}" ]; then
    echo "ERROR: S3_BUCKET not set. Usage: S3_BUCKET=s3://your-bucket bash $0"
    exit 1
fi

S3_PREFIX="${S3_BUCKET%/}/gr-demo"

echo "=========================================="
echo "Upload gr-demo artifacts to S3"
echo "  Destination: ${S3_PREFIX}"
echo "=========================================="
echo ""

# ── SID cache (84M) ──────────────────────────────────────────
echo ">>> [1/4] SID cache: exp013-4096x3-12d-binary (84M)"
aws s3 sync \
    experiments/sid_cache/exp013-4096x3-12d-binary/ \
    "${S3_PREFIX}/experiments/sid_cache/exp013-4096x3-12d-binary/" \
    --no-progress
echo "  Done."
echo ""

# ── NTP 训练数据 (223M) ──────────────────────────────────────
echo ">>> [2/4] NTP data: exp023-14d-features (223M)"
aws s3 sync \
    experiments/ntp_data/exp023-14d-features/ \
    "${S3_PREFIX}/experiments/ntp_data/exp023-14d-features/" \
    --no-progress
echo "  Done."
echo ""

# ── SFT checkpoint exp020 (175M) ─────────────────────────────
echo ">>> [3/4] SFT checkpoint: exp020-hard-lam03 (175M)"
aws s3 sync \
    experiments/ntp_checkpoints/exp020-hard-lam03/ \
    "${S3_PREFIX}/experiments/ntp_checkpoints/exp020-hard-lam03/" \
    --no-progress
echo "  Done."
echo ""

# ── exp025-beam-passes (log only, probe.pt missing) ──────────
echo ">>> [4/4] exp025-beam-passes (WARNING: probe.pt missing, log only)"
aws s3 sync \
    experiments/ntp_checkpoints/exp025-beam-passes/ \
    "${S3_PREFIX}/experiments/ntp_checkpoints/exp025-beam-passes/" \
    --no-progress
echo "  Done (log only)."
echo ""

# ── Optional: upload completed exp028/029/030 if they exist ──
for EXP in exp028-ecpo-weighted-w003-r100 exp029-ecpo-onpolicy-w003-r100 \
           exp030-a2po-nll-hepo-w003-r100 exp030-a2po-only-w003-r100; do
    CKPT="experiments/ntp_checkpoints/${EXP}"
    if [ -f "${CKPT}/probe.pt" ]; then
        echo ">>> Uploading completed checkpoint: ${EXP}"
        aws s3 sync \
            "${CKPT}/" \
            "${S3_PREFIX}/experiments/ntp_checkpoints/${EXP}/" \
            --no-progress
        echo "  Done."
        echo ""
    fi
done

echo "=========================================="
echo "Upload complete!"
echo ""
echo "Summary of what's uploaded:"
aws s3 ls "${S3_PREFIX}/experiments/" --recursive --human-readable | grep -E "\.pt$|\.npz$|\.npy$|\.json$" | awk '{print $3, $4, $5}' | sort -h
echo "=========================================="
