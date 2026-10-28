#!/usr/bin/env bash
# Score UNREFINED base DAD ensemble (mean + median) for given horizons/K.
# Aggregates K base members (analysis/tH/dumps_members12/test/member_0..K-1) -> mean & median,
# then scores both with the same recipe as refined (fair CRPS over the K base members).
# usage: score_base_ensemble.sh <gpu> <H:K> [<H:K> ...]   e.g. 0 1:8 1:12 2:8 2:12
set -u
GPU="$1"; shift
PY=/home/user/miniconda3/envs/wind_power/bin/python
IO=/DATA/shinsung/inference_outputs
MET="$(cd "$(dirname "$0")" && pwd)/forecast_metrics_eval_fid500.py"   # 같은 폴더의 정본 채점기
for spec in "$@"; do
  IFS=: read -r H K <<<"$spec"
  MEM="$IO/analysis/t$H/dumps_members12/test"
  OUT="$IO/analysis/t$H/base_K$K"
  TGT="/DATA/shinsung/inference_outputs/gt/t${H}"
  mkdir -p "$OUT"
  echo "[baseK gpu$GPU] t+$H K=$K AGG $(date '+%F %T')"
  $PY "$IO/make_base_ens.py" "$MEM" "$OUT" "$K" >> "$OUT/agg.log" 2>&1 \
    || { echo "[baseK gpu$GPU] t+$H K=$K AGG FAILED"; continue; }
  # samples/member_0..K-1 -> base member dirs (fair CRPS over K)
  mkdir -p "$OUT/samples"
  for k in $(seq 0 $((K-1))); do ln -sfn "$MEM/member_$k" "$OUT/samples/member_$k"; done
  echo "[baseK gpu$GPU] t+$H K=$K MEAN-SCORE $(date '+%F %T')"
  $PY "$MET" --pred-dir "$OUT/prediction" --target-dir "$TGT" --outdir "$OUT/metrics" \
     --metrics mse_m11 ssim crps lpips fid --value-range m11 --device "cuda:$GPU" >> "$OUT/metrics.log" 2>&1 \
    || echo "[baseK gpu$GPU] t+$H K=$K MEAN-SCORE FAILED"
  echo "[baseK gpu$GPU] t+$H K=$K MEDIAN-SCORE $(date '+%F %T')"
  $PY "$MET" --pred-dir "$OUT/prediction_median" --target-dir "$TGT" --outdir "$OUT/metrics_median" \
     --metrics mse_m11 ssim lpips fid --value-range m11 --device "cuda:$GPU" >> "$OUT/metrics_median.log" 2>&1 \
    || echo "[baseK gpu$GPU] t+$H K=$K MEDIAN-SCORE FAILED"
  echo "[baseK gpu$GPU] t+$H K=$K DONE $(date '+%F %T')"
done
echo "[baseK gpu$GPU] ALL DONE $(date '+%F %T')"
