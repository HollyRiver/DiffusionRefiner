#!/usr/bin/env bash
# Per horizon: compute MEDIAN ensemble from refined members, then score MEAN and MEDIAN
# point-predictions with the same recipe. CRPS (member-based) is identical for mean/median,
# so it's computed only in the mean pass. Idempotent: mean pass skipped if already scored.
# usage: score_ensembles.sh <gpu> <8mem|12mem> <H>...
set -u
GPU="$1"; CONF="$2"; shift 2
PY=/home/user/miniconda3/envs/wind_power/bin/python
IO=/DATA/shinsung/inference_outputs
MET=/DATA/shinsung/forecast_metrics_eval_fid500.py
for H in "$@"; do
  root="$IO/analysis/t$H"
  if [ "$CONF" = 8mem ]; then WORK="$root/work30_inf_best"; else WORK="$root/work_valm0_12_inf"; fi
  OUT="$WORK/outputs_t${H}_member"
  TGT="/DATA/shinsung/lon_refiner/ablation_vol_gOFF_lON_t${H}/test_outputs/target"
  if [ ! -d "$OUT/prediction" ]; then echo "[score $CONF gpu$GPU] t+$H MISSING $OUT"; continue; fi
  mkdir -p "$OUT/samples"
  for md in "$OUT"/refined_member_*; do
    [ -d "$md" ] && ln -sfn "../$(basename "$md")" "$OUT/samples/member_${md##*refined_member_}"
  done
  npred=$(ls "$OUT/prediction"/*.pt 2>/dev/null | wc -l)
  nmed=$(ls "$OUT/prediction_median"/*.pt 2>/dev/null | wc -l)
  if [ "$nmed" -ge "$npred" ] && [ "$npred" -gt 0 ]; then
    echo "[score $CONF gpu$GPU] t+$H MEDIAN-COMPUTE skip ($nmed/$npred exist)"
  else
    echo "[score $CONF gpu$GPU] t+$H MEDIAN-COMPUTE $(date '+%F %T')"
    $PY "$IO/make_median.py" "$OUT" >> "$OUT/median.log" 2>&1 \
      || { echo "[score $CONF gpu$GPU] t+$H MEDIAN-COMPUTE FAILED"; continue; }
  fi
  if [ ! -f "$OUT/metrics/metrics_summary.json" ]; then
    echo "[score $CONF gpu$GPU] t+$H MEAN-SCORE $(date '+%F %T')"
    $PY "$MET" --pred-dir "$OUT/prediction" --target-dir "$TGT" --outdir "$OUT/metrics" \
       --metrics mse_m11 ssim crps lpips fid --value-range m11 --device "cuda:$GPU" >> "$OUT/metrics.log" 2>&1 \
      || echo "[score $CONF gpu$GPU] t+$H MEAN-SCORE FAILED"
  else
    echo "[score $CONF gpu$GPU] t+$H MEAN-SCORE skip (exists)"
  fi
  if [ -f "$OUT/metrics_median/metrics_summary.json" ]; then
    echo "[score $CONF gpu$GPU] t+$H MEDIAN-SCORE skip (exists)"
  else
    echo "[score $CONF gpu$GPU] t+$H MEDIAN-SCORE $(date '+%F %T')"
    $PY "$MET" --pred-dir "$OUT/prediction_median" --target-dir "$TGT" --outdir "$OUT/metrics_median" \
       --metrics mse_m11 ssim lpips fid --value-range m11 --device "cuda:$GPU" >> "$OUT/metrics_median.log" 2>&1 \
      || echo "[score $CONF gpu$GPU] t+$H MEDIAN-SCORE FAILED"
  fi
  echo "[score $CONF gpu$GPU] t+$H DONE $(date '+%F %T')"
done
echo "[score $CONF gpu$GPU] ALL DONE $(date '+%F %T')"
