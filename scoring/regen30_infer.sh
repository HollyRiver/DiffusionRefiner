#!/usr/bin/env bash
# 30ep best 체크포인트의 멤버-보정 추론 재생성 (canonical "8 members refined mean").
# 한 지평선. work_memberset/best_R.pt (30ep 학습 best) 사용. seed 1253, --save_members.
# 산출: {ROOT}/work30_inf_best/outputs_t{H}_member/{prediction,refined_member_00..07}/
# Usage: regen30_infer.sh <horizon 1|2|3|4> <gpu>
# 주의: 착수 전 dumps_members/test 가 정확히 8멤버(member_0..7)인지 재확인할 것.
set -u
H="$1"; GPU="$2"
PY=/home/user/miniconda3/envs/wind_power/bin/python
IO=/DATA/shinsung/inference_outputs
MOD=/DATA/shinsung/refiner_module
ROOT="$IO/analysis/t$H"
CKPT="$IO/refined/K8/30ep_best/ckpt/t${H}/best_R.pt"
DUMPS="$ROOT/dumps_members"
WORK="$ROOT/work30_inf_best"
DONE="$WORK/outputs_t${H}_member/prediction"

if [ ! -f "$CKPT" ]; then echo "[ERR] ckpt 없음: $CKPT"; exit 1; fi
nmem=$(ls -d "$DUMPS/test"/member_* 2>/dev/null | wc -l)
if [ "$nmem" -ne 8 ]; then
  echo "[ERR] t+$H dumps_members/test 멤버 수=$nmem (기대 8). 착수 중단 — canonical은 8멤버."; exit 2
fi
# 완결성 스킵: 이미 target 수만큼 나온 경우 건너뜀
ntgt=$(ls /DATA/shinsung/lon_refiner/ablation_vol_gOFF_lON_t${H}/test_outputs/target/*.pt 2>/dev/null | wc -l)
npred=$(ls "$DONE"/*.pt 2>/dev/null | wc -l)
if [ "$npred" -ge "$ntgt" ] && [ "$ntgt" -gt 0 ]; then
  echo "[skip] t+$H already complete ($npred/$ntgt)"; exit 0
fi
mkdir -p "$WORK"
echo "[regen30] t+$H ckpt=$CKPT gpu=$GPU seed=1253 30ep_best(t1=ep27,t2=ep26,t3=ep30,t4=ep28) $(date)"
cd "$MOD" && $PY infer_refiner.py --mode member --offset "$H" --split test --gpu "$GPU" \
  --dumps_root "$DUMPS" --raw_data_dir /DATA/solar \
  --work_dir "$WORK" --R_ckpt "$CKPT" --save_members --seed 1253 \
  >> "$WORK/infer.log" 2>&1
echo "[regen30] t+$H done $(date) -> $(ls "$DONE"/*.pt 2>/dev/null | wc -l) preds"
