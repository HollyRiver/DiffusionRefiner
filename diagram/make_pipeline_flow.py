"""refiner_module 파이프라인 흐름도 (pipeline_flow.png) 생성 스크립트.

    python diagram/make_pipeline_flow.py      # → pipeline_flow.png (README.md 와 같은 최상위 폴더)

matplotlib 만 사용(CPU). 내용(경로·명령)을 고칠 때는 아래 텍스트만 바꾸면 된다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "pipeline_flow.png"
SANS = ["Noto Sans CJK JP", "DejaVu Sans"]
MONO = ["DejaVu Sans Mono", "Noto Sans CJK JP"]

# 색 (배경 / 테두리·머리띠)
C = {
    "in_arch": ("#e8f1fb", "#2f6db3"),    # 아카이브 안 데이터
    "in_out": ("#fdf0e2", "#c26a12"),     # 아카이브 밖 데이터
    "env": ("#f0f0f0", "#6b6b6b"),        # 실행 환경
    "step": ("#ffffff", "#333333"),       # 코드 단계
    "opt": ("#ffffff", "#8a8a8a"),        # 선택 단계
    "out": ("#e9f5ec", "#2e7d4a"),        # 산출물
}
SCN = {"A": "#7b4bb7", "B": "#1f77b4", "C": "#2e8b57", "D": "#d2691e"}

LH = 1.32        # 본문 줄 높이 (데이터 단위)
TH = 2.1         # 제목 띠 높이
FS, TFS = 10.5, 12.5

fig = plt.figure(figsize=(28, 20), dpi=110)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 140); ax.set_ylim(0, 100); ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, ytop, w, title, lines, kind, chips=(), dashed=False):
    """제목 띠 + 본문 줄. lines 항목이 '$' 또는 '`'로 시작하면 고정폭. 반환: (x, ybot, w, h)."""
    face, edge = C[kind]
    h = TH + LH * len(lines) + 0.9
    yb = ytop - h
    ax.add_patch(FancyBboxPatch((x, yb), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=face, ec=edge, lw=2.0, ls="--" if dashed else "-", zorder=2))
    ax.add_patch(FancyBboxPatch((x, ytop - TH), w, TH, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=edge, ec=edge, lw=0, zorder=3))
    ax.text(x + 0.8, ytop - TH / 2, title, fontsize=TFS, fontweight="bold", color="white",
            va="center", ha="left", family=SANS, zorder=4)
    cx = x + w - 0.7
    for c in reversed(chips):            # 시나리오 칩 (제목 띠 오른쪽)
        ax.text(cx, ytop - TH / 2, c, fontsize=10.5, fontweight="bold", color="white",
                ha="right", va="center", family=SANS, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc=SCN[c[0]], ec="white", lw=1.2))
        cx -= 1.9 if len(c) == 1 else 4.6
    for i, ln in enumerate(lines):
        mono = ln[:1] in "$`"
        txt = ln[1:] if ln[:1] == "`" else ln
        ax.text(x + 0.9, ytop - TH - 0.75 - LH * (i + 0.5), txt, fontsize=FS - (0.6 if mono else 0),
                family=MONO if mono else SANS, va="center", ha="left", zorder=4, color="#1a1a1a")
    return x, yb, w, h


def arrow(p, q, color="#333333", ls="-", rad=0.0, lw=2.0, label=None, lab_xy=None, lab_color=None):
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=18,
                                shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(*lab_xy, label, fontsize=10, color=lab_color or color, family=SANS,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


# ── 제목 · 범례 ─────────────────────────────────────────────────────────────
ax.text(2, 97.6, "refiner_module 파이프라인 — 필요한 것 · 단계 · 산출물 위치", fontsize=24,
        fontweight="bold", family=SANS, va="center")
ax.text(2, 94.9, "잔차 확산 보정기(RefinementUNet, member 모드, K=8). 지평선 H = 1~4 (+30/60/90/120분)마다 "
        "같은 흐름을 따로 돌린다.  {stem} = 예측 대상 시각(UTC), W = --work_dir",
        fontsize=12.5, family=SANS, va="center", color="#333333")
lx = 101
for kind, lab in [("in_arch", "데이터 — 아카이브 안"), ("in_out", "데이터 — 아카이브 밖"), ("env", "실행 환경"),
                  ("out", "산출물 / 아카이브 위치")]:
    face, edge = C[kind]
    ax.add_patch(FancyBboxPatch((lx, 96.9), 1.6, 1.3, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=face, ec=edge, lw=1.8))
    ax.text(lx + 2.1, 97.55, lab, fontsize=10.5, family=SANS, va="center")
    lx += 9.6 if kind != "out" else 0
ax.text(101, 95.1, "칩 = 그 단계를 거치는 시나리오 (아래 표)", fontsize=10.5, family=SANS, va="center")
for i, c in enumerate("ABCD"):
    ax.text(120.5 + i * 1.9, 95.1, c, fontsize=10.5, fontweight="bold", color="white", ha="center",
            va="center", family=SANS, bbox=dict(boxstyle="round,pad=0.25", fc=SCN[c], ec="white"))

for x, t in [(2, "필요한 것 (입력 · 환경)"), (40, "단계 (refiner_module 코드)"),
             (98, "산출물  →  inference_outputs/ 안의 위치")]:
    ax.text(x, 91.6, t, fontsize=15, fontweight="bold", family=SANS, va="center", color="#222222")

# ── 단계 (가운데) ───────────────────────────────────────────────────────────
SX, SW, GAP = 40, 52, 2.6
y = 89.8
S = {}
S[0] = box(SX, y, SW, "⓪ 덤프 연결 — 입력 계약 맞추기", [
    "`{dumps_root}/{train,val,test}/member_*/{stem}.pt  (이 모듈의 입력 계약)",
    "아카이브에는 이미 연결돼 있다:  analysis/t{H}/dumps_members/{train,val,test}",
    "    = base/t{H}/{train,valid,test}/members 로의 링크  (split 이름 valid → val)",
], "step", chips=("A", "B"))
y = S[0][1] - GAP
S[1] = box(SX, y, SW, "① 학습 — train_refiner.py", [
    "$ python train_refiner.py --mode member --offset H \\",
    "$     --train_members all --val_members all --expect_members 8 \\",
    "$     --epochs 30 --save_every 1 --seed 1253 \\",
    "$     --dumps_root D --raw_data_dir /DATA/solar --work_dir W",
    "train 8멤버 전부로 잔차(GT − 덤프) 학습, valid 8멤버로 best epoch 선택",
    "코드: config.py (하이퍼파라미터 기본값), refiner/{dataset, unet, utils}.py",
    "※ 정본 체크포인트는 학습 시드 도입 전에 학습됨 → 재학습 값은 정확히 같지 않다",
], "step", chips=("A",))
y = S[1][1] - GAP
S[2] = box(SX, y, SW, "② 보정 추론 — infer_refiner.py   (실행 기록: scoring/regen30_infer.sh)", [
    "$ python infer_refiner.py --mode member --offset H --split test \\",
    "$     --R_ckpt best_R.pt --save_members --seed 1253 \\",
    "$     --dumps_root D --raw_data_dir /DATA/solar --work_dir W  [--shards N --shard k]",
    "test 멤버 8개를 각각 보정(DDIM 40) → 평균 = 최종 예측",
    "초기 노이즈 시드가 샘플명 기반 → 같은 체크포인트면 비트 단위로 같은 결과",
], "step", chips=("A", "B"))
y = S[2][1] - GAP
S[3] = box(SX, y, SW, "③ 앙상블 집계 (선택)", [
    "scoring/make_base_ens.py  :  보정 전 base 멤버 → 평균 / median",
    "scoring/make_median.py    :  보정 멤버 refined_member_* → median",
], "opt", chips=("A", "B", "C"), dashed=True)
y = S[3][1] - GAP
S[4] = box(SX, y, SW, "④ 채점 — scoring/forecast_metrics_eval_fid500.py  (+ compute_mae.py)", [
    "보정 앙상블 K=8       : scoring/score_ensembles.sh  (samples/member_XX 링크 → fair CRPS)",
    "보정 단일 member_00  : 점예보로 채점 — CRPS 대신 MAE (전용 스크립트 미보존)",
    "보정 전 base K=8      : scoring/score_base_ensemble.sh",
    "MAE                  : scoring/compute_mae.py",
    "지표: MSE·MAE (m11), SSIM (win7), CRPS (fair), LPIPS (AlexNet), FID (Inception 2048-d)",
    "GPU 권장 (LPIPS·FID). 채점기 import 전에 CUDA 컨텍스트를 먼저 만들 것",
], "step", chips=("A", "B", "C"))
y = S[4][1] - GAP
S[5] = box(SX, y, SW, "⑤ 표 · 분석 · 그림", [
    "표   : analysis/final_table_*.csv,  baselines/summary_baselines.csv",
    "분석 : analysis/{mae, mse_decomposition, seasonal, spatial_cluster}/",
    "그림 : fig/{figure3a, make_qualitative_panel, make_shape_compare}.py",
], "step", chips=("A", "B", "C", "D"))

for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
    xa, ya, wa, _ = S[a]
    xb, yb, wb, hb = S[b]
    arrow((xa + wa * 0.5, ya), (xb + wb * 0.5, yb + hb))

# ── 입력 (왼쪽) ─────────────────────────────────────────────────────────────
IX, IW = 2, 33
I = {}
I["dump"] = box(IX, 89.8, IW, "DAD 예측 덤프 (Stage 1 출력, K=8)", [
    "`base/t{H}/{train,valid,test}/",
    "`    members/member_0..7/{stem}.pt",
    "m11 [-1, 1], (1, 512, 512) float32",
    "train·valid = ① 학습용,  test = ②③④용",
    "✔ 아카이브 안 (401G; test만 128G)",
    "생성 코드(DAD, defaultcode2)는 범위 밖",
], "in_arch")
I["raw"] = box(IX, S[1][1] + S[1][3] - 3.5, IW, "원시 위성 프레임", [
    "`/DATA/solar/{stem}.npy",
    "30분 간격 일사량 원값 (0 ~ 26.347)",
    "덤프 파일명에서 시각을 역산해 읽음:",
    "  context 5장 (①②) + 학습 GT (①)",
    "✘ 아카이브 밖",
], "in_out")
I["gt"] = box(IX, S[4][1] + S[4][3] - 0.5, IW, "채점용 GT", [
    "`lon_refiner/ablation_vol_gOFF_lON_t{H}/",
    "`    test_outputs/target/{stem}.pt",
    "= 원시 프레임의 m11 정규화",
    "✘ 아카이브 밖 (원시 프레임으로 재생성)",
], "in_out")
I["env"] = box(IX, I["gt"][1] - 2.4, IW, "실행 환경 (env: wind_power)", [
    "학습·추론: torch, numpy",
    "채점: scikit-image, lpips, pytorch-fid,",
    "      scipy, Pillow, torchvision",
    "GPU: 학습 ~33분/epoch, 추론 ~4시간 (L40S, t+1)",
], "env")

def right(b, f=0.5):
    return (b[0] + b[2], b[1] + b[3] * f)

def left(b, f=0.5):
    return (b[0], b[1] + b[3] * f)

arrow(right(I["dump"], 0.55), left(S[0], 0.55), color=C["in_arch"][1], lw=2.4)
arrow(right(I["raw"], 0.62), left(S[1], 0.45), color=C["in_out"][1], lw=2.4, rad=-0.1)
arrow(right(I["raw"], 0.35), left(S[2], 0.6), color=C["in_out"][1], lw=2.4, rad=0.1)
arrow(right(I["gt"], 0.5), left(S[4], 0.6), color=C["in_out"][1], lw=2.4)

# ── 산출물 (오른쪽) ─────────────────────────────────────────────────────────
OX, OW = 98, 40
O = {}
O[0] = box(OX, S[0][1] + S[0][3], OW, "링크만 (복사 없음)", [
    "analysis/t{H}/dumps_members/  → 그대로 D 로 써도 됨",
    "새로 만들 때는 split 폴더 단위 심볼릭 링크",
], "out")
O[1] = box(OX, S[1][1] + S[1][3], OW, "W/refiner_t{H}_member/", [
    "best_R.pt · last_R.pt · ep{N}_R.pt · loss_log.csv · train_cmd.txt",
    "→ 아카이브: refined/K8/30ep_best/ckpt/t{H}/",
    "        best_R.pt, loss_log.csv  (best epoch 27 / 26 / 30 / 28)",
    "체크포인트 1개 ≈ 9MB (2.25M 파라미터)",
], "out")
O[2] = box(OX, S[2][1] + S[2][3], OW, "W/outputs_t{H}_member/", [
    "prediction/              8멤버 평균 = 최종 예측 (Ours)",
    "refined_member_00..07/   멤버별 보정 결과",
    "→ 아카이브: refined/K8/30ep_best/t{H}/  (test, 128G)",
], "out")
O[3] = box(OX, S[3][1] + S[3][3], OW, "집계 결과", [
    "analysis/t{H}/base_K8/prediction  (= base/t{H}/test/ensemble_mean 링크)",
    "median 텐서는 삭제됨 — 지표(metrics_median/)만 보존",
], "out")
O[4] = box(OX, S[4][1] + S[4][3], OW, "metrics_summary.json + per_image_metrics.csv", [
    "→ refined/K8/30ep_best/t{H}/metrics/            (refined, K=8)",
    "→ refined/K8/30ep_best/t{H}/member0_eval/metrics/ (refined single)",
    "→ analysis/t{H}/base_K8/metrics/                  (base, K=8)",
    "→ analysis/mae/t{H}/summary.json                  (MAE)",
], "out")
O[5] = box(OX, S[5][1] + S[5][3], OW, "결과표 · 그림", [
    "analysis/final_table_mean_K8_DADvRefined.csv  (메인 표)",
    "baselines/{acunet, hrnet, convlstm}/  — 같은 채점기로 채점",
    "fig/*.png  (Figure 3(a) 등) — 설명: fig/README.md",
], "out")
for k in range(6):
    arrow(right(S[k], 0.5 if k else 0.5), left(O[k], 0.5), color=C["out"][1], lw=2.0)

# C 시나리오: 아카이브 예측 → ④
arrow(left(O[2], 0.15), right(S[4], 0.88), color=SCN["C"], ls="--", rad=0.25, lw=2.2,
      label="C: 아카이브\n예측 재채점", lab_xy=(94.6, (S[2][1] + S[3][1] + S[3][3]) / 2),
      lab_color=SCN["C"])
# B 시나리오: 아카이브 best_R.pt → ② (① 학습을 건너뜀)
arrow(left(O[1], 0.2), right(S[2], 0.85), color=SCN["B"], ls="--", rad=0.25, lw=2.2,
      label="B: 아카이브\nbest_R.pt", lab_xy=(94.6, (O[1][1] + S[2][1] + S[2][3]) / 2 - 0.4),
      lab_color=SCN["B"])

# ── 시나리오 표 (아래 왼쪽) ─────────────────────────────────────────────────
TX, TW, TT = 2, 90, S[5][1] - 3.2
rows = [
    ("A", "처음부터 재학습", "⓪→①→②→④→⑤",
     "DAD 덤프(train·valid·test), 원시 프레임, GT, GPU", "지평선당 학습 ~17시간 + 추론 ~4시간"),
    ("B", "정본 가중치로 예측 재생성", "⓪→②→④→⑤",
     "best_R.pt(아카이브), DAD 덤프(test), 원시 프레임, GT, GPU", "아카이브 예측과 비트 단위 동일"),
    ("C", "다시 채점만", "(③)→④→⑤",
     "보정 예측(아카이브 refined/), GT  (+ base 행은 DAD test 멤버)", "원시 프레임·학습 불필요"),
    ("D", "표·그림만", "⑤",
     "metrics/, 예측 텐서  (그림 일부는 GT·원시 프레임)", "GPU 불필요"),
]
rh = 2.55
th = TH + rh * len(rows) + 1.6
ax.add_patch(FancyBboxPatch((TX, TT - th), TW, th, boxstyle="round,pad=0,rounding_size=0.6",
                            fc="white", ec="#333333", lw=2.0))
ax.add_patch(FancyBboxPatch((TX, TT - TH), TW, TH, boxstyle="round,pad=0,rounding_size=0.6",
                            fc="#333333", ec="#333333", lw=0))
ax.text(TX + 0.8, TT - TH / 2, "어디서 시작하나 — 시나리오별 경로와 필요한 것", fontsize=TFS,
        fontweight="bold", color="white", va="center", family=SANS)
cols = [TX + 0.9, TX + 3.2, TX + 18.5, TX + 31.5, TX + 70.5]
hy = TT - TH - 1.1
for cx, h in zip(cols[1:], ["시나리오", "단계", "필요한 것", "비고"]):
    ax.text(cx, hy, h, fontsize=10.5, fontweight="bold", family=SANS, va="center", color="#555555")
for i, (c, name, path, need, note) in enumerate(rows):
    yy = hy - 1.9 - i * rh
    ax.text(cols[0] + 0.6, yy, c, fontsize=11, fontweight="bold", color="white", ha="center",
            va="center", family=SANS, bbox=dict(boxstyle="round,pad=0.3", fc=SCN[c], ec="white"))
    ax.text(cols[1], yy, name, fontsize=11, fontweight="bold", family=SANS, va="center")
    ax.text(cols[2], yy, path, fontsize=11, family=SANS, va="center")
    ax.text(cols[3], yy, need, fontsize=10.5, family=SANS, va="center")
    ax.text(cols[4], yy, note, fontsize=10.5, family=SANS, va="center", color="#444444")

# ── 주의사항 (아래 오른쪽) ──────────────────────────────────────────────────
notes = [
    "· 파일명 {stem} = 예측 대상(valid) 시각, UTC. KST = UTC+9. 모든 단계가 이 이름으로 짝을 맞춘다",
    "· 정규화 m11:  x_m11 = 2·x / 26.347150802612 − 1  (data_min 0, config.py)",
    "· 멤버 폴더: base 는 member_0..7 (한 자리), 보정 출력은 refined_member_00..07 (두 자리)",
    "· 실험마다 --work_dir 를 분리하고, --expect_members 8 로 멤버 수를 고정할 것",
    "· scoring/ 스크립트 안 경로는 서버 절대경로 → scoring/README.md 의 경로 대응표",
    "   (score_base_ensemble.sh 의 dumps_members12 는 삭제됨 → base/t{H}/test/members)",
    "· --mode ensemble (평균 후 1회 보정)은 비교 실험에서 기각. 정본은 --mode member",
]
NX, NW = 96, 42
nh = TH + 1.32 * len(notes) + 1.0
ax.add_patch(FancyBboxPatch((NX, TT - nh), NW, nh, boxstyle="round,pad=0,rounding_size=0.6",
                            fc="#fffbea", ec="#b08900", lw=2.0))
ax.add_patch(FancyBboxPatch((NX, TT - TH), NW, TH, boxstyle="round,pad=0,rounding_size=0.6",
                            fc="#b08900", ec="#b08900", lw=0))
ax.text(NX + 0.8, TT - TH / 2, "주의사항", fontsize=TFS, fontweight="bold", color="white",
        va="center", family=SANS)
for i, n in enumerate(notes):
    ax.text(NX + 0.8, TT - TH - 0.75 - 1.32 * (i + 0.5), n, fontsize=10, family=SANS, va="center")

foot_y = TT - th - 1.6
ax.text(138, foot_y, "생성: diagram/make_pipeline_flow.py  ·  2026-10-19 기준 (inference_outputs 아카이브 레이아웃)",
        fontsize=9.5, family=SANS, ha="right", va="center", color="#777777")
# 아래 빈 공간 잘라내기: 1 단위 = 0.2 inch 를 유지하도록 그림 높이도 함께 줄인다
ymin = foot_y - 1.6
ax.set_ylim(ymin, 100)
fig.set_size_inches(28, 20 * (100 - ymin) / 100)
fig.savefig(OUT, dpi=110, facecolor="white")
print(f"saved {OUT}")
