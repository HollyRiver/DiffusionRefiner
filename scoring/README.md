# scoring/ — 정본 수치를 만든 채점·실행 파일 (원본 사본)

논문 표의 보정기 수치(K=8 정본)를 실제로 산출한 파일들을 **수정 없이** 모았다(원본과 sha256 동일, 2026-10-17 수집).
파일 안의 경로는 연구 서버 절대경로 그대로다 — 다른 환경에서 쓰려면 경로만 바꾸면 된다.

**2026-10-28 변경**: 서버 루트의 `/DATA/shinsung/forecast_metrics_eval_fid500.py`(이 폴더의 것과 sha256 동일)를 지웠다.
그래서 `score_ensembles.sh`·`score_base_ensemble.sh` 의 `MET=` 한 줄만 **이 폴더의 채점기**를 가리키도록 바꿨다(그 밖의 줄과 채점기 자체는 원본 그대로).
`compute_mae.py` 는 주석 한 줄만 바꿨다.
같은 날 채점 GT 를 `lon_refiner/ablation_vol_gOFF_lON_tH/test_outputs/target` 에서 **`inference_outputs/gt/tH`** 로 옮겨,
`score_ensembles.sh`·`score_base_ensemble.sh`·`regen30_infer.sh`·`compute_mae.py` 의 GT 경로 한 줄씩을 새 위치로 바꿨다.

| 파일 | 원래 위치 | 역할 |
|---|---|---|
| `forecast_metrics_eval_fid500.py` | `/DATA/shinsung/` (2026-10-28 삭제, 이제 이 사본이 유일) | **정본 채점기** — MSE(m11)·SSIM(win7)·CRPS(fair)·LPIPS(alex)·FID(pytorch_fid 2048-d). CLI 또는 `compute_metrics()` |
| `compute_mae.py` | `inference_outputs/analysis/mae/` | 앙상블 MAE (채점기에 MAE가 없어 따로 계산) |
| `make_base_ens.py` | `inference_outputs/` | 보정 전 DAD 멤버 → 평균/median 앙상블 |
| `make_median.py` | `inference_outputs/` | 보정 멤버(`refined_member_*`) → median 앙상블 |
| `regen30_infer.sh` | `inference_outputs/` | 정본 보정 추론 실행 (`infer_refiner.py --mode member --save_members --seed 1253`, best_R) |
| `score_ensembles.sh` | `inference_outputs/` | 보정 앙상블 채점 실행 (`samples/member_XX` 링크 → fair CRPS) |
| `score_base_ensemble.sh` | `inference_outputs/` | 보정 전 base 앙상블 집계·채점 실행 |

## 실행 순서 (정본)

1. 학습: `train_refiner.py --mode member --train_members all --epochs 30` (나머지 `config.py` 기본값).
   정본 체크포인트(2026-07~08)는 실행 명령이 기록되지 않아 학습 로그에서 역산했다. 학습 시드가 없던 시기라 재학습해도 같은 값이 나오지 않는다.
2. 추론: `regen30_infer.sh <H> <gpu>`
3. 채점: `score_ensembles.sh <gpu> 8mem <H>` (보정 앙상블), `score_base_ensemble.sh <gpu> <H>:8` (base 앙상블), `compute_mae.py --H <H>` (MAE)
   - 단일 멤버("refined (single)" 행): `member0_eval/prediction/`에 `refined_member_00`의 파일 링크를 두고 채점기를 돌렸다
     (형제 `samples/`가 없으면 점예보로 채점되며 crps 칸 = MAE). 따로 남은 스크립트는 없다.

## 경로 대응 (2026-10-17 아카이브 기준)

| 스크립트 안 경로 | 현재 위치 |
|---|---|
| `inference_outputs/analysis/tH/work30_inf_best/outputs_tH_member` | `inference_outputs/refined/K8/30ep_best/tH` |
| `inference_outputs/refined/K8/30ep_best/ckpt/tH/best_R.pt` | 그대로 |
| `inference_outputs/analysis/tH/dumps_members` | 그대로 (→ `base/tH/{train,valid,test}/members`) |
| `inference_outputs/analysis/tH/dumps_members12` | 삭제됨(K=12). K=8 base 멤버 = `base/tH/test/members` |
| `inference_outputs/make_*.py` | `inference_outputs/scripts_logs/` (그리고 이 폴더) |
| GT `inference_outputs/gt/tH` (옛 `lon_refiner/ablation_vol_gOFF_lON_tH/test_outputs/target`) | 그대로 (2026-10-28 이동, 스크립트도 새 경로) |

## 채점 의존성 (검증된 버전)

`scikit-image` 0.25.2, `lpips` 0.1.4, `pytorch-fid` 0.3.0 (+ `scipy` 1.15.3, `Pillow` 12.0.0, `torchvision` 0.24.1), `torch` 2.9.1.
GPU로 채점할 때는 채점기를 import하기 **전에** CUDA 컨텍스트를 먼저 만들 것. 순서가 반대면 `torch.cuda.is_available()`이 False로 굳어 조용히 CPU로 떨어진 사례가 있다.
