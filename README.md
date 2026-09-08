# Diffusion-Based Residual Generation Refiner (아마도?)

&nbsp;베이스 예측 모델(DAD)의 추론 덤프를 입력으로, 잔차 확산(residual diffusion) Refiner를 학습하고 최종 보정 예측을 산출하는 자립형 모듈입니다. 올인원 리포지토리입니다. 또한, 설명의 대부분은 클로드 코드에 의해 작성되었습니다. 저도 아직 이해중입니다.

- 의존성: `torch`, `numpy` (그 외 없음)
- 원리: `residual = GT − y_blur` 를 x0-파라미터화 확산으로 학습, 추론 시 DDIM으로
  residual을 샘플링해 `y_sharp = y_blur + residual`

> 📖 **기술 문서** ☆★☆★☆★☆ 여기에 이론적 내용이 적혀있음!!! ★☆★☆★☆★: 원리 · 수학적 정식화 · 아키텍처 해부 · 실험 결과를 다이어그램과
> 함께 정리한 문서 — [RefinementUNet 기술 문서](https://hollyriver.github.io/DiffusionRefiner/)
>
> ⭐ 연구에서 **채택된 방식은 member 모드** — 멤버별 보정 후 평균("8 members refined
> mean")입니다. ensemble 모드(평균 후 1회 보정)는 비교 실험에서 정확도·확률 지표
> (MSE·SSIM·CRPS) 열세로 채택되지 않았으며, 코드에는 비교·일반 용도로 남아 있습니다.

## 입력 계약

```
{dumps_root}/                      # config.py 또는 --dumps_root
├── train/member_00/2021-06-01_0600.pt    # 파일명 = "예측 대상" 타임스탬프
│         member_01/...                   # 멤버 여러 개 = 앙상블 (1개만 있어도 OK)
├── val/member_00/...                     # 검증용 (없으면 경고 후 진행)
└── test/member_00/...                    # 최종 보정 대상

{raw_data_dir}/2021-06-01_0600.npy        # 원본 프레임 (30분 간격 raw 값)
```

- 덤프 `.pt`: shape `(1,H,W)` 또는 `(H,W)`, 값 범위 **m11** (`2*(x-min)/(max-min)-1`)
- 원본 `.npy`: 물리 단위 raw 값. Refiner가 여기서 context 5장(+학습 시 GT)을 읽습니다.
  **원본 프레임 디렉토리는 필수입니다** — 덤프만으로는 동작하지 않습니다.
- horizon `--offset N`: 예측 시점이 마지막 context로부터 N프레임(=N×30분) 뒤라는 뜻

## 사용법

먼저 `config.py`에서 `dumps_root` / `raw_data_dir` / `work_dir`를 설정하세요
(또는 매번 CLI 인자로 지정).

**모드 A — train/val에도 앙상블이 있는 경우 (`ensemble`)**
멤버 평균을 학습하고, test 멤버 평균을 1회 보정합니다.

```bash
python train_refiner.py --mode ensemble --offset 1 --gpu 0
python infer_refiner.py --mode ensemble --offset 1 --gpu 0
```

**모드 B — train/val에 단일 멤버만 있는 경우 (`member`)**
member_00 하나로 학습하고, test 멤버 K개를 각각 보정한 뒤 평균해 최종 출력을 만듭니다.

```bash
python train_refiner.py --mode member --offset 1 --gpu 0
python infer_refiner.py --mode member --offset 1 --gpu 0 --save_members
```

**모드 B 변형 — train/val에도 멤버가 전부 있는 경우 (`--train_members all`)**
모든 멤버를 union으로 학습합니다(타임스탬프당 멤버 수만큼 학습 샘플).
검증 풀은 `--val_members`로 **학습과 독립적으로** 정합니다: `first`를 주면
member_0 하나로만 검증해 에폭당 검증 비용이 1/K로 줄고, val MSE는 단일 멤버의
수천 샘플 평균만으로도 안정적이라 best 선택에 미치는 영향은 미미합니다.
추론은 모드 B와 같습니다(멤버 각각 보정 → 평균). 에폭당 데이터가 멤버 수배가 되므로
`--epochs`/`--warmup_epochs`를 그에 맞게 줄이는 것을 권장합니다.

`--expect_members K`를 함께 주세요. dumps의 split 폴더가 실제 멤버 폴더로의 심링크인
구성에서는 그 대상에 멤버가 추가되면 **에러 없이 K가 바뀝니다**. K를 표에 쓰는 실험에서는
발견된 멤버 수가 기대와 다를 때 즉시 중단시키는 편이 안전합니다(종료코드 1).

```bash
python train_refiner.py --mode member --train_members all --val_members first \
    --expect_members 8 --offset 1 --gpu 0 --epochs 12 --warmup_epochs 4
python infer_refiner.py --mode member --offset 1 --gpu 0 --save_members
```

> ⚠️ **2026-09-08 동작 변경 2건** (재현성 수정)
>
> 1. `--val_members all`이 이제 실제로 전 멤버 검증을 합니다. 이전 구현은
>    `train_members == "all"`일 때만 `all`을 인정해, **기본값 조합**
>    (`--train_members first` + `--val_members all`)에서 조용히 member_0 단독 검증을
>    했습니다. 플래그를 명시해 온 런(`--train_members all --val_members first` 등)의
>    동작과 과거 결과 해석은 바뀌지 않습니다.
> 2. 검증의 확산 타임스텝·노이즈가 **매 epoch 고정 추출**로 바뀌었습니다. 이전에는 매
>    epoch 새로 뽑아 val 값에 샘플링 노이즈가 섞였고, `best_R.pt` 선택이 그 노이즈에
>    좌우될 수 있었습니다(모델은 그대로인데 쉬운 t가 뽑힌 epoch이 best가 되는 식).
>    옛 동작은 `--val_stochastic`.

> ✅ **완료 (2026-08-26)**: 앙상블을 12멤버로 확장한 union 재학습(이 조합
> `--train_members all --val_members first`, member_0 단독 검증)과 mean·median 집계
> 비교를 완료했습니다. 결론 — **median 집계는 소폭의 정확도 손실로 FID·LPIPS를 개선**하고,
> **8→12 확장은 뚜렷한 이득이 없습니다**(FID·LPIPS·CRPS는 8멤버가 우위). 전체 결과 표는
> [기술 문서 §9](https://hollyriver.github.io/DiffusionRefiner/#s9)를 참고하세요.

주의: 변형(예: first vs all)마다 체크포인트 폴더명이 같으므로(`refiner_t{N}_member`)
**실험별로 `--work_dir`를 분리**하세요.

최종 출력: `{work_dir}/outputs_t{N}_{mode}/prediction/{timestamp}.pt`
(m11 범위, `(1,H,W)`, 타임스탬프당 1개). 물리 단위 복원: `x_raw = (x+1)/2*(max-min)+min`.

체크포인트: `{work_dir}/refiner_t{N}_{mode}/`에 `best_R.pt`(최저 val), `last_R.pt`(최종
epoch)와 함께 기본값으로 **매 epoch `ep{에폭}_R.pt`가 저장됩니다** (`--save_every N`,
0=끄기, 개당 ~9MB — 30 epoch에 ~270MB). 중간-epoch 성능 비교나 특정 epoch 재현이
나중에 필요해질 수 있으므로 끄지 않기를 권장합니다.

horizon이 여러 개면 `--offset 1..4`를 각각 실행하세요 (horizon마다 별도 refiner).

## 멀티-GPU 추론 (선택)

테스트 세트를 strided로 쪼개 GPU마다 한 프로세스씩 돌릴 수 있습니다. 초기 노이즈 시드가
**샘플명 기반**(`seed_from_name(base_seed + 1000*k, name)`)이라 샤드 구성이 결과에 영향을
주지 않습니다 — 샤드 합집합이 단일 실행과 비트 단위로 동일함을 확인했습니다.

```bash
# GPU 0, 1 에 각각
python infer_refiner.py --mode member --offset 1 --gpu 0 --save_members --shards 2 --shard 0
python infer_refiner.py --mode member --offset 1 --gpu 1 --save_members --shards 2 --shard 1
```

`--shard`는 `[0, --shards)` 범위여야 하며 벗어나면 중단합니다(종료코드 1). 검사가 없으면
파이썬 슬라이스가 에러 없이 다른 샤드와 겹치는 구간을 돌려주어, 일부 샘플이 두 번 계산되고
일부는 누락된 채 조용히 끝납니다. `--work_dir`는 샤드별로 분리하고, 채점 전에 합치세요.

## 재현성

`--seed`(기본 `config.py: seed`)가 학습 난수 4곳을 모두 고정합니다 — **모델 초기화**,
DataLoader shuffle, 확산 타임스텝 `t`, 노이즈. 같은 시드·같은 설정이면 체크포인트
가중치가 비트 단위로 일치합니다(검증됨). 비트 단위 재현을 더 강하게 원하면
`--cudnn_deterministic`(느려짐)을 추가하세요.

호출 CLI 전문은 `{work_dir}/refiner_t{N}_{mode}/train_cmd.txt`에 실행마다 append됩니다.
나중에 "이 체크포인트가 어떤 설정으로 나왔는지"를 역산하지 않으려면 이 파일을
체크포인트와 함께 보관하세요.

## 검증

데이터 없이 모듈 자체를 검증하려면 (CPU만으로 수 분):

```bash
python tests/run_smoke.py     # 마지막 줄 SMOKE OK 확인
python tests/test_utils.py
python tests/test_dataset.py
```

## ⭐ 적응 가이드 — 환경이 이 계약과 다를 때 고치는 곳

| # | 다른 점 | 고치는 곳 |
|---|---------|-----------|
| 1 | 파일명 타임스탬프 규칙이 다름 (예: `20210601T0600.pt`) | `config.py: filename_format` (원본 프레임 파일명 생성용) + `refiner/utils.py: DT_FORMATS / ParseDtFromName` (덤프 파일명 파싱용) — **둘 다** 맞춰야 함 |
| 2 | 멤버 폴더 이름이 `member_XX`가 아님 (예: `seed0/`) | `config.py: member_glob` (예: `"seed*"`). 패턴으로 안 되면 `refiner/dataset.py: discover_members` 한 함수만 수정 |
| 3 | 덤프 값이 m11이 아니라 물리 단위(raw) | `config.py: dump_value_range = "raw"` (자동으로 m11 변환됨) |
| 4 | 프레임 간격이 30분이 아님 / context 길이가 5가 아님 | `config.py: frame_interval_minutes`, `num_context` (context 길이 변경 시 재학습 필요 — 모델 입력 채널이 바뀜) |
| 5 | 정규화 상수(min/max)가 다름 | `config.py: data_min`, `data_max` |
| 6 | 원본 프레임이 `.npy`가 아님 (예: `.png`, `.nc`) | `refiner/dataset.py: _Base._load_raw_m11` 한 함수만 수정 (반환: m11 `(1,H,W)` 텐서) |
| 7 | 덤프 파일명이 "예측 대상 시각"이 아니라 "발화(issue) 시각" | `refiner/dataset.py: build_items / build_multi_items` — `dt`를 target 시각으로 가정하고 context를 역산하므로, issue 시각이면 `target_dt = dt + offset*간격`으로 보정 후 사용 |
| 8 | train/val 분할이 없고 통짜 폴더만 있음 | `{dumps_root}/train/member_00/`로 심볼릭 링크를 만들거나 폴더를 나눠서 계약에 맞추는 쪽을 권장 (코드 수정 불필요) |

수정 후에는 반드시 `python tests/run_smoke.py`로 모듈이 여전히 통과하는지,
그리고 실데이터 소량(`--max_samples 8` 등)으로 `[items] built=...` 카운트가
0이 아닌지 확인하세요. `built=0`이면 대부분 #1(파일명) 또는 #4(간격/offset) 불일치입니다.

## 학습 데이터가 어떻게 구성되는지 (참고)

Refiner 학습 샘플 하나 = (덤프된 y_blur, 원본에서 읽은 context 5장, 원본 GT target).
context/GT는 **덤프 파일명의 타임스탬프에서 역산**해 원본 디렉토리에서 찾습니다:

```
target_dt       = 덤프 파일명의 시각
last_context    = target_dt - offset × frame_interval
context 5장     = last_context에서 끝나는 연속 5프레임
```

원본에 해당 프레임이 없는 샘플은 자동 스킵되고 개수가 로그에 출력됩니다.
