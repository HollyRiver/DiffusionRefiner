"""Central config for the portable refiner module.

모든 경로/데이터 계약/하이퍼파라미터가 여기 모여 있다. CLI 인자가 지정되면
CLI가 우선하고, 지정하지 않으면 이 CFG 값이 기본값으로 쓰인다.
다른 환경으로 옮길 때 최소한 dumps_root / raw_data_dir / work_dir 를 바꿔야 한다.
데이터 규칙이 다른 환경에 대한 수정 지점은 README.md의 "적응 가이드" 참고.
"""

CFG = {
    # ── 경로 (환경마다 반드시 확인) ───────────────────────────────────────
    # {dumps_root}/{split}/member_XX/{timestamp}.pt  (split: train/val/test)
    "dumps_root": "/path/to/refiner_inputs",
    # 원본 프레임 {timestamp}.npy (context 5장 + 학습 시 GT target)
    "raw_data_dir": "/DATA/solar",
    # 평균 캐시 / 체크포인트 / 최종 출력이 담기는 작업 폴더
    "work_dir": "./work",

    # ── 데이터 계약 ─────────────────────────────────────────────────────
    "filename_format": "%Y-%m-%d_%H%M",   # 타임스탬프 파일명 규칙
    "member_glob": "member_*",            # 멤버 서브폴더 패턴
    "dump_value_range": "m11",            # 덤프 값 범위: "m11" | "raw"
    "data_min": 0.0,
    "data_max": 26.347150802612,
    "frame_interval_minutes": 30,
    "num_context": 5,

    # ── 학습 하이퍼파라미터 (baseline_ver03 refiner 레시피) ─────────────
    "epochs": 30,
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 10,
    "min_lr": 1e-6,
    "grad_clip": 1.0,
    "num_workers": 4,
    # 매 N epoch마다 ep{N}_R.pt 체크포인트 저장 (0=끄기). 개당 ~9MB.
    # 중간-epoch 비교/재현이 나중에 필요해도 재학습 없이 가능하도록 기본 ON.
    "save_every": 1,

    # ── 모델 (baseline_ver03 refiner와 동일) ────────────────────────────
    "input_size": 512,
    "base_ch": 32,
    "num_levels": 3,
    "ch_mult": (1, 2, 4),
    "num_res_blocks": 2,

    # ── 추론 ────────────────────────────────────────────────────────────
    "ddim_steps": 40,
    "seed": 1253,   # 샘플별 시드 파생의 base (재실행 시 동일 출력 보장)
}
