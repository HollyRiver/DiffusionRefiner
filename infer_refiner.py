"""Refine test dumps into final predictions (.pt). 메트릭 없음 — 순수 추론.

모드:
  --mode ensemble : 멤버들의 평균 y_blur를 1회 보정 -> 최종 출력
  --mode member   : 멤버 K개를 각각 보정 -> 보정 결과 평균 -> 최종 출력
                    (--save_members 로 멤버별 보정본도 저장)

재현성: 샘플명+멤버 인덱스에서 파생한 고정 시드로 DDIM 초기 노이즈를 만들므로
(eta=0 결정적 DDIM), 재실행해도 동일한 출력이 나온다.

사용 예 (모듈 폴더에서):
  python infer_refiner.py --mode ensemble --offset 1 --gpu 0
  python infer_refiner.py --mode member   --offset 1 --gpu 0 --save_members
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CFG
from refiner.unet import build_refiner_unet, NoiseSchedule, r_ddim_refine
from refiner.dataset import (discover_members, build_multi_items,
                             MultiRefinerDataset, multi_collate)
from refiner.utils import seed_from_name, ensure_dir
from train_refiner import resolve_device


def make_init_noise(names, k, H, W, base_seed):
    """샘플명 x 멤버 인덱스 -> 결정적 초기 노이즈 (B,1,H,W). CPU 생성 후 이동(디바이스 무관 재현)."""
    outs = []
    for name in names:
        g = torch.Generator().manual_seed(seed_from_name(base_seed + 1000 * k, name))
        outs.append(torch.randn(1, H, W, generator=g))
    return torch.stack(outs, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["ensemble", "member"])
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--gpu", type=int, default=0, help="-1 = CPU 강제")
    ap.add_argument("--dumps_root", type=str, default=CFG["dumps_root"])
    ap.add_argument("--raw_data_dir", type=str, default=CFG["raw_data_dir"])
    ap.add_argument("--work_dir", type=str, default=CFG["work_dir"])
    ap.add_argument("--member_glob", type=str, default=CFG["member_glob"])
    ap.add_argument("--dump_value_range", type=str, default=CFG["dump_value_range"],
                    choices=["m11", "raw"])
    ap.add_argument("--ddim_steps", type=int, default=CFG["ddim_steps"])
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    ap.add_argument("--num_workers", type=int, default=CFG["num_workers"])
    ap.add_argument("--R_ckpt", type=str, default=None,
                    help="기본: {work_dir}/refiner_t{N}_{mode}/best_R.pt")
    ap.add_argument("--save_members", action="store_true",
                    help="(mode=member) 멤버별 보정본도 저장")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    device = resolve_device(args.gpu)
    print(f"[device] {device}")
    print(f"[mode]   {args.mode}  offset=t+{args.offset}  split={args.split}  "
          f"ddim_steps={args.ddim_steps}")

    # ── R 로드 ────────────────────────────────────────────────────────────
    R = build_refiner_unet(
        input_size=CFG["input_size"], num_context=CFG["num_context"],
        base_ch=CFG["base_ch"], num_levels=CFG["num_levels"],
        ch_mult=tuple(CFG["ch_mult"]), num_res_blocks=CFG["num_res_blocks"],
        num_leads_out=1,
    ).to(device)
    ckpt = args.R_ckpt or str(Path(args.work_dir) / f"refiner_t{args.offset}_{args.mode}"
                              / "best_R.pt")
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"R checkpoint not found: {ckpt}\n"
            f"train_refiner.py --mode {args.mode} --offset {args.offset} 를 먼저 실행했는지, "
            f"--work_dir 가 학습 때와 같은지 확인.")
    R.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    R.eval()
    print(f"[R]      loaded {ckpt}")
    noise_sched = NoiseSchedule(T=1000, device=device)

    # ── 데이터 ────────────────────────────────────────────────────────────
    member_dirs = discover_members(Path(args.dumps_root) / args.split, args.member_glob)
    print(f"[data]   {len(member_dirs)} member dir(s): {[d.name for d in member_dirs]}")
    items = build_multi_items(member_dirs, args.raw_data_dir, args.offset,
                              num_context=CFG["num_context"],
                              frame_min=CFG["frame_interval_minutes"],
                              filename_format=CFG["filename_format"])
    if len(items) == 0:
        raise RuntimeError(
            f"No items for offset={args.offset} split={args.split}. "
            f"raw_data_dir={args.raw_data_dir} 의 context 프레임/파일명 규칙 확인 "
            f"(README '적응 가이드').")
    if args.max_samples is not None:
        items = items[:args.max_samples]
    ds = MultiRefinerDataset(items, CFG["data_min"], CFG["data_max"], args.dump_value_range)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=multi_collate,
                        drop_last=False)

    out_dir = Path(args.work_dir) / f"outputs_t{args.offset}_{args.mode}"
    final_dir = out_dir / "prediction"
    ensure_dir(final_dir)
    member_out_dirs = []
    if args.mode == "member" and args.save_members:
        K = len(member_dirs)
        member_out_dirs = [out_dir / f"refined_member_{k:02d}" for k in range(K)]
        for d in member_out_dirs:
            ensure_dir(d)

    # ── 보정 루프 ─────────────────────────────────────────────────────────
    n_done = 0
    with torch.no_grad():
        for y_all, ctx, day, hour, names in loader:
            y_all = y_all.to(device); ctx = ctx.to(device)
            day = day.to(device); hour = hour.to(device)
            B, K = y_all.shape[:2]
            H, W = y_all.shape[-2:]
            lead_idx = torch.zeros(B, dtype=torch.long, device=device)

            if args.mode == "ensemble":
                y_list = [y_all.mean(dim=1)]   # 평균을 1회 보정
            else:
                y_list = [y_all[:, k] for k in range(K)]  # 멤버 각각 보정

            refined = []
            for k, y_blur in enumerate(y_list):
                init = make_init_noise(names, k, H, W, args.seed).to(device)
                ys = r_ddim_refine(R, ctx, y_blur, day, hour, lead_idx, noise_sched,
                                   num_steps=args.ddim_steps,
                                   init_noise=init).clamp(-1.0, 1.0)
                refined.append(ys)
                if member_out_dirs:
                    ys_cpu = ys.detach().cpu()
                    for b in range(B):
                        # .clone() 필수: 배치 텐서의 view를 그대로 save하면 전체 storage가
                        # 함께 저장돼 파일이 수십 배 커진다 (eval_refiner에서 실측된 함정).
                        torch.save(ys_cpu[b].clone(), member_out_dirs[k] / f"{names[b]}.pt")

            final = torch.stack(refined, dim=0).mean(dim=0).clamp(-1.0, 1.0).detach().cpu()
            for b in range(B):
                torch.save(final[b].clone(), final_dir / f"{names[b]}.pt")
            n_done += B
            if n_done % 100 < B:
                print(f"  [{n_done}/{len(ds)}] refined")

    print(f"[done]   {n_done} final predictions -> {final_dir}")
    if member_out_dirs:
        print(f"         member outputs -> {out_dir}/refined_member_XX/")


if __name__ == "__main__":
    main()
