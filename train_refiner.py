"""Train the residual-diffusion refiner R for one horizon offset.

  residual_gt = GT_target - y_blur
  x_noisy     = sqrt(abar_t)*residual_gt + sqrt(1-abar_t)*noise
  loss        = MSE(R(x_noisy, ctx, t, day, hour, lead=0, y_blur), residual_gt)

모드:
  --mode ensemble : train/val 멤버들의 '평균' y_blur로 학습 (평균은 work_dir/mean_cache에 1회 구체화)
  --mode member   : 첫 번째 멤버 폴더(정렬순, 예: member_00)의 단일 y_blur로 학습

사용 예 (모듈 폴더에서):
  python train_refiner.py --mode ensemble --offset 1 --gpu 0
  python train_refiner.py --mode member   --offset 1 --gpu 0
CPU 강제(스모크/디버그): --gpu -1
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from refiner.unet import build_refiner_unet, NoiseSchedule
from refiner.dataset import (discover_members, materialize_mean, build_items,
                             RefinerDataset, refiner_collate)


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def resolve_device(gpu):
    """--gpu >= 0 인데 CUDA 불가면 재시도 후 하드 abort (침묵 CPU 폴백 금지)."""
    if gpu < 0:
        return torch.device("cpu")
    for _try in range(5):
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu}")
            _ = torch.empty(1, device=device)  # CUDA 컨텍스트 조기 고정
            return device
        print(f"[device] CUDA not ready (attempt {_try+1}/5), retrying...", flush=True)
        time.sleep(10)
    raise RuntimeError(
        f"--gpu {gpu} requested but torch.cuda.is_available()=False after retries. "
        "Refusing to train on CPU (would take days). Use --gpu -1 to force CPU explicitly.")


def resolve_train_dirs(mode, dumps_root, work_dir, member_glob, train_members="first"):
    """모드별 학습/검증 y_blur 디렉토리 결정.

    member 모드 + train_members="all" 이면 멤버 디렉토리 '리스트'를 반환한다
    (모든 멤버를 union으로 학습/검증 — 한 타임스탬프당 멤버 수만큼 샘플).
    """
    dumps_root = Path(dumps_root)
    if mode == "ensemble":
        train_dir = materialize_mean(dumps_root / "train", Path(work_dir) / "mean_cache" / "train",
                                     member_glob)
        try:
            val_dir = materialize_mean(dumps_root / "val", Path(work_dir) / "mean_cache" / "val",
                                       member_glob)
        except FileNotFoundError as e:
            print(f"[warn] no val dumps ({e}); validation skipped, best==train")
            val_dir = None
    else:  # member
        tdirs = discover_members(dumps_root / "train", member_glob)
        if train_members == "all":
            train_dir = tdirs
            print(f"[member] training on ALL {len(tdirs)} members (union)")
        else:
            train_dir = tdirs[0]
            print(f"[member] training on single member: {train_dir}")
        try:
            vdirs = discover_members(dumps_root / "val", member_glob)
            val_dir = vdirs if train_members == "all" else vdirs[0]
        except FileNotFoundError as e:
            print(f"[warn] no val dumps ({e}); validation skipped, best==train")
            val_dir = None
    return train_dir, val_dir


def build_items_any(dir_or_dirs, raw_dir, offset, **kw):
    """단일 디렉토리 또는 디렉토리 리스트(멤버 union)를 items로 페어링."""
    dirs = dir_or_dirs if isinstance(dir_or_dirs, (list, tuple)) else [dir_or_dirs]
    items = []
    for d in dirs:
        items += build_items(d, raw_dir, offset, **kw)
    return items


def run_val(R, val_loader, noise_sched, device):
    R.eval()
    val_sum, val_n = 0.0, 0
    with torch.no_grad():
        for y_blur, ctx, target, day, hour, _ in val_loader:
            y_blur = y_blur.to(device); ctx = ctx.to(device); target = target.to(device)
            day = day.to(device); hour = hour.to(device)
            B = y_blur.size(0)
            residual = target - y_blur
            t = noise_sched.sample_t(B)
            x_t = noise_sched.add_noise(residual, t, torch.randn_like(residual))
            lead_idx = torch.zeros(B, dtype=torch.long, device=device)
            pred = R(x_t, ctx, t, day, hour, lead_idx, y_blur)
            val_sum += F.mse_loss(pred, residual).item() * B
            val_n += B
    return val_sum / max(1, val_n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["ensemble", "member"])
    ap.add_argument("--offset", type=int, required=True, help="horizon offset N (프레임 단위)")
    ap.add_argument("--gpu", type=int, default=0, help="-1 = CPU 강제")
    ap.add_argument("--dumps_root", type=str, default=CFG["dumps_root"])
    ap.add_argument("--raw_data_dir", type=str, default=CFG["raw_data_dir"])
    ap.add_argument("--work_dir", type=str, default=CFG["work_dir"])
    ap.add_argument("--member_glob", type=str, default=CFG["member_glob"])
    ap.add_argument("--train_members", type=str, default="first", choices=["first", "all"],
                    help="member 모드 학습 데이터: first=첫 멤버만, all=모든 멤버 union "
                         "(타임스탬프당 멤버 수만큼 학습 샘플; val도 동일하게 확장)")
    ap.add_argument("--dump_value_range", type=str, default=CFG["dump_value_range"],
                    choices=["m11", "raw"])
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--weight_decay", type=float, default=CFG["weight_decay"])
    ap.add_argument("--warmup_epochs", type=int, default=CFG["warmup_epochs"])
    ap.add_argument("--min_lr", type=float, default=CFG["min_lr"])
    ap.add_argument("--num_workers", type=int, default=CFG["num_workers"])
    ap.add_argument("--max_iters_per_epoch", type=int, default=None)
    args = ap.parse_args()

    device = resolve_device(args.gpu)
    print(f"[device] {device}")
    print(f"[mode]   {args.mode}  offset=t+{args.offset}")

    save_dir = Path(args.work_dir) / f"refiner_t{args.offset}_{args.mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save]   {save_dir}")

    train_dir, val_dir = resolve_train_dirs(args.mode, args.dumps_root, args.work_dir,
                                            args.member_glob, args.train_members)
    kw = dict(num_context=CFG["num_context"], frame_min=CFG["frame_interval_minutes"],
              filename_format=CFG["filename_format"])
    train_items = build_items_any(train_dir, args.raw_data_dir, args.offset, **kw)
    val_items = build_items_any(val_dir, args.raw_data_dir, args.offset, **kw) if val_dir else []
    if len(train_items) == 0:
        raise RuntimeError(
            f"No train items for offset={args.offset} in {train_dir}. "
            f"raw_data_dir={args.raw_data_dir} 안에 context/GT 프레임이 있는지, "
            f"파일명 규칙이 맞는지 확인 (README '적응 가이드').")
    if len(val_items) == 0:
        print("[warn]   0 val items — validation skipped, best==train")
    print(f"[data]   train={len(train_items)} val={len(val_items)}")

    dmin, dmax = CFG["data_min"], CFG["data_max"]
    train_ds = RefinerDataset(train_items, dmin, dmax, args.dump_value_range)
    val_ds = RefinerDataset(val_items, dmin, dmax, args.dump_value_range)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=refiner_collate,
                              drop_last=len(train_ds) >= args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(0, args.num_workers // 2),
                            collate_fn=refiner_collate, drop_last=False)

    R = build_refiner_unet(
        input_size=CFG["input_size"], num_context=CFG["num_context"],
        base_ch=CFG["base_ch"], num_levels=CFG["num_levels"],
        ch_mult=tuple(CFG["ch_mult"]), num_res_blocks=CFG["num_res_blocks"],
        num_leads_out=1,
    ).to(device)
    print(f"[R]      params={sum(p.numel() for p in R.parameters()):,} "
          f"in_channels={R.in_channels}")
    noise_sched = NoiseSchedule(T=1000, device=device)

    opt = torch.optim.AdamW(R.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.max_iters_per_epoch or max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    best_val = float("inf")
    log_path = save_dir / "loss_log.csv"
    with open(log_path, "w") as flog:
        flog.write("epoch,train_residual_mse,val_residual_mse,lr,time_s\n")

    gstep = 0
    print(f"[train]  epochs={args.epochs} bs={args.batch_size} steps/ep={steps_per_epoch}")
    for epoch in range(args.epochs):
        R.train()
        ep_sum, ep_n = 0.0, 0
        t0 = time.time()
        for it, (y_blur, ctx, target, day, hour, _) in enumerate(train_loader):
            y_blur = y_blur.to(device); ctx = ctx.to(device); target = target.to(device)
            day = day.to(device); hour = hour.to(device)
            B = y_blur.size(0)

            residual = target - y_blur
            t = noise_sched.sample_t(B)
            x_t = noise_sched.add_noise(residual, t, torch.randn_like(residual))
            lead_idx = torch.zeros(B, dtype=torch.long, device=device)

            pred = R(x_t, ctx, t, day, hour, lead_idx, y_blur)
            loss = F.mse_loss(pred, residual)

            lr = cosine_lr(gstep, total_steps, warmup_steps, args.lr, args.min_lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(R.parameters(), max_norm=CFG["grad_clip"])
            opt.step()

            ep_sum += loss.item() * B
            ep_n += B
            gstep += 1
            if args.max_iters_per_epoch and it + 1 >= args.max_iters_per_epoch:
                break

        train_avg = ep_sum / max(1, ep_n)
        val_avg = run_val(R, val_loader, noise_sched, device) if len(val_ds) > 0 else float("nan")
        dt = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]
        print(f"[ep {epoch+1}/{args.epochs}] train_residual_mse={train_avg:.6f} "
              f"val={val_avg:.6f} lr={cur_lr:.2e} time={dt:.1f}s")
        with open(log_path, "a") as flog:
            flog.write(f"{epoch+1},{train_avg:.6f},{val_avg:.6f},{cur_lr:.6e},{dt:.2f}\n")

        crit = val_avg if len(val_ds) > 0 else train_avg
        if crit < best_val:
            best_val = crit
            torch.save(R.state_dict(), save_dir / "best_R.pt")
            print(f"  * saved best_R.pt (crit={crit:.6f})")

    torch.save(R.state_dict(), save_dir / "last_R.pt")
    print(f"[done]   best={best_val:.6f}  ckpts in {save_dir}")


if __name__ == "__main__":
    main()
