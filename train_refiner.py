"""Train the residual-diffusion refiner R for one horizon offset.

  residual_gt = GT_target - y_blur
  x_noisy     = sqrt(abar_t)*residual_gt + sqrt(1-abar_t)*noise
  loss        = MSE(R(x_noisy, ctx, t, day, hour, lead=0, y_blur), residual_gt)

모드:
  --mode ensemble : train/val 멤버들의 '평균' y_blur로 학습 (평균은 work_dir/mean_cache에 1회 구체화)
  --mode member   : 멤버 폴더의 y_blur로 학습. --train_members first(기본)=첫 멤버만,
                    all=전 멤버 union. 검증 멤버는 --val_members 로 독립 지정.

재현성:
  --seed (기본 CFG["seed"]) 가 모델 초기화 / DataLoader shuffle / 확산 t / 노이즈를 모두
  고정한다. 검증도 기본으로 매 epoch 동일한 t·노이즈를 쓰므로 val 곡선이 모델 개선만
  반영한다(옛 동작은 --val_stochastic). 호출 CLI는 work_dir 아래 train_cmd.txt 에 남는다.

K 안전장치:
  --expect_members K 를 주면 발견된 멤버 수가 K와 다를 때 즉시 중단한다. dumps 의 split
  폴더가 심링크인 구성에서는 멤버가 늘어도 에러 없이 K가 바뀌므로 표에 쓰는 실험은 필수.

사용 예 (모듈 폴더에서):
  python train_refiner.py --mode ensemble --offset 1 --gpu 0
  python train_refiner.py --mode member   --offset 1 --gpu 0 \
      --train_members all --val_members first --expect_members 8 --save_every 1
CPU 강제(스모크/디버그): --gpu -1
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from refiner.unet import build_refiner_unet, NoiseSchedule
from refiner.dataset import (discover_members, materialize_mean, build_items,
                             RefinerDataset, refiner_collate)
from refiner.utils import set_seed


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


def _assert_member_count(dirs, expect, split, dumps_root):
    """멤버 수 단정. dumps_root/<split> 이 실제 멤버 폴더를 담은 디렉토리로의 심링크인
    경우(예: analysis/t{h}/dumps_members/train -> base/t{h}/train/members) 그 폴더에
    멤버가 추가되면 K가 조용히 바뀐다. K를 표에 쓰는 실험에서는 반드시 단정할 것."""
    if expect and len(dirs) != expect:
        raise SystemExit(
            f"[abort] --expect_members {expect} 이지만 {dumps_root}/{split} 에서 "
            f"{len(dirs)}개 멤버를 찾았다: {[d.name for d in dirs]}\n"
            f"        K가 바뀌면 결과를 기존 표와 비교할 수 없다. dumps 구성을 확인하거나 "
            f"--expect_members 를 맞춰라 (0=검사 끔).")


def resolve_train_dirs(mode, dumps_root, work_dir, member_glob, train_members="first",
                       val_members="all", expect_members=0):
    """모드별 학습/검증 y_blur 디렉토리 결정.

    member 모드 + train_members="all" 이면 멤버 디렉토리 '리스트'를 반환한다
    (모든 멤버를 union으로 학습/검증 — 한 타임스탬프당 멤버 수만큼 샘플).

    val_members 는 검증 멤버 선택을 학습과 **독립적으로** 정한다:
    "all"=모든 멤버로 검증, "first"=첫 멤버(member_0)만으로 검증
    (에폭당 val 비용 K배 절감, best 선택 품질엔 영향 미미 — val MSE는 수천 샘플
    평균만으로 이미 안정적).

    [2026-09-08 수정] 이전 구현은 `val_members == "all" and train_members == "all"`
    일 때만 "all"을 인정해, 기본값 조합(train_members="first", val_members="all")에서
    조용히 member_0 단독 검증을 했다. help 문구가 약속한 독립성과 어긋났으므로
    조건을 제거했다. 기존 정본 런들은 플래그를 명시했으므로(run_planA.sh:
    `--train_members all --val_members first`) 과거 결과의 해석은 바뀌지 않는다.
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
        _assert_member_count(tdirs, expect_members, "train", dumps_root)
        if train_members == "all":
            train_dir = tdirs
            print(f"[member] training on ALL {len(tdirs)} members (union): "
                  f"{[d.name for d in tdirs]}")
        else:
            train_dir = tdirs[0]
            print(f"[member] training on single member: {train_dir}")
        try:
            vdirs = discover_members(dumps_root / "val", member_glob)
            _assert_member_count(vdirs, expect_members, "val", dumps_root)
            if val_members == "all":
                val_dir = vdirs
                print(f"[member] validating on ALL {len(vdirs)} members (union)")
            else:
                val_dir = vdirs[0]
                print(f"[member] validating on single member: {vdirs[0]}")
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


def run_val(R, val_loader, noise_sched, device, seed=None):
    """검증 잔차 MSE.

    seed가 주어지면 확산 타임스텝 t와 노이즈를 **매 epoch 동일하게** 추출한다.

    [2026-09-08 수정] 이전 구현은 매 epoch 새로 t/노이즈를 뽑았다. 그러면 epoch 간
    val 차이에 샘플링 노이즈가 섞이고, `best_R.pt` 선택이 그 노이즈에 좌우된다
    (모델은 그대로인데 운 좋게 쉬운 t가 뽑힌 epoch이 best가 될 수 있다).
    고정 추출로 바꾸면 val 곡선이 모델 개선만 반영한다. seed=None 이면 옛 동작.
    """
    R.eval()
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
    val_sum, val_n = 0.0, 0
    with torch.no_grad():
        for y_blur, ctx, target, day, hour, _ in val_loader:
            y_blur = y_blur.to(device); ctx = ctx.to(device); target = target.to(device)
            day = day.to(device); hour = hour.to(device)
            B = y_blur.size(0)
            residual = target - y_blur
            if g is None:
                t = noise_sched.sample_t(B)
                noise = torch.randn_like(residual)
            else:
                t = torch.randint(0, noise_sched.T, (B,), device=device, generator=g)
                noise = torch.randn(residual.shape, device=device,
                                    dtype=residual.dtype, generator=g)
            x_t = noise_sched.add_noise(residual, t, noise)
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
    ap.add_argument("--val_members", type=str, default="all", choices=["first", "all"],
                    help="검증 멤버 선택(train_members와 독립): all=모든 멤버(기본), "
                         "first=member_0만 (val 비용 K배↓, best 선택 품질 영향 미미)")
    ap.add_argument("--expect_members", type=int, default=0,
                    help="K 단정: >0 이면 발견된 멤버 수가 이 값과 다를 때 즉시 중단. "
                         "dumps 의 split 폴더가 심링크면 멤버가 늘어도 에러 없이 K가 "
                         "바뀌므로, K를 표에 쓰는 실험은 항상 지정할 것 (0=검사 끔)")
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
    ap.add_argument("--seed", type=int, default=CFG["seed"],
                    help="학습 시드. 모델 초기화 / DataLoader shuffle / 확산 t / 노이즈를 "
                         "모두 고정한다. 같은 시드·같은 설정이면 같은 학습 곡선이 나온다")
    ap.add_argument("--cudnn_deterministic", action="store_true",
                    help="cuDNN 비결정 커널까지 차단 (비트 단위 재현에 근접, 느려짐)")
    ap.add_argument("--val_stochastic", action="store_true",
                    help="검증 t/노이즈를 매 epoch 새로 추출 (2026-09-08 이전 동작). "
                         "기본은 고정 추출이라 val 곡선이 모델 개선만 반영한다")
    ap.add_argument("--save_every", type=int, default=CFG["save_every"],
                    help="N>0이면 매 N epoch마다 ep{에폭}_R.pt 저장 (0=끄기, 기본=CFG)")
    ap.add_argument("--resume_ckpt", type=str, default=None,
                    help="이 체크포인트 가중치에서 이어서 학습 (예: ep45_R.pt). "
                         "--start_epoch, --init_best_val과 함께 사용. 크래시 복구용.")
    ap.add_argument("--start_epoch", type=int, default=0,
                    help="이어서 시작할 0-indexed epoch (예: 45 → 46번째 epoch부터). "
                         "LR 코사인 스케줄과 loss_log가 이 값 기준으로 정렬됨.")
    ap.add_argument("--init_best_val", type=float, default=float("inf"),
                    help="resume 시 best 판정 초기값 (기존 best_R.pt의 val). "
                         "이보다 나빠지면 best_R.pt를 덮어쓰지 않음.")
    args = ap.parse_args()

    device = resolve_device(args.gpu)
    print(f"[device] {device}")
    print(f"[mode]   {args.mode}  offset=t+{args.offset}")

    # 모델 생성/DataLoader 구성 전에 시드를 박아야 초기화와 shuffle까지 고정된다
    set_seed(args.seed, deterministic=args.cudnn_deterministic)
    print(f"[seed]   {args.seed} (cudnn_deterministic={args.cudnn_deterministic}, "
          f"val_sampling={'stochastic' if args.val_stochastic else 'fixed'})")

    save_dir = Path(args.work_dir) / f"refiner_t{args.offset}_{args.mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save]   {save_dir}")
    # 재현을 위해 호출 CLI 전문을 남긴다 (정본 K=8 런은 이게 없어 설정을 역산해야 했다)
    with open(save_dir / "train_cmd.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{' '.join(sys.argv)}\n")

    train_dir, val_dir = resolve_train_dirs(args.mode, args.dumps_root, args.work_dir,
                                            args.member_glob, args.train_members,
                                            args.val_members, args.expect_members)
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
    # shuffle 순서를 전역 RNG에 의존시키지 않고 시드로 못박는다
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(int(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=refiner_collate,
                              drop_last=len(train_ds) >= args.batch_size,
                              generator=shuffle_gen)
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
    if args.resume_ckpt:
        R.load_state_dict(torch.load(args.resume_ckpt, map_location=device, weights_only=True))
        print(f"[resume] loaded {args.resume_ckpt}; start_epoch={args.start_epoch} "
              f"init_best_val={args.init_best_val}")
    noise_sched = NoiseSchedule(T=1000, device=device)

    opt = torch.optim.AdamW(R.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.max_iters_per_epoch or max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    best_val = args.init_best_val
    log_path = save_dir / "loss_log.csv"
    if args.start_epoch > 0 and log_path.exists():
        # resume: 기존 loss_log에서 <= start_epoch 행만 남기고 이어붙임 (중복 방지)
        import csv as _csv
        with open(log_path) as f:
            kept = [r for r in _csv.reader(f)]
        header, body = kept[0], [r for r in kept[1:] if r and int(r[0]) <= args.start_epoch]
        with open(log_path, "w", newline="") as f:
            w = _csv.writer(f); w.writerow(header); w.writerows(body)
        print(f"[resume] loss_log truncated to <= ep{args.start_epoch} ({len(body)} rows)")
    else:
        with open(log_path, "w") as flog:
            flog.write("epoch,train_residual_mse,val_residual_mse,lr,time_s\n")

    gstep = args.start_epoch * steps_per_epoch
    print(f"[train]  epochs={args.epochs} bs={args.batch_size} steps/ep={steps_per_epoch} "
          f"start_epoch={args.start_epoch}")
    for epoch in range(args.start_epoch, args.epochs):
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
        val_seed = None if args.val_stochastic else int(args.seed) + 1
        val_avg = (run_val(R, val_loader, noise_sched, device, seed=val_seed)
                   if len(val_ds) > 0 else float("nan"))
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

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(R.state_dict(), save_dir / f"ep{epoch+1}_R.pt")
            print(f"  * saved ep{epoch+1}_R.pt")

    torch.save(R.state_dict(), save_dir / "last_R.pt")
    print(f"[done]   best={best_val:.6f}  ckpts in {save_dir}")


if __name__ == "__main__":
    main()
