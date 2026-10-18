"""Aggregate UNREFINED base DAD ensemble (mean + per-pixel median) from K member dumps.

Reads member_0..member_{K-1} (numeric order) from a dumps test dir, writes:
  {out}/prediction/{stem}.pt         = per-pixel MEAN over K base members
  {out}/prediction_median/{stem}.pt  = per-pixel MEDIAN (quantile 0.5) over K base members

usage: python make_base_ens.py <members_test_dir> <out_dir> <K>
"""
import sys
from pathlib import Path
import torch

memroot = Path(sys.argv[1])   # e.g. analysis/tH/dumps_members12/test
out = Path(sys.argv[2])
K = int(sys.argv[3])
memdirs = [memroot / f"member_{k}" for k in range(K)]   # numeric order, member_0..K-1
for md in memdirs:
    if not md.is_dir():
        raise SystemExit(f"missing member dir: {md}")
(out / "prediction").mkdir(parents=True, exist_ok=True)
(out / "prediction_median").mkdir(parents=True, exist_ok=True)

stems = sorted(p.stem for p in memdirs[0].glob("*.pt"))
done = 0
for nm in stems:
    paths = [md / f"{nm}.pt" for md in memdirs]
    if not all(p.exists() for p in paths):
        continue
    st = torch.stack([torch.load(p, weights_only=True).float() for p in paths], dim=0)  # (K,1,H,W)
    torch.save(st.mean(dim=0).contiguous().clone(), out / "prediction" / f"{nm}.pt")
    torch.save(st.quantile(0.5, dim=0).contiguous().clone(), out / "prediction_median" / f"{nm}.pt")
    done += 1
print(f"[base_ens] {out.name} K={K}: {done}/{len(stems)} mean+median tensors")
