"""Per-pixel MEDIAN ensemble from saved refined members.

Reads {outputs_dir}/refined_member_*/{stem}.pt (K members, each (1,H,W) m11),
writes {outputs_dir}/prediction_median/{stem}.pt = per-pixel median over members.

Median convention: torch.quantile(0.5) → linear interpolation (mean of the two central
values for even K), the standard statistical median. (torch.median would give the lower
of the two middle values for even K.)

usage: python make_median.py <outputs_tH_member_dir>
"""
import sys
from pathlib import Path
import torch

out = Path(sys.argv[1])
mem_dirs = sorted(out.glob("refined_member_*"))
if len(mem_dirs) < 2:
    raise SystemExit(f"need >=2 member dirs, found {len(mem_dirs)} in {out}")
med_dir = out / "prediction_median"
med_dir.mkdir(exist_ok=True)

stems = sorted(p.stem for p in mem_dirs[0].glob("*.pt"))
K = len(mem_dirs)
done = 0
for nm in stems:
    paths = [md / f"{nm}.pt" for md in mem_dirs]
    if not all(p.exists() for p in paths):
        continue
    stack = torch.stack([torch.load(p, weights_only=True).float() for p in paths], dim=0)  # (K,1,H,W)
    med = stack.quantile(0.5, dim=0)  # (1,H,W)
    torch.save(med.contiguous().clone(), med_dir / f"{nm}.pt")
    done += 1
print(f"[median] {out.name}: K={K} members -> {done}/{len(stems)} median tensors in {med_dir}")
