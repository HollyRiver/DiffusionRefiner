"""Synthetic dumps_root + raw_data_dir generator shared by tests.

레이아웃은 실제 입력 계약과 동일:
  {root}/dumps/{split}/member_00../{timestamp}.pt   (m11)
  {root}/raw/{timestamp}.npy                        (raw 0..26)
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

FMT = "%Y-%m-%d_%H%M"


def make_fake_tree(root, splits=("train", "val", "test"), n_members=2,
                   n_targets=4, offset_max=2, hw=64):
    root = Path(root)
    dumps_root = root / "dumps"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # raw 프레임: 2021-06-01 00:00 ~ 12:00, 30분 간격 (25장) — context가 항상 존재하도록
    base = datetime(2021, 6, 1, 0, 0)
    for i in range(25):
        dt = base + timedelta(minutes=30 * i)
        np.save(raw_dir / f"{dt.strftime(FMT)}.npy",
                (np.random.rand(hw, hw).astype("float32")) * 20.0)

    # 예측 덤프: target 06:00부터 30분 간격 n_targets개 (offset<=offset_max 어떤 값에도 context 확보됨)
    t0 = datetime(2021, 6, 1, 6, 0)
    target_names = [(t0 + timedelta(minutes=30 * i)).strftime(FMT) for i in range(n_targets)]
    g = torch.Generator().manual_seed(7)
    for split in splits:
        for k in range(n_members):
            mdir = dumps_root / split / f"member_{k:02d}"
            mdir.mkdir(parents=True, exist_ok=True)
            for name in target_names:
                yb = (torch.rand(1, hw, hw, generator=g) * 2 - 1) * 0.5
                torch.save(yb, mdir / f"{name}.pt")

    return {"dumps_root": dumps_root, "raw_data_dir": raw_dir, "target_names": target_names}


if __name__ == "__main__":
    out = make_fake_tree(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("_fake"))
    print(out)
