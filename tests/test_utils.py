"""Plain-assert tests for refiner.utils. Run:
cd /DATA/shinsung/refiner_module && python tests/test_utils.py
"""
import sys, tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from refiner.utils import (ParseDtFromName, NpyToM11, load_tensor_any,
                           load_dump_m11, seed_from_name, ensure_dir)

# ── ParseDtFromName: 3개 기본 포맷 + 정규식 fallback + 실패 시 None ──
assert ParseDtFromName("2021-06-01_0630.pt") == datetime(2021, 6, 1, 6, 30)
assert ParseDtFromName("2021_06_01_0630.npy") == datetime(2021, 6, 1, 6, 30)
assert ParseDtFromName("20210601_0630.pt") == datetime(2021, 6, 1, 6, 30)
assert ParseDtFromName("pred_2021-06-01_0630_final.pt") == datetime(2021, 6, 1, 6, 30)
assert ParseDtFromName("not_a_date.pt") is None

# ── NpyToM11: [min,max] -> [-1,1], NaN -> 0 처리 ──
arr = np.array([[0.0, 13.0], [26.0, np.nan]], dtype=np.float32)
t = NpyToM11(arr, 0.0, 26.0)
assert t.shape == (1, 2, 2)
assert abs(t[0, 0, 0].item() - (-1.0)) < 1e-6
assert abs(t[0, 1, 0].item() - 1.0) < 1e-4
assert abs(t[0, 1, 1].item() - (-1.0)) < 1e-6  # NaN -> 0.0 -> -1

# ── load_tensor_any: .pt/.npy, (H,W) -> (1,H,W) ──
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    torch.save(torch.zeros(4, 4), td / "a.pt")
    np.save(td / "b.npy", np.ones((4, 4), dtype=np.float32))
    assert load_tensor_any(td / "a.pt").shape == (1, 4, 4)
    assert load_tensor_any(td / "b.npy").shape == (1, 4, 4)

    # ── load_dump_m11: m11이면 그대로, raw면 m11 변환 ──
    torch.save(torch.full((1, 4, 4), 13.0), td / "raw.pt")
    x_m11 = load_dump_m11(td / "raw.pt", "raw", 0.0, 26.0)
    assert abs(x_m11[0, 0, 0].item() - 0.0) < 1e-4          # 13/26 -> 0.0
    torch.save(torch.full((1, 4, 4), 0.5), td / "m11.pt")
    x_pass = load_dump_m11(td / "m11.pt", "m11", 0.0, 26.0)
    assert abs(x_pass[0, 0, 0].item() - 0.5) < 1e-6          # 통과

    # ── ensure_dir ──
    ensure_dir(td / "x" / "y")
    assert (td / "x" / "y").is_dir()

# ── seed_from_name: 결정적 + name/base 민감 ──
assert seed_from_name(1, "a") == seed_from_name(1, "a")
assert seed_from_name(1, "a") != seed_from_name(2, "a")
assert seed_from_name(1, "a") != seed_from_name(1, "b")

print("test_utils OK")
