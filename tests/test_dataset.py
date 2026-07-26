"""Plain-assert tests for refiner.dataset. Run:
cd /DATA/shinsung/refiner_module && python tests/test_dataset.py
"""
import shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refiner.dataset import (discover_members, materialize_mean, build_items,
                             build_multi_items, RefinerDataset, refiner_collate,
                             MultiRefinerDataset, multi_collate)
from tests.make_fake_data import make_fake_tree

TMP = Path(__file__).parent / "_dataset_tmp"
if TMP.exists():
    shutil.rmtree(TMP)
info = make_fake_tree(TMP, n_members=2, n_targets=4, hw=32)
dumps, raw = info["dumps_root"], info["raw_data_dir"]

# ── discover_members: 정렬된 2개, 없는 split은 에러 ──
members = discover_members(dumps / "train")
assert [m.name for m in members] == ["member_00", "member_01"]
try:
    discover_members(dumps / "nope")
    raise AssertionError("should have raised")
except FileNotFoundError:
    pass

# ── materialize_mean: 평균값 정확성 + 스킵 동작 ──
cache = TMP / "mean_cache" / "train"
out_dir = materialize_mean(dumps / "train", cache)
name0 = info["target_names"][0]
m0 = torch.load(dumps / "train" / "member_00" / f"{name0}.pt", weights_only=True)
m1 = torch.load(dumps / "train" / "member_01" / f"{name0}.pt", weights_only=True)
mean = torch.load(out_dir / f"{name0}.pt", weights_only=True)
assert torch.allclose(mean, (m0 + m1) / 2, atol=1e-6)
# 재호출 시 전부 스킵돼도 같은 결과
out_dir2 = materialize_mean(dumps / "train", cache)
assert out_dir2 == out_dir

# ── 교집합 처리: member_01에서 파일 하나 제거 -> 교집합 3개만 ──
missing = dumps / "train" / "member_01" / f"{info['target_names'][3]}.pt"
missing.unlink()
cache2 = TMP / "mean_cache_intersect"
out3 = materialize_mean(dumps / "train", cache2)
assert len(list(out3.glob("*.pt"))) == 3

# ── build_items: 4개 페어링, offset=1 context 역산 확인 ──
items = build_items(dumps / "val" / "member_00", raw, offset=1)
assert len(items) == 4
it = items[0]
assert len(it["context_paths"]) == 5
assert it["target_path"].endswith(f"{it['name']}.npy")
# 마지막 context = target - 30분
from datetime import timedelta
last_ctx = Path(it["context_paths"][-1]).stem
assert last_ctx == (it["target_dt"] - timedelta(minutes=30)).strftime("%Y-%m-%d_%H%M")

# ── build_items: raw 프레임 밖 타임스탬프는 스킵 ──
extra = dumps / "val" / "member_00" / "2030-01-01_0600.pt"
torch.save(torch.zeros(1, 32, 32), extra)
items2 = build_items(dumps / "val" / "member_00", raw, offset=1)
assert len(items2) == 4  # 스킵됨
extra.unlink()

# ── RefinerDataset + collate ──
ds = RefinerDataset(items, 0.0, 26.0)
y, ctx, tgt, day, hour, name = ds[0]
assert y.shape == (1, 32, 32) and ctx.shape == (5, 1, 32, 32) and tgt.shape == (1, 32, 32)
assert tgt.min() >= -1.0 and tgt.max() <= 1.0
yb, cb, tb, db, hb, names = refiner_collate([ds[0], ds[1]])
assert yb.shape == (2, 1, 32, 32) and cb.shape == (2, 5, 1, 32, 32) and tb.shape == (2, 1, 32, 32)

# ── build_multi_items + MultiRefinerDataset (test split, K=2) ──
mdirs = discover_members(dumps / "test")
mitems = build_multi_items(mdirs, raw, offset=1)
assert len(mitems) == 4 and len(mitems[0]["y_blur_paths"]) == 2
mds = MultiRefinerDataset(mitems, 0.0, 26.0)
ya, ctx, day, hour, name = mds[0]
assert ya.shape == (2, 1, 32, 32)
yab, cab, dab, hab, mnames = multi_collate([mds[0], mds[1]])
assert yab.shape == (2, 2, 1, 32, 32)

shutil.rmtree(TMP)
print("test_dataset OK")
