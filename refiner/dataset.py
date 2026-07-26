"""Dataset wiring for the portable refiner module.

- discover_members / materialize_mean: 멤버 서브폴더 계약 처리 (앙상블 평균은 학습 전 1회 구체화)
- build_items: 예측 덤프 파일명(=target 타임스탬프)에서 GT target + context 5장을 역산 페어링
- RefinerDataset: 학습/검증용 (y_blur, ctx, target, day, hour, name)
- MultiRefinerDataset: 추론용 — 멤버 K개의 y_blur를 한 번에 (K,1,H,W)로 반환

[적응 포인트 #2] 멤버 폴더 명명이 member_00 형태가 아니면: config.CFG["member_glob"] 수정
  (예: "seed*"). 그래도 안 맞으면 discover_members 한 함수만 고치면 된다.
[적응 포인트 #6] 원본 프레임이 .npy가 아니면: _Base._load_raw_m11 한 함수만 고치면 된다.
[적응 포인트 #7] 덤프 파일명이 target 시각이 아니라 발화(issue) 시각이면:
  build_items/build_multi_items 의 `dt`가 '예측 대상 시각'이라는 가정을 고쳐야 한다
  (issue 시각이라면 target_dt = dt + offset*frame_min 로 바꾸고 context 역산도 그에 맞춤).
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (ParseDtFromName, NpyToM11, load_tensor_any, load_dump_m11,
                    ensure_dir)


def discover_members(split_dir, member_glob="member_*"):
    """split 디렉토리 아래 멤버 서브폴더들을 정렬해 반환."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"split dir not found: {split_dir}\n"
            f"expected layout: {split_dir}/member_00/<timestamp>.pt")
    dirs = sorted(d for d in split_dir.glob(member_glob) if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"no member dirs matching {member_glob!r} under {split_dir}\n"
            f"expected e.g. {split_dir}/member_00/  (README '적응 가이드' #2 참고)")
    return dirs


def _dump_names(d):
    return {p.name for p in Path(d).glob("*.pt")} | {p.name for p in Path(d).glob("*.npy")}


def _common_names(member_dirs, verbose=True, what="mean"):
    sets = [_dump_names(d) for d in member_dirs]
    common = sorted(set.intersection(*sets))
    union_n = len(set.union(*sets))
    if len(common) < union_n and verbose:
        print(f"[{what}] WARNING: member file sets differ — "
              f"intersection={len(common)} union={union_n}; using intersection only")
    return common


def materialize_mean(split_dir, cache_dir, member_glob="member_*", verbose=True):
    """멤버 평균을 cache_dir에 1회 구체화(.pt). 이미 있는 파일은 스킵.

    값 변환은 하지 않는다(평균만) — m11/raw 변환은 Dataset(load_dump_m11)에서 일원화.
    산술평균은 affine 변환과 교환 가능하므로 순서는 결과에 영향 없음.
    """
    member_dirs = discover_members(split_dir, member_glob)
    cache_dir = Path(cache_dir)
    ensure_dir(cache_dir)
    common = _common_names(member_dirs, verbose=verbose, what="mean")
    if not common:
        raise RuntimeError(f"no common dump files across members under {split_dir}")
    n_write = n_skip = 0
    for fname in common:
        out = cache_dir / (Path(fname).stem + ".pt")
        if out.exists():
            n_skip += 1
            continue
        acc = None
        for d in member_dirs:
            x = load_tensor_any(d / fname)
            acc = x if acc is None else acc + x
        torch.save(acc / float(len(member_dirs)), out)
        n_write += 1
    if verbose:
        print(f"[mean] {split_dir} K={len(member_dirs)} -> {cache_dir}: "
              f"wrote={n_write} skipped={n_skip}")
    return cache_dir


def _context_paths_for(dt, raw_dir, offset, num_context, frame_min, filename_format):
    """target 시각 dt에서 context 프레임 경로들을 역산 (마지막 context = dt - offset*frame_min)."""
    last_ctx = dt - timedelta(minutes=frame_min * offset)
    ctx_dts = [last_ctx - timedelta(minutes=frame_min * (num_context - 1 - j))
               for j in range(num_context)]
    return [Path(raw_dir) / f"{c.strftime(filename_format)}.npy" for c in ctx_dts]


def build_items(y_blur_dir, raw_dir, offset, num_context=5, frame_min=30,
                filename_format="%Y-%m-%d_%H%M", require_target=True, verbose=True):
    """단일 y_blur 디렉토리를 raw target/context와 페어링 (학습/검증용)."""
    y_blur_dir = Path(y_blur_dir)
    raw_dir = Path(raw_dir)
    items = []
    n_miss_ctx = n_miss_tgt = n_no_dt = 0
    files = sorted(list(y_blur_dir.glob("*.pt")) + list(y_blur_dir.glob("*.npy")))
    for f in files:
        dt = ParseDtFromName(f.name)
        if dt is None:
            n_no_dt += 1
            continue
        target_path = raw_dir / f"{dt.strftime(filename_format)}.npy"
        if require_target and not target_path.exists():
            n_miss_tgt += 1
            continue
        ctx_paths = _context_paths_for(dt, raw_dir, offset, num_context, frame_min,
                                       filename_format)
        if not all(p.exists() for p in ctx_paths):
            n_miss_ctx += 1
            continue
        items.append({
            "name": f.stem,
            "target_dt": dt,
            "y_blur_path": str(f),
            "context_paths": [str(p) for p in ctx_paths],
            "target_path": str(target_path),
        })
    if verbose:
        print(f"[items] {y_blur_dir}: built={len(items)} "
              f"miss_ctx={n_miss_ctx} miss_tgt={n_miss_tgt} no_dt={n_no_dt} (offset={offset})")
    return items


def build_multi_items(member_dirs, raw_dir, offset, num_context=5, frame_min=30,
                      filename_format="%Y-%m-%d_%H%M", verbose=True):
    """멤버 K개 디렉토리의 교집합 파일명을 context와 페어링 (추론용, GT 불필요)."""
    member_dirs = [Path(d) for d in member_dirs]
    raw_dir = Path(raw_dir)
    common = _common_names(member_dirs, verbose=verbose, what="multi")
    items = []
    n_miss_ctx = n_no_dt = 0
    for fname in common:
        dt = ParseDtFromName(fname)
        if dt is None:
            n_no_dt += 1
            continue
        ctx_paths = _context_paths_for(dt, raw_dir, offset, num_context, frame_min,
                                       filename_format)
        if not all(p.exists() for p in ctx_paths):
            n_miss_ctx += 1
            continue
        items.append({
            "name": Path(fname).stem,
            "target_dt": dt,
            "y_blur_paths": [str(d / fname) for d in member_dirs],
            "context_paths": [str(p) for p in ctx_paths],
        })
    if verbose:
        print(f"[multi] K={len(member_dirs)}: built={len(items)} "
              f"miss_ctx={n_miss_ctx} no_dt={n_no_dt} (offset={offset})")
    return items


class _Base(Dataset):
    def __init__(self, items, data_min, data_max, dump_value_range="m11"):
        self.items = items
        self.data_min = float(data_min)
        self.data_max = float(data_max)
        self.dump_value_range = str(dump_value_range)

    def __len__(self):
        return len(self.items)

    def _load_raw_m11(self, path):
        # [적응 포인트 #6] 원본 프레임 로더 — .npy가 아니면 여기만 수정
        return NpyToM11(np.load(path), self.data_min, self.data_max)  # (1,H,W)

    def _load_dump(self, path):
        return load_dump_m11(path, self.dump_value_range, self.data_min, self.data_max)

    @staticmethod
    def _day_hour(dt):
        return float(dt.timetuple().tm_yday), float(dt.hour) + float(dt.minute) / 60.0


class RefinerDataset(_Base):
    """학습/검증용: (y_blur(1,H,W), ctx(T,1,H,W), target(1,H,W), day, hour, name)."""

    def __init__(self, items, data_min, data_max, dump_value_range="m11", load_target=True):
        super().__init__(items, data_min, data_max, dump_value_range)
        self.load_target = bool(load_target)

    def __getitem__(self, idx):
        s = self.items[idx]
        y_blur = self._load_dump(s["y_blur_path"])
        ctx = torch.stack([self._load_raw_m11(p) for p in s["context_paths"]], dim=0)
        target = self._load_raw_m11(s["target_path"]) if self.load_target \
            else torch.zeros_like(y_blur)
        day, hour = self._day_hour(s["target_dt"])
        return y_blur, ctx, target, day, hour, s["name"]


def refiner_collate(batch):
    y_blur = torch.stack([b[0] for b in batch], dim=0)
    ctx = torch.stack([b[1] for b in batch], dim=0)
    target = torch.stack([b[2] for b in batch], dim=0)
    day = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    hour = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    names = [b[5] for b in batch]
    return y_blur, ctx, target, day, hour, names


class MultiRefinerDataset(_Base):
    """추론용: (y_all(K,1,H,W), ctx(T,1,H,W), day, hour, name). GT 불필요."""

    def __getitem__(self, idx):
        s = self.items[idx]
        y_all = torch.stack([self._load_dump(p) for p in s["y_blur_paths"]], dim=0)
        ctx = torch.stack([self._load_raw_m11(p) for p in s["context_paths"]], dim=0)
        day, hour = self._day_hour(s["target_dt"])
        return y_all, ctx, day, hour, s["name"]


def multi_collate(batch):
    y_all = torch.stack([b[0] for b in batch], dim=0)
    ctx = torch.stack([b[1] for b in batch], dim=0)
    day = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    hour = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    names = [b[4] for b in batch]
    return y_all, ctx, day, hour, names
