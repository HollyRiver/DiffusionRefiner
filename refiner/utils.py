"""Minimal utilities for the portable refiner module (no external deps beyond torch/numpy).

ParseDtFromName / NpyToM11 은 baseline_ver03.data 에서, load_tensor_any / seed_from_name 은
baseline_ver03.utils 에서 이식 (CFG 의존 제거).

[적응 포인트 #1] 파일명 규칙이 다르면: DT_FORMATS 에 포맷을 추가하거나
ParseDtFromName 의 정규식을 수정한다. config.CFG["filename_format"] 도 함께 맞출 것
(그쪽은 '타임스탬프 -> 원본 프레임 파일명' 생성에 쓰인다).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

DT_FORMATS = ["%Y-%m-%d_%H%M", "%Y_%m_%d_%H%M", "%Y%m%d_%H%M"]


def ParseDtFromName(name, formats=None):
    """파일명(stem)에서 타임스탬프를 파싱. 실패 시 None."""
    stem = Path(name).stem
    for fmt in (formats or DT_FORMATS):
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            pass
    m = re.search(r"(\d{4})[-_]?(\d{1,2})[-_]?(\d{1,2})[_-]?(\d{2})(\d{2})", stem)
    if m is None:
        return None
    y, mo, d, h, mi = m.groups()
    try:
        return datetime(int(y), int(mo), int(d), int(h), int(mi))
    except ValueError:
        return None


def NpyToM11(arr, data_min, data_max):
    """raw numpy (H,W) -> m11 torch (1,H,W). NaN/inf는 0으로."""
    x = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.from_numpy(x).float().unsqueeze(0)
    data_min = float(data_min)
    data_max = float(data_max)
    den = data_max - data_min
    if (not np.isfinite(den)) or (den <= 0.0):
        raise ValueError(f"Invalid min/max for NpyToM11: min={data_min} max={data_max}")
    t = torch.clamp(t, min=data_min, max=data_max)
    return 2.0 * (t - data_min) / (den + 1e-9) - 1.0


def load_tensor_any(path):
    """.pt 또는 .npy 텐서 로드 -> float32 (1,H,W)."""
    path = str(path)
    if path.endswith(".pt"):
        x = torch.load(path, map_location="cpu", weights_only=True)
    elif path.endswith(".npy"):
        x = torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"Unsupported dump extension (want .pt/.npy): {path}")
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return x.to(torch.float32)


def load_dump_m11(path, dump_value_range="m11", data_min=0.0, data_max=1.0):
    """예측 덤프 로드. dump_value_range="raw"면 m11로 변환해 반환.

    [적응 포인트 #3] 덤프가 m11이 아니라 물리 단위(raw)면 config.CFG["dump_value_range"]="raw".
    """
    x = load_tensor_any(path)
    if dump_value_range == "raw":
        data_min = float(data_min)
        data_max = float(data_max)
        den = data_max - data_min
        x = 2.0 * (x.clamp(min=data_min, max=data_max) - data_min) / (den + 1e-9) - 1.0
    elif dump_value_range != "m11":
        raise ValueError(f"dump_value_range must be 'm11' or 'raw', got {dump_value_range!r}")
    return x


def seed_from_name(base_seed, name):
    """(base_seed, 샘플명) -> 결정적 32bit 시드 (재실행 시 동일 출력 보장용)."""
    h = hashlib.sha1((str(name) + str(base_seed)).encode()).hexdigest()
    return int(h[:8], 16)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
