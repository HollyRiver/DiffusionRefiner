#!/usr/bin/env python3
"""
forecast_metrics_eval_fid500.py
===============================
Saved-prediction metrics for the DAD test pipeline. Provides

    compute_metrics(pred_dir, target_dir, outdir, data_min, data_max, metrics,
                    value_range, range_policy, point_pred_mode, fid_pred_mode,
                    fid_backend, fid_batch_size, fid_dims, fid_weights_path,
                    lpips_net, lpips_batch_size, device, fid_low_ram, **kw) -> dict

exactly as train.py's _run_saved_prediction_metrics calls it. Returns/writes:
    mse_m11, mse_data, rmse_data, ssim, crps, lpips, fid  (+ bookkeeping keys)
    <outdir>/metrics_summary.json  and  <outdir>/per_image_metrics.csv

Metric definitions
------------------
mse_m11   : per-image MSE in [-1,1] space, averaged over images.
mse_data  : mse_m11 * ((data_max-data_min)/2)^2  (exact linear-rescale identity).
rmse_data : sqrt(mse_data).
ssim      : AC U-Net convention (utils_nmode_abb) — skimage.structural_similarity on
            uint8 [0,255] maps (m11 -> (x+1)/2*255 -> clip -> uint8), data_range=255,
            win_size=7 (skimage default = UNIFORM 7x7 window). Adopted 2026-08-26.
crps      : ensemble CRPS (fair estimator)
                CRPS = (1/M) sum_i |x_i - y| - (1/(2M(M-1))) sum_{i != j} |x_i - x_j|
            Members are auto-discovered:
              - pred_dir=.../prediction      -> ../samples/member_*/<name>.pt
              - pred_dir=.../samples/member_k-> sibling member_* dirs
            If no members exist (deterministic forecast, e.g. AC U-Net),
            CRPS of a point forecast reduces to MAE -> reported with a note.
lpips     : LPIPS (alex/vgg) on 3-channel-replicated [-1,1] maps (optional dep).
fid       : pytorch_fid (or clean_fid fallback) between gray→RGB(3ch replicated)
            PNG renders of pred vs target. Full test set by default (no cap).
            Standard diffusion practice: InceptionV3 pool3 2048-d features,
            gray→explicit RGB 3-channel replication (not L mode conversion).

Missing optional dependencies or member dirs degrade the affected metric to
nan with an explanatory note in the summary; nothing crashes the test run.

Standalone CLI (e.g. AC U-Net baseline dirs):
    python -m defaultcode2.forecast_metrics_eval_fid500 \
        --pred-dir ./ACUNET1/60m/prediction --target-dir ./ACUNET1/60m/target \
        --outdir ./ACUNET1/60m/metrics --data-min 0.0 --data-max 26.347150802612
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_METRICS = ["mse_m11", "mse_data", "rmse_data", "ssim", "crps", "lpips", "fid"]
FID_MAX_IMAGES_DEFAULT = 10**9  # no cap by default: use full test set (standard for papers)


# ----------------------------------------------------------------------------
# io helpers
# ----------------------------------------------------------------------------
def _extract_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in ("pred", "prediction", "out", "tensor", "data", "x", "sample", "mean"):
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
    raise ValueError(f"unsupported .pt payload: {type(obj)}")


def _load_hw(path: Path, point_pred_mode: str = "mean") -> torch.Tensor:
    """Load a .pt -> (H,W) float32 cpu tensor; (M,...) stacks reduced by mean/first."""
    t = _extract_tensor(torch.load(str(path), map_location="cpu")).detach().float()
    if t.ndim == 4:                       # (M,C,H,W)
        t = t.mean(0) if point_pred_mode == "mean" else t[0]
    if t.ndim == 3 and t.shape[0] > 1:    # (M,H,W) or (C>1,H,W)
        t = t.mean(0, keepdim=True) if point_pred_mode == "mean" else t[:1]
    while t.ndim > 2:
        t = t.squeeze(0)
    return t


def _to_m11(x: torch.Tensor, value_range: str) -> torch.Tensor:
    if value_range == "01":
        return x * 2.0 - 1.0
    return x  # "m11" (default) or anything else: assume already [-1,1]


def _stems(d: Path) -> List[str]:
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.pt"))


def _discover_member_dirs(pred_dir: Path) -> List[Path]:
    """Find ensemble member dirs related to pred_dir (see module docstring)."""
    cands: List[Path] = []
    if pred_dir.name.startswith("member_"):
        cands = sorted(p for p in pred_dir.parent.glob("member_*") if p.is_dir())
    else:
        for sub in ("samples", "members"):
            root = pred_dir.parent / sub
            if root.is_dir():
                cands = sorted(p for p in root.glob("member_*") if p.is_dir())
                if cands:
                    break
    return cands


# ----------------------------------------------------------------------------
# SSIM (single channel, gaussian 11x11 sigma 1.5, L=1 on [0,1])
# ----------------------------------------------------------------------------
def _gaussian_window(ks: int = 11, sigma: float = 1.5, device="cpu") -> torch.Tensor:
    xs = torch.arange(ks, dtype=torch.float32, device=device) - (ks - 1) / 2.0
    g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # (1,1,ks,ks)
    return w


@torch.no_grad()
def _ssim_pair(a01: torch.Tensor, b01: torch.Tensor, win: torch.Tensor) -> float:
    """a01,b01: (1,1,H,W) in [0,1]. Returns mean SSIM."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = F.conv2d(a01, win)
    mu_b = F.conv2d(b01, win)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    s_a = F.conv2d(a01 * a01, win) - mu_a2
    s_b = F.conv2d(b01 * b01, win) - mu_b2
    s_ab = F.conv2d(a01 * b01, win) - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * s_ab + C2)) / ((mu_a2 + mu_b2 + C1) * (s_a + s_b + C2))
    return float(ssim_map.mean().item())


def _ssim_acunet(pred_m11: torch.Tensor, tgt_m11: torch.Tensor) -> float:
    """AC U-Net SSIM (utils_nmode_abb): m11 -> uint8[0,255], skimage SSIM win_size=7,
    data_range=255 (uniform 7x7 window). Adopted 2026-08-26 as the canonical SSIM."""
    from skimage.metrics import structural_similarity as _sk
    o = np.clip((pred_m11.detach().cpu().numpy() + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8).squeeze()
    t = np.clip((tgt_m11.detach().cpu().numpy() + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8).squeeze()
    return float(_sk(t, o, data_range=255, win_size=7))


# ----------------------------------------------------------------------------
# CRPS (fair ensemble estimator)
# ----------------------------------------------------------------------------
@torch.no_grad()
def _crps_image(members_m11: torch.Tensor, tgt_m11: torch.Tensor) -> float:
    """members: (M,H,W), tgt: (H,W); returns pixel-mean CRPS in m11 units.
    M==1 reduces to MAE (the CRPS of a point forecast)."""
    M = members_m11.shape[0]
    term1 = (members_m11 - tgt_m11.unsqueeze(0)).abs().mean()
    if M > 1:
        diffs = (members_m11.unsqueeze(0) - members_m11.unsqueeze(1)).abs()  # (M,M,H,W)
        term2 = diffs.sum(dim=(0, 1)).mean() / (2.0 * M * (M - 1))
    else:
        term2 = torch.zeros((), device=members_m11.device)
    return float((term1 - term2).item())


# ----------------------------------------------------------------------------
# FID via pytorch_fid on gray->RGB PNG dumps
# Standard diffusion practice: explicit 3-channel RGB replication (not L mode)
# to avoid PIL L->RGB rounding differences. No image cap (use full test set).
# ----------------------------------------------------------------------------
def _save_gray_rgb_png(x_m11: torch.Tensor, path: str):
    """Save a [-1,1] grayscale map as a 3-channel RGB PNG (gray replicated 3x)."""
    from PIL import Image
    a = ((x_m11.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()
    rgb = np.stack([a, a, a], axis=-1)  # (H,W,3)
    Image.fromarray(rgb, mode="RGB").save(path)


def _prepare_local_fid_weights(fid_weights_path: Optional[str]) -> Optional[str]:
    """Pre-cache InceptionV3 weights to torch hub dir if a local copy exists."""
    expected_name = "pt_inception-2015-12-05-6726825d.pth"
    candidates: List[Path] = []
    if fid_weights_path:
        candidates.append(Path(fid_weights_path))
    env_path = os.environ.get("FID_WEIGHTS_PATH")
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve().parent
    for base in [here, Path.cwd()]:
        candidates.append(base / expected_name)
        candidates.append(base / "weights" / expected_name)
        candidates.append(base / "fid_weights" / expected_name)
    for cand in candidates:
        try:
            if cand.is_file():
                hub_dir = Path(torch.hub.get_dir()) / "checkpoints"
                hub_dir.mkdir(parents=True, exist_ok=True)
                dst = hub_dir / expected_name
                if not dst.exists():
                    shutil.copy2(cand, dst)
                return str(dst)
        except Exception:
            pass
    return None


def _compute_fid(pairs, value_range, point_pred_mode, device, dims, batch_size,
                 fid_max_images, outdir: Path, errors: Dict[str, str],
                 fid_weights_path=None, fid_backend="pytorch_fid") -> float:
    use = pairs if fid_max_images >= len(pairs) else pairs[: int(fid_max_images)]
    tmp_root = Path(tempfile.mkdtemp(prefix="fid_", dir=str(outdir)))
    real_d, fake_d = tmp_root / "real", tmp_root / "fake"
    real_d.mkdir(parents=True); fake_d.mkdir(parents=True)
    try:
        for nm, ppath, tpath in use:
            pred = _to_m11(_load_hw(ppath, point_pred_mode), value_range)
            tgt = _to_m11(_load_hw(tpath, point_pred_mode), value_range)
            _save_gray_rgb_png(pred, str(fake_d / f"{nm}.png"))
            _save_gray_rgb_png(tgt, str(real_d / f"{nm}.png"))
        dev = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

        backend = str(fid_backend).lower()
        _prepare_local_fid_weights(fid_weights_path)

        if backend in {"auto", "pytorch_fid"}:
            try:
                from pytorch_fid.fid_score import calculate_fid_given_paths
                fid = calculate_fid_given_paths(
                    [str(real_d), str(fake_d)], batch_size=max(1, int(batch_size)),
                    device=dev, dims=int(dims), num_workers=0,
                )
                return float(fid)
            except Exception as e:
                errors["fid_pytorch"] = str(e)
                if backend == "pytorch_fid":
                    return float("nan")

        if backend in {"auto", "clean_fid", "cleanfid"}:
            try:
                from cleanfid import fid as cleanfid
                fid = cleanfid.compute_fid(str(real_d), str(fake_d), mode="clean", device=dev)
                return float(fid)
            except Exception as e:
                errors["fid_cleanfid"] = str(e)

        errors["fid"] = f"no FID backend succeeded (tried: {backend})"
        return float("nan")
    except Exception as e:
        errors["fid"] = f"fid failed: {e}"
        return float("nan")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ----------------------------------------------------------------------------
# main entry (signature must match train.py's call)
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_metrics(
    pred_dir: Path,
    target_dir: Path,
    outdir: Path,
    data_min: float,
    data_max: float,
    metrics: Optional[List[str]] = None,
    value_range: str = "m11",
    range_policy: str = "report_only",
    point_pred_mode: str = "mean",
    fid_pred_mode: str = "point",
    fid_backend: str = "pytorch_fid",
    fid_batch_size: int = 1,
    fid_dims: int = 2048,
    fid_weights_path=None,
    lpips_net: str = "alex",
    lpips_batch_size: int = 8,
    device: str = "cuda",
    fid_low_ram: bool = True,
    fid_max_images: int = FID_MAX_IMAGES_DEFAULT,
    **_ignored,
) -> Dict:
    pred_dir, target_dir, outdir = Path(pred_dir), Path(target_dir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = list(metrics or DEFAULT_METRICS)
    dev = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
    errors: Dict[str, str] = {}

    # ---- pair stems ----
    ps, ts = set(_stems(pred_dir)), set(_stems(target_dir))
    names = sorted(ps & ts)
    pairs = [(nm, pred_dir / f"{nm}.pt", target_dir / f"{nm}.pt") for nm in names]
    if not pairs:
        summary = {"n_pairs": 0, "error": f"no common .pt stems between {pred_dir} and {target_dir}"}
        (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # ---- members for CRPS ----
    member_dirs = _discover_member_dirs(pred_dir) if "crps" in metrics else []
    crps_mode = f"ensemble({len(member_dirs)} members)" if len(member_dirs) > 1 else \
                ("point-forecast (CRPS==MAE)" if "crps" in metrics else "off")

    # ---- lpips model (optional) ----
    lpips_model = None
    if "lpips" in metrics:
        try:
            import lpips as _lpips
            lpips_model = _lpips.LPIPS(net=lpips_net).to(dev).eval()
        except Exception as e:
            errors["lpips"] = f"lpips unavailable: {e}"

    win = _gaussian_window(device=dev) if "ssim" in metrics else None
    scale = (float(data_max) - float(data_min)) / 2.0  # m11 -> data units

    rows = []
    range_violations = 0
    lp_buf_a, lp_buf_b, lp_idx = [], [], []

    def _flush_lpips():
        nonlocal lp_buf_a, lp_buf_b, lp_idx
        if lpips_model is None or not lp_idx:
            lp_buf_a, lp_buf_b, lp_idx = [], [], []
            return
        a = torch.cat(lp_buf_a).to(dev)
        b = torch.cat(lp_buf_b).to(dev)
        d = lpips_model(a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1)).flatten().cpu()
        for j, ridx in enumerate(lp_idx):
            rows[ridx]["lpips"] = float(d[j].item())
        lp_buf_a, lp_buf_b, lp_idx = [], [], []

    for nm, ppath, tpath in pairs:
        try:
            pred = _to_m11(_load_hw(ppath, point_pred_mode), value_range)
            tgt = _to_m11(_load_hw(tpath, point_pred_mode), value_range)
        except Exception as e:
            errors.setdefault("load", f"{nm}: {e}")
            continue
        if float(pred.min()) < -1.001 or float(pred.max()) > 1.001:
            range_violations += 1
            if range_policy == "clip":
                pred = pred.clamp(-1, 1)
        row = {"name": nm}
        pred_d, tgt_d = pred.to(dev), tgt.to(dev)

        if "mse_m11" in metrics or "mse_data" in metrics or "rmse_data" in metrics:
            row["mse_m11"] = float(((pred_d - tgt_d) ** 2).mean().item())
        if "ssim" in metrics:
            row["ssim"] = _ssim_acunet(pred_d, tgt_d)
        if "crps" in metrics:
            try:
                if len(member_dirs) > 1:
                    mems = [
                        _to_m11(_load_hw(md / f"{nm}.pt", "first"), value_range)
                        for md in member_dirs if (md / f"{nm}.pt").exists()
                    ]
                    if len(mems) >= 2:
                        row["crps"] = _crps_image(torch.stack(mems).to(dev), tgt_d)
                    else:
                        row["crps"] = _crps_image(pred_d.unsqueeze(0), tgt_d)
                else:
                    row["crps"] = _crps_image(pred_d.unsqueeze(0), tgt_d)  # == MAE
            except Exception as e:
                errors.setdefault("crps", f"{nm}: {e}")
        rows.append(row)
        if lpips_model is not None:
            lp_buf_a.append(pred.unsqueeze(0).unsqueeze(0))
            lp_buf_b.append(tgt.unsqueeze(0).unsqueeze(0))
            lp_idx.append(len(rows) - 1)
            if len(lp_idx) >= max(1, int(lpips_batch_size)):
                _flush_lpips()
    _flush_lpips()

    # ---- aggregate ----
    def _mean(key):
        vals = [r[key] for r in rows if key in r and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    summary: Dict = {"n_pairs": len(rows), "value_range": value_range,
                     "range_violations": range_violations, "crps_mode": crps_mode,
                     "fid_backend": fid_backend, "point_pred_mode": point_pred_mode}
    mse_m11 = _mean("mse_m11")
    if "mse_m11" in metrics:
        summary["mse_m11"] = mse_m11
    if "mse_data" in metrics:
        summary["mse_data"] = mse_m11 * (scale ** 2) if mse_m11 == mse_m11 else float("nan")
    if "rmse_data" in metrics:
        summary["rmse_data"] = math.sqrt(summary.get("mse_data", float("nan"))) \
            if summary.get("mse_data", float("nan")) == summary.get("mse_data") else float("nan")
    if "ssim" in metrics:
        summary["ssim"] = _mean("ssim")
    if "crps" in metrics:
        summary["crps"] = _mean("crps")
    if "lpips" in metrics:
        summary["lpips"] = _mean("lpips") if lpips_model is not None else float("nan")
    if "fid" in metrics:
        summary["fid"] = _compute_fid(pairs, value_range, point_pred_mode, str(dev),
                                      fid_dims, fid_batch_size, fid_max_images, outdir, errors,
                                      fid_weights_path=fid_weights_path, fid_backend=fid_backend)
        summary["fid_n_images"] = min(len(pairs), int(fid_max_images))
    if errors:
        summary["errors"] = errors

    # ---- write ----
    keys = ["name"] + [k for k in ("mse_m11", "ssim", "crps", "lpips") if any(k in r for r in rows)]
    with open(outdir / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ----------------------------------------------------------------------------
# standalone CLI (e.g. AC U-Net baseline folders)
# ----------------------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="6-metric evaluation of saved .pt predictions")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-min", type=float, default=0.0)
    ap.add_argument("--data-max", type=float, default=26.347150802612)
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--value-range", default="m11", choices=["m11", "01"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fid-max-images", type=int, default=FID_MAX_IMAGES_DEFAULT)
    ap.add_argument("--lpips-net", default="alex")
    a = ap.parse_args()
    s = compute_metrics(
        pred_dir=Path(a.pred_dir), target_dir=Path(a.target_dir), outdir=Path(a.outdir),
        data_min=a.data_min, data_max=a.data_max, metrics=a.metrics,
        value_range=a.value_range, range_policy="report_only", point_pred_mode="mean",
        fid_pred_mode="point", fid_backend="pytorch_fid", fid_batch_size=1,
        fid_dims=2048, fid_weights_path=None, lpips_net=a.lpips_net,
        lpips_batch_size=8, device=a.device, fid_low_ram=True,
        fid_max_images=a.fid_max_images,
    )
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    _cli()
