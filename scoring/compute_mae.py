"""Ensemble-level MAE (m11 space) for the K=8 canonical prediction sets, per horizon.

Convention (matches forecast_metrics_eval_fid500.py mse_m11 in this folder):
  per-image mean over pixels in [-1,1] space, NO clipping (range_policy=report_only),
  then plain average over images.  MAE = mean|pred - target| computed the same way.
MSE is recomputed alongside as a cross-check against the canonical metrics_summary.json.

Run (CPU is enough):
  /home/user/miniconda3/envs/wind_power/bin/python compute_mae.py --H 1
Outputs: analysis/mae/t{H}/per_image_mse_mae.csv  (wide: name, {set}_mse, {set}_mae)
         analysis/mae/t{H}/summary.json
"""
import argparse, csv, json
from pathlib import Path
import torch

IO = Path("/DATA/shinsung/inference_outputs")
OUT = IO / "analysis" / "mae"

def sets_for(H):
    R = IO / "refined/K8/30ep_best" / f"t{H}"
    B = IO / "analysis" / f"t{H}" / "base_K8"
    return {
        # name: (prediction dir, canonical metrics_summary.json)
        "base_mean":        (B / "prediction",        B / "metrics/metrics_summary.json"),
        "base_median":      (B / "prediction_median", B / "metrics_median/metrics_summary.json"),
        "refined_mean":     (R / "prediction",        R / "metrics/metrics_summary.json"),
        "refined_median":   (R / "prediction_median", R / "metrics_median/metrics_summary.json"),
        "refined_member0":  (R / "member0_eval/prediction", R / "member0_eval/metrics/metrics_summary.json"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, required=True)
    a = ap.parse_args()
    H = a.H
    torch.set_num_threads(8)
    tgt_dir = Path(f"/DATA/shinsung/lon_refiner/ablation_vol_gOFF_lON_t{H}/test_outputs/target")
    sets = sets_for(H)
    names = sorted(p.stem for p in tgt_dir.glob("*.pt"))
    for k, (d, _) in sets.items():
        pn = sorted(p.stem for p in d.glob("*.pt"))
        assert pn == names, f"{k}: file set mismatch ({len(pn)} vs {len(names)})"
    outdir = OUT / f"t{H}"; outdir.mkdir(parents=True, exist_ok=True)
    acc = {k: {"mse": 0.0, "mae": 0.0} for k in sets}
    rows = []
    for i, nm in enumerate(names):
        t = torch.load(tgt_dir / f"{nm}.pt", weights_only=True).float()
        row = {"name": nm}
        for k, (d, _) in sets.items():
            p = torch.load(d / f"{nm}.pt", weights_only=True).float()
            assert p.shape == t.shape, (k, nm, p.shape, t.shape)
            diff = p - t
            mse = float((diff ** 2).mean().item()); mae = float(diff.abs().mean().item())
            row[f"{k}_mse"] = mse; row[f"{k}_mae"] = mae
            acc[k]["mse"] += mse; acc[k]["mae"] += mae
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"t{H} {i+1}/{len(names)}", flush=True)
    n = len(names)
    with open(outdir / "per_image_mse_mae.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summ = {"horizon": H, "n": n, "target_dir": str(tgt_dir), "sets": {}}
    for k, (d, js) in sets.items():
        canon = json.load(open(js))
        mse = acc[k]["mse"] / n; mae = acc[k]["mae"] / n
        summ["sets"][k] = {
            "pred_dir": str(d), "n": n, "mse_m11": mse, "mae_m11": mae,
            "canonical_mse_m11": canon["mse_m11"], "mse_abs_diff": abs(mse - canon["mse_m11"]),
            "canonical_crps": canon.get("crps"), "canonical_crps_mode": canon.get("crps_mode"),
        }
        flag = "" if abs(mse - canon["mse_m11"]) < 1e-7 else "  <-- MSE MISMATCH"
        print(f"t{H} {k:16s} n={n} MSE={mse:.8f} (canon {canon['mse_m11']:.8f}) MAE={mae:.8f}{flag}", flush=True)
    json.dump(summ, open(outdir / "summary.json", "w"), indent=2)

if __name__ == "__main__":
    main()
