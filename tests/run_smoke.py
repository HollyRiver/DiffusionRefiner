"""End-to-end smoke: fake data -> train(2ep) x2 modes -> infer x2 modes -> asserts.
GPU 불필요. Run:
  cd /DATA/shinsung/refiner_module && python tests/run_smoke.py
"""
import shutil, subprocess, sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
TMP = MODULE / "tests" / "_smoke_tmp"


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=MODULE)
    assert r.returncode == 0, f"command failed: {cmd}"


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    from tests.make_fake_data import make_fake_tree
    make_fake_tree(TMP, hw=64)

    py = sys.executable
    common = ["--offset", "1", "--gpu", "-1",
              "--dumps_root", str(TMP / "dumps"),
              "--raw_data_dir", str(TMP / "raw"),
              "--work_dir", str(TMP / "work"),
              "--num_workers", "0", "--batch_size", "2"]

    for mode in ("ensemble", "member"):
        run([py, "train_refiner.py", "--mode", mode, *common, "--epochs", "2"])
    for mode in ("ensemble", "member"):
        run([py, "infer_refiner.py", "--mode", mode, *common, "--ddim_steps", "4"])

    import torch
    for mode in ("ensemble", "member"):
        ckpt = TMP / "work" / f"refiner_t1_{mode}" / "best_R.pt"
        assert ckpt.exists(), ckpt
        preds = sorted((TMP / "work" / f"outputs_t1_{mode}" / "prediction").glob("*.pt"))
        assert len(preds) == 4, (mode, len(preds))
        x = torch.load(preds[0], weights_only=True)
        assert x.shape[0] == 1 and x.min() >= -1.0 and x.max() <= 1.0
        assert torch.isfinite(x).all()

    shutil.rmtree(TMP)
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
