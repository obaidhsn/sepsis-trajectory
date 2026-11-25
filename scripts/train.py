"""Train one model with one seed, then evaluate on val / test_A / test_B.

    python scripts/train.py --config configs/experiments/gat_transformer.yaml --seed 0
    python scripts/train.py --config configs/experiments/transformer.yaml --override data.lookback=48

Outputs in results/<experiment>/seed<k>/:
    config.yaml, run_info.json, best.pt, history.json, thresholds.json,
    metrics.json, predictions_{val,test_A,test_B}.npz
    With per-task selection there is also best_<task>.pt, one snapshot per head.
"""
import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap_path  # noqa: F401
import numpy as np
import torch
import yaml

from sepsis.config import load_config
from sepsis.constants import DYNAMIC, VITALS
from sepsis.data.dataset import PatientSequenceDataset, SepsisWindowDataset
from sepsis.data.preprocess import load_processed
from sepsis.evaluation import (evaluate_predictions, select_threshold_fixed_fa,
                               select_threshold_utility)
from sepsis.graphs import build_adjacency
from sepsis.models import build_model, count_parameters
from sepsis.data.targets import onset_bucket
from sepsis.training import Trainer, onset_class_weights_from_counts, predict
from sepsis.utils import load_json, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None, help="experiment name (default: config file stem)")
    ap.add_argument("--processed", default=None, help="override paths.processed_dir")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load the saved best_<task>.pt snapshots of this run and re-evaluate")
    a = ap.parse_args()

    cfg = load_config(a.config, a.override)
    if a.seed is not None:
        cfg["train"]["seed"] = a.seed
    if a.processed:
        cfg["paths"]["processed_dir"] = a.processed
    seed = cfg["train"]["seed"]
    set_seed(seed)
    device = cfg["train"]["device"]
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple-silicon GPU
        else:
            device = "cpu"

    name = a.name or Path(a.config).stem
    out_dir = Path(cfg["paths"]["results_dir"]) / name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    started = time.time()
    previous = load_json(out_dir / "run_info.json") if a.eval_only else {}
    run_info = {
        "experiment": name, "seed": seed, "config_file": a.config, "overrides": a.override,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
    }
    if a.eval_only:  # keep the training run's metadata, record the re-evaluation separately
        run_info = {**previous, "re_evaluated": {k: run_info[k] for k in ("command", "started_utc", "device")}}
    save_json(run_info, out_dir / "run_info.json")

    patients, splits = load_processed(cfg["paths"]["processed_dir"])
    d = cfg["data"]
    if d.get("clip_inputs", True):
        # Clip every normalised variable to the range seen in the training patients. Training data is
        # unchanged; val/test values outside that range (e.g. data-entry errors such as an FiO2 of
        # thousands at Hospital B) can otherwise blow up models without input normalisation.
        lo = np.min([patients[p]["x"].min(0) for p in splits["train"]], axis=0)
        hi = np.max([patients[p]["x"].max(0) for p in splits["train"]], axis=0)
        n_clipped = 0
        for rec in patients.values():
            n_clipped += int(((rec["x"] < lo) | (rec["x"] > hi)).sum())
            np.clip(rec["x"], lo, hi, out=rec["x"])
        run_info["inputs_clipped_to_train_range"] = n_clipped
        print(f"clipped {n_clipped:,} input values outside the training range")
    adjacency = None
    if cfg["model"]["name"] == "gat_transformer":
        adjacency = build_adjacency(cfg["model"].get("graph", "learned"),
                                    [patients[p] for p in splits["train"]])
        np.save(out_dir / "adjacency.npy", adjacency)

    model = build_model(cfg, adjacency)
    n_params = count_parameters(model)
    print(f"{name}: {n_params:,} trainable parameters on {device}")

    train_ds = SepsisWindowDataset(patients, splits["train"], d["lookback"], d["horizon"],
                                   d["windows_per_patient"], d["septic_oversample"], seed, d["onset_edges"])
    seq = {s: PatientSequenceDataset(patients, splits[s], d["lookback"], d["horizon"], d["onset_edges"])
           for s in ("val", "test_A", "test_B")}

    # Onset class weights: "auto" = balanced weights from the training windows' bucket counts.
    loss_cfg = cfg["train"]["loss"]
    if loss_cfg.get("onset_class_weights") == "auto":
        k = len(d["onset_edges"]) + 1
        counts = np.bincount([onset_bucket(t, patients[pid]["t_sepsis"], d["onset_edges"])
                              for pid, t in train_ds.index], minlength=k)
        loss_cfg["onset_class_weights"] = onset_class_weights_from_counts(counts)
        run_info["onset_bucket_counts_train"] = counts.tolist()
        print(f"onset buckets in training windows {counts.tolist()} -> weights "
              f"{[round(w, 2) for w in loss_cfg['onset_class_weights']]}")
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)

    trainer = Trainer(model, cfg, out_dir, device)
    if a.eval_only:
        history = load_json(out_dir / "history.json")
        trainer.load_selected(history["best_epoch_per_task"])
    else:
        history = trainer.fit(train_ds, seq["val"])
        save_json(history, out_dir / "history.json")
    train_seconds = time.time() - started

    preds = {s: predict(trainer.selected, ds, cfg["train"]["eval_batch_size"], device) for s, ds in seq.items()}

    # Thresholds chosen on validation only, then frozen.
    thresholds = {}
    if "pred_risk" in preds["val"]:
        v = preds["val"]
        thresholds = {
            "max_utility": select_threshold_utility(v["y_risk"], v["pred_risk"], v["patient"], v["hour"]),
            "fixed_fa": select_threshold_fixed_fa(v["y_risk"], v["pred_risk"], v["patient"], v["hour"],
                                                  cfg["eval"]["fixed_false_alarm_rate"]),
        }
    save_json(thresholds, out_dir / "thresholds.json")

    norm = load_json(Path(cfg["paths"]["processed_dir"]) / "normalizer.json")
    vital_std = np.asarray(norm["dyn_std"])[[DYNAMIC.index(v) for v in VITALS]]
    metrics = {"n_params": n_params}
    for s, p in preds.items():
        t_sep = np.array([patients[pid]["t_sepsis"] for pid in seq[s].ids])
        metrics[s] = evaluate_predictions(p, thresholds, vital_std, t_sep)
        np.savez_compressed(out_dir / f"predictions_{s}.npz", **p)
    save_json(metrics, out_dir / "metrics.json")

    run_info.update(n_params=n_params, best_epoch=history["best_epoch"], epochs_run=history["epochs_run"],
                    selection=history["selection"], best_epoch_per_task=history["best_epoch_per_task"],
                    monitor=history["monitor"], best_val_monitor=history["best_val_monitor"],
                    train_seconds=previous.get("train_seconds") if a.eval_only else round(train_seconds, 1),
                    total_seconds=previous.get("total_seconds") if a.eval_only else round(time.time() - started, 1),
                    finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    save_json(run_info, out_dir / "run_info.json")
    print(f"done in {run_info['total_seconds']:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
