"""Aggregate results/*/seed*/metrics.json into tables.

Writes to results/tables/:
    runs.csv                   one row per (experiment, seed, split), every scalar metric
    summary_<split>.csv / .md  mean ± std across seeds per experiment
    vitals_<split>.md          forecast MAE per vital (clinical units), mean across seeds

    python scripts/summarize.py                     # all splits
    python scripts/summarize.py --split test_A      # print one split
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import _bootstrap_path  # noqa: F401
import numpy as np

from sepsis.constants import VITALS
from sepsis.utils import load_json

SPLITS = ("val", "test_A", "test_B")
# Display order; anything else found in the results folder is appended after these.
ORDER = ["persistence", "linear", "mlp", "grud", "informer", "autoformer", "transformer",
         "gat_transformer", "gat_clinical_prior", "gat_correlation"]

# (column name, path in metrics.json[split], decimals)
COLUMNS = [
    ("AUROC", "risk.auroc", 3),
    ("AUPRC", "risk.auprc", 3),
    ("Brier", "risk.brier", 4),
    ("ECE", "risk.ece", 4),
    ("Utility", "early_warning@max_utility.utility", 3),
    ("Detect", "early_warning@max_utility.detection_rate", 3),
    ("Warn h (median)", "early_warning@max_utility.median_warning_hours", 1),
    ("FA h/pt-day", "early_warning@max_utility.false_alarm_hours_per_patient_day", 2),
    ("Utility@1FA", "early_warning@fixed_fa.utility", 3),
    ("Detect@1FA", "early_warning@fixed_fa.detection_rate", 3),
    ("Onset macro-F1", "onset.macro_f1", 3),
    ("Onset bal. acc.", "onset.balanced_accuracy", 3),
    ("Onset macro-F1 (septic pts)", "onset_septic.macro_f1", 3),
    ("Vitals MAE", "vitals.mae", 3),
    ("Vitals RMSE", "vitals.rmse", 3),
    ("Vitals MAE pre-onset", "vitals_pre_onset.mae", 3),
]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out.update({f"{key}.{i}": x for i, x in enumerate(v)})
        else:
            out[key] = v
    return out


def collect(results: Path):
    runs = []
    for f in sorted(results.glob("*/seed*/metrics.json")):
        exp = f.parent.parent.name
        info_f = f.parent / "run_info.json"
        info = load_json(info_f) if info_f.exists() else {}
        runs.append((exp, int(f.parent.name[4:]), load_json(f), info))
    exps = sorted({r[0] for r in runs}, key=lambda e: (ORDER.index(e) if e in ORDER else len(ORDER), e))
    return runs, exps


def fmt(vals, dec):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{dec}f}"
    return f"{np.mean(vals):.{dec}f} ± {np.std(vals):.{dec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--split", default=None, help="print only this split")
    a = ap.parse_args()
    results = Path(a.results)
    out = results / "tables"
    out.mkdir(parents=True, exist_ok=True)
    runs, exps = collect(results)

    # 1. every run, every split, every scalar
    flat_rows = []
    for exp, seed, m, info in runs:
        for split in SPLITS:
            if split not in m:
                continue
            row = {"experiment": exp, "seed": seed, "split": split, "n_params": m.get("n_params"),
                   "best_epoch": info.get("best_epoch"), "epochs_run": info.get("epochs_run"),
                   "selection": info.get("selection"),
                   **{f"best_epoch_{k}": v for k, v in (info.get("best_epoch_per_task") or {}).items()},
                   "train_seconds": info.get("train_seconds")}
            row.update(flatten(m[split]))
            flat_rows.append(row)
    keys = list(dict.fromkeys(k for r in flat_rows for k in r))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat_rows)

    # 2. mean ± std per experiment
    for split in SPLITS:
        table = defaultdict(lambda: defaultdict(list))
        for r in flat_rows:
            if r["split"] != split:
                continue
            t = table[r["experiment"]]
            t["seeds"].append(r["seed"])
            t["params"].append(r["n_params"])
            for col, path, _ in COLUMNS:
                if path in r:
                    t[col].append(r[path])
        header = ["Model", "Seeds", "Params"] + [c for c, _, _ in COLUMNS]
        md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        csv_rows = []
        for exp in exps:
            if exp not in table:
                continue
            t = table[exp]
            cells = [exp, str(len(t["seeds"])), f"{t['params'][0]:,}"]
            csv_row = {"experiment": exp, "n_seeds": len(t["seeds"]), "n_params": t["params"][0]}
            for col, _, dec in COLUMNS:
                cells.append(fmt(t.get(col, []), dec))
                v = [x for x in t.get(col, []) if x is not None and np.isfinite(x)]
                csv_row[f"{col} mean"] = float(np.mean(v)) if v else ""
                csv_row[f"{col} std"] = float(np.std(v)) if v else ""
            md.append("| " + " | ".join(cells) + " |")
            csv_rows.append(csv_row)
        (out / f"summary_{split}.md").write_text("\n".join(md) + "\n")
        if csv_rows:
            with open(out / f"summary_{split}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
                w.writeheader()
                w.writerows(csv_rows)

        # per-vital forecast error
        vh = ["Model"] + VITALS + [f"t+{h + 1}" for h in range(6)]
        vmd = ["| " + " | ".join(vh) + " |", "|" + "---|" * len(vh)]
        for exp in exps:
            rs = [r for r in flat_rows if r["experiment"] == exp and r["split"] == split]
            if not rs or "vitals.mae" not in rs[0]:
                continue
            cells = [exp] + [fmt([r[f"vitals.mae_per_vital.{v}"] for r in rs], 2) for v in VITALS]
            cells += [fmt([r[f"vitals.mae_per_horizon.{h}"] for r in rs], 2) for h in range(6)]
            vmd.append("| " + " | ".join(cells) + " |")
        (out / f"vitals_{split}.md").write_text("\n".join(vmd) + "\n")

        if a.split in (None, split):
            print(f"\n## {split}\n")
            print("\n".join(md))
    print(f"\ntables written to {out}/")


if __name__ == "__main__":
    main()
