"""Paired, patient-level bootstrap of every model against a reference, using
all seeds the two experiments share (metric = mean over seeds).

    python scripts/bootstrap_compare.py                         # all vs transformer
    python scripts/bootstrap_compare.py --pairs gat_transformer:gat_none

Writes results/tables/comparisons.{csv,md,json}.
"""
import argparse
import csv
from pathlib import Path

import _bootstrap_path  # noqa: F401
import numpy as np

from sepsis.constants import DYNAMIC, VITALS
from sepsis.evaluation.fast_bootstrap import (RankedScores, mae_sums, multi_seed_bootstrap,
                                              utility_sums)
from sepsis.utils import load_json, save_json

DEFAULT_MODELS = ["persistence", "linear", "mlp", "grud", "gat_transformer",
                  "gat_none", "gat_correlation", "gat_clinical_prior"]
EXTRA_PAIRS = ["gat_transformer:gat_none", "gat_clinical_prior:gat_none", "gat_correlation:gat_none"]
# metric name, higher_is_better
METRICS = [("auprc", True), ("auroc", True), ("utility", True), ("vitals_mae", False)]


def seeds_of(results: Path, exp: str, split: str) -> list[int]:
    return sorted(int(p.parent.name[4:]) for p in results.glob(f"{exp}/seed*/predictions_{split}.npz"))


def metric_fns(results: Path, exp: str, seeds: list[int], split: str, vital_std, n_patients):
    fns = {m: [] for m, _ in METRICS}
    for s in seeds:
        run = results / exp / f"seed{s}"
        p = dict(np.load(run / f"predictions_{split}.npz"))
        if "pred_risk" in p:
            r = RankedScores(p["y_risk"], p["pred_risk"], p["patient"])
            fns["auprc"].append(r.auprc)
            fns["auroc"].append(r.auroc)
            thr = load_json(run / "thresholds.json")["max_utility"]
            fns["utility"].append(utility_sums(p, thr, n_patients).value)
        if "pred_vitals" in p:
            fns["vitals_mae"].append(mae_sums(p, vital_std, n_patients).value)
    return fns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--reference", default="transformer")
    ap.add_argument("--pairs", nargs="*", default=None, help="a:b pairs; default all vs reference + graph pairs")
    ap.add_argument("--splits", nargs="*", default=["test_A", "test_B"])
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    results = Path(a.results)
    norm = load_json(Path(a.processed) / "normalizer.json")
    vital_std = np.asarray(norm["dyn_std"])[[DYNAMIC.index(v) for v in VITALS]]

    pairs = a.pairs or [f"{m}:{a.reference}" for m in DEFAULT_MODELS] + EXTRA_PAIRS
    rows = []
    for split in a.splits:
        cache = {}
        for pair in pairs:
            ea, eb = pair.split(":")
            seeds = sorted(set(seeds_of(results, ea, split)) & set(seeds_of(results, eb, split)))
            if not seeds:
                continue
            n_patients = int(np.load(results / ea / f"seed{seeds[0]}" / f"predictions_{split}.npz")["patient"].max()) + 1
            for e in (ea, eb):
                if (e, tuple(seeds)) not in cache:
                    cache[(e, tuple(seeds))] = metric_fns(results, e, seeds, split, vital_std, n_patients)
            fa, fb = cache[(ea, tuple(seeds))], cache[(eb, tuple(seeds))]
            for metric, higher in METRICS:
                if not fa[metric] or not fb[metric]:
                    continue
                r = multi_seed_bootstrap(fa[metric], fb[metric], n_patients, a.n_boot)
                better = (r["diff"] > 0) == higher
                sig = r["ci_low"] > 0 or r["ci_high"] < 0
                rows.append({"split": split, "model_a": ea, "model_b": eb, "seeds": len(seeds), "metric": metric,
                             **r, "significant": sig,
                             "verdict": ("A better" if better else "B better") if sig else "no sig. difference"})
                print(f"{split:6s} {ea:>20s} vs {eb:<16s} {metric:10s} {r['a']:.4f} vs {r['b']:.4f} "
                      f"diff={r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] {rows[-1]['verdict']}",
                      flush=True)

    out = results / "tables"
    out.mkdir(parents=True, exist_ok=True)
    save_json(rows, out / "comparisons.json")
    if rows:
        with open(out / "comparisons.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    md = ["| Split | A | B | Seeds | Metric | A | B | A − B [95% CI] | p | Verdict |", "|" + "---|" * 10]
    for r in rows:
        md.append(f"| {r['split']} | {r['model_a']} | {r['model_b']} | {r['seeds']} | {r['metric']} | "
                  f"{r['a']:.4f} | {r['b']:.4f} | {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                  f"{r['p_value']:.3f} | {r['verdict']} |")
    (out / "comparisons.md").write_text("\n".join(md) + "\n")
    print(f"-> {out}/comparisons.*")


if __name__ == "__main__":
    main()
