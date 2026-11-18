"""Raw .psv -> data/processed/{patients.pkl, normalizer.json, splits.json}

    python scripts/preprocess.py                       # full data
    python scripts/preprocess.py --limit 500           # quick debug subset
    python scripts/preprocess.py --raw data/synthetic --out data/processed_synthetic
"""
import argparse

import _bootstrap_path  # noqa: F401

from sepsis.data import load_raw_patients, make_splits, preprocess_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="max patients per hospital")
    a = ap.parse_args()
    raw = load_raw_patients(a.raw, limit=a.limit)
    splits = make_splits(raw, seed=a.split_seed)
    print({k: len(v) for k, v in splits.items() if isinstance(v, list)})
    out = preprocess_all(raw, splits, a.out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
