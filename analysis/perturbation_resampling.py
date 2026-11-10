#!/usr/bin/env python3
"""Stability of the perturbation-baseline variance under subsampling of the accepted perturbations.

For a target, the accepted perturbations (seed 0, as in clearing.py) are subsampled without replacement to n = 10, 15,
20 and 25 (sizes up to the accepted count), with 200 subsamples per size (random-number seed 1). For every coordinate the variance of each
subsample is compared with the variance of the full accepted set. Reported per size:
  max relative deviation  max over coordinates of |mean subsample variance - full variance| / full variance
  max CV                  max over coordinates of SD / mean of the subsample variances
Coordinates with zero full-sample variance are skipped.

Usage:
    python perturbation_resampling.py --targets t07 7kd1 --sample-root ../data/Predictions \\
        --bpmaps-root ../data/Inputs --out perturbation_resampling.json
    python perturbation_resampling.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIZES = (10, 15, 20, 25)


def subsample_stability(zs, sizes=SIZES, n_resamples=200, seed=1):
    rng = np.random.default_rng(seed)
    full = zs.var(axis=0)
    keep = full > 0
    out = {}
    for n in sizes:
        if n > len(zs):
            continue
        v = np.array([zs[rng.choice(len(zs), n, replace=False)].var(axis=0) for _ in range(n_resamples)])[:, keep]
        mean, sd = v.mean(axis=0), v.std(axis=0)
        out[n] = {"max_relative_deviation": float(np.max(np.abs(mean - full[keep]) / full[keep])),
                  "max_cv": float(np.max(sd / mean))}
    return out


def selftest():
    rng = np.random.default_rng(1)
    zs = rng.normal(0, [1.0, 5.0, 20.0], size=(40, 3))
    res = subsample_stability(zs)
    assert set(res) == set(SIZES) and all(res[n]["max_cv"] > 0 for n in res)
    assert res[25]["max_cv"] < res[10]["max_cv"]
    print("max CV by size:", {n: round(r["max_cv"], 2) for n, r in res.items()})
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--n-perturbations", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("perturbation_resampling.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    sys.path.insert(0, str(a.pipeline_dir))
    from clearing import read_c1_coords, load_consensus_structure, perturbed_z
    result = {}
    for tid in a.targets:
        db = load_consensus_structure(tid, a.bpmaps_root / tid)
        first = sorted((a.sample_root / tid).glob("Sample_*.pdb"))[0]
        zs, _ = perturbed_z(read_c1_coords(first), db, a.n_perturbations, seed=0)
        res = subsample_stability(zs)
        result[tid] = {"n_accepted": int(len(zs)), "sizes": res}
        print(tid, "accepted", len(zs))
        for n, r in res.items():
            print(f"  n={n}: max relative deviation {r['max_relative_deviation']:.3f}, max CV {r['max_cv']:.3f}")
    a.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
