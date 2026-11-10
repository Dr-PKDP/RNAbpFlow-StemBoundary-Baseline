"""Association between Rfam-family training exposure and clearing across the analyzable targets.

For each target in data/clearing_signal_ratios.csv, the clearing proportion of each coordinate class is the share of
matched coordinates with signal-to-baseline ratio > 1; the target value is the mean of the angle and distance
proportions. The family exposure of a target is training_chains_same_family in data/cohort/cohort_audit.csv. The
statistic is Spearman's rho, with a two-sided permutation p-value.

  python family_exposure_check.py --ratios ../data/clearing_signal_ratios.csv --audit ../data/cohort/cohort_audit.csv
  python family_exposure_check.py --selftest
"""
import argparse, csv, collections
import numpy as np
from scipy import stats


def target_clearing(path):
    acc = collections.defaultdict(lambda: ([], []))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d, a = acc[r["target"]]
            d.append(float(r["distance_signal_to_baseline"]) > 1)
            a.append(float(r["angle_signal_to_baseline"]) > 1)
    return {t: (np.mean(d) + np.mean(a)) / 2 for t, (d, a) in acc.items()}


def family_counts(path):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["pdb_id"], int(r["training_chains_same_family"]))
    return out


def association(clearing, family, n_perm=100000, seed=0):
    keys = sorted(clearing)
    x = np.array([family[k] for k in keys], float)
    y = np.array([clearing[k] for k in keys], float)
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    hits = sum(abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= abs(rho) - 1e-12 for _ in range(n_perm))
    return len(keys), float(rho), hits / n_perm


def selftest():
    c = {f"t{i}": i / 10 for i in range(8)}
    f = {f"t{i}": i for i in range(8)}
    n, rho, p = association(c, f, n_perm=2000)
    assert n == 8 and abs(rho - 1) < 1e-9 and p < 0.01
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios"); ap.add_argument("--audit"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        n, rho, p = association(target_clearing(a.ratios), family_counts(a.audit))
        print(f"targets: {n}  spearman rho: {rho:.2f}  permutation p: {p:.2f}")
