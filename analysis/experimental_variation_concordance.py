#!/usr/bin/env python3
"""Does baseline clearing track experimental variation?

Question. The paper treats a coordinate whose ensemble variance exceeds the
stem-corresponding perturbation baseline as carrying more than coordinate-definition
noise. If that reading is right, coordinates that clear the baseline in the generated
ensemble should also vary in the experimental structures of the same RNA, beyond the
variation the baseline attributes to the coordinate definition.

Data. For every coordinate of every target with experimental structures:
  v_gen   variance across the generated samples
  v_base  variance across the perturbation baseline (mean-centred, 1/n, as in clearing.py)
  v_exp   variance across NMR models, or the variance of the two values of an apo/holo pair
Torsions are handled with wrapped differences.

Dispersion. --centre mean (default) uses mean-centred variances with the 1/n divisor for all three quantities, the
estimator of clearing.py; an apo/holo pair then has v_exp = delta^2 / 4. --centre median uses the second moment
about the median with divisor n - 1 (apo/holo pair delta^2 / 2).

Statistic. Because v_base is the denominator of both the clearing ratio and the
experimental excess, correlating the two ratios would give a positive result even for
unrelated v_gen and v_exp. The primary statistic is therefore the partial Spearman
correlation between log v_gen and log v_exp given log v_base and coordinate type
(distance, angle, torsion), with a 95% interval from a bootstrap over targets
(targets, not coordinates, are resampled). A 2 x 2 table of "clears the baseline" against
"experimental variance exceeds the baseline" is reported as a descriptive complement.

Coverage (descriptive). Per coordinate, the fraction of experimental values inside the
central 90% interval of the generated samples.

Targets are skipped, with the reason recorded, if the experimental structure cannot be
aligned to the model input (fewer than --min-mapped, default 85%, of input residues mapped), if fewer than 2
coordinates are shared, or if the perturbation baseline is not estimable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from displacement_direction_test import diff, read_models, map_to_input, labeled_z  # noqa: E402
except ImportError:
    from displacement_direction_test import diff, read_models, map_to_input, labeled_z  # noqa: E402


def ctype(label):
    return "distance" if label.startswith("d[") else ("angle" if label.startswith("theta[") else "torsion")


def _rank(x):
    try:
        from scipy.stats import rankdata
        return rankdata(x)
    except Exception:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r


def _resid(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(lg, le, lb, types):
    """Partial Spearman of lg and le given lb and type dummies."""
    cols = [np.ones(len(lg)), _rank(lb)]
    for t in sorted(set(types)):
        if t != sorted(set(types))[0]:
            cols.append((np.asarray(types) == t).astype(float))
    X = np.column_stack(cols)
    rg, re = _resid(_rank(lg), X), _resid(_rank(le), X)
    if rg.std() < 1e-12 or re.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rg, re)[0, 1])


def cohort_statistics(rows, n_boot=5000, seed=0):
    """rows: list of dicts with target, type, v_gen, v_base, v_exp."""
    rows = [r for r in rows if min(r["v_gen"], r["v_base"], r["v_exp"]) > 1e-12]
    if len(rows) < 8 or len({r["target"] for r in rows}) < 3:
        return {"testable": False, "reason": "fewer than 8 usable coordinates or 3 targets", "n_coordinates": len(rows)}
    tg = np.array([r["target"] for r in rows])
    types = np.array([r["type"] for r in rows])
    lg = np.log([r["v_gen"] for r in rows]); le = np.log([r["v_exp"] for r in rows]); lb = np.log([r["v_base"] for r in rows])
    rho = partial_spearman(lg, le, lb, types)
    naive = float(np.corrcoef(_rank(lg - lb), _rank(le - lb))[0, 1])
    rng = np.random.default_rng(seed)
    uniq = np.unique(tg)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(tg == t)[0] for t in pick])
        if len(idx) < 8:
            continue
        v = partial_spearman(lg[idx], le[idx], lb[idx], types[idx])
        if not np.isnan(v):
            boots.append(v)
    boots = np.array(boots)
    cleared = np.exp(lg - lb) > 1
    exp_up = np.exp(le - lb) > 1
    tab = {"cleared_and_exp_exceeds": int((cleared & exp_up).sum()), "cleared_only": int((cleared & ~exp_up).sum()),
           "exp_exceeds_only": int((~cleared & exp_up).sum()), "neither": int((~cleared & ~exp_up).sum())}
    return {"testable": True, "n_coordinates": int(len(rows)), "n_targets": int(len(uniq)),
            "partial_spearman": rho, "partial_spearman_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "fraction_of_bootstrap_positive": float((boots > 0).mean()), "naive_spearman_of_ratios": naive,
            "table_2x2": tab,
            "agreement": float((tab["cleared_and_exp_exceeds"] + tab["neither"]) / len(rows)),
            "fraction_cleared": float(cleared.mean()), "fraction_exp_exceeds": float(exp_up.mean())}


def coordinate_rows(tid, labels, Z, is_tors, v_base, E, centre="mean"):
    """E: (n_exp, p) experimental coordinate values (apo/holo pair gives n_exp = 2)."""
    K = len(Z)
    assert centre in ("mean", "median")
    mid = np.median if centre == "median" else np.mean
    off = 1 if centre == "median" else 0
    c0 = mid(Z, axis=0).astype(float)
    for j in np.where(is_tors)[0]:
        r = np.deg2rad(Z[:, j]); c0[j] = np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    dev = diff(Z, c0, is_tors)
    v_gen = (dev ** 2).sum(axis=0) / (K - off)
    edev = diff(E, c0, is_tors)
    if len(E) == 2:
        v_exp = diff(E[1], E[0], is_tors) ** 2 / (2.0 if centre == "median" else 4.0)
    else:
        ec = mid(E, axis=0).astype(float)
        for j in np.where(is_tors)[0]:
            r = np.deg2rad(E[:, j]); ec[j] = np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
        v_exp = (diff(E, ec, is_tors) ** 2).sum(axis=0) / (len(E) - off)
    lo, hi = np.percentile(dev, 5, axis=0), np.percentile(dev, 95, axis=0)
    cover = ((edev >= lo) & (edev <= hi)).mean(axis=0)
    out = []
    for j, l in enumerate(labels):
        out.append({"target": tid, "coordinate": l, "type": ctype(l), "v_gen": float(v_gen[j]), "v_base": float(v_base[j]),
                    "v_exp": float(v_exp[j]), "log2_gen_over_base": float(np.log2(v_gen[j] / v_base[j])) if min(v_gen[j], v_base[j]) > 0 else None,
                    "log2_exp_over_base": float(np.log2(v_exp[j] / v_base[j])) if min(v_exp[j], v_base[j]) > 0 else None,
                    "coverage_central90": float(cover[j]), "n_exp": int(len(E))})
    return out


def selftest():
    rng = np.random.default_rng(3)

    def build(link):
        rows = []
        for t in range(9):
            for j in range(int(rng.integers(3, 8))):
                typ = ("distance", "angle")[j % 2]
                lb = rng.normal(0, 2.0)
                le = rng.normal(0, 1.0)
                if link:
                    lgv = 0.3 * lb + 0.9 * le + rng.normal(0, 0.4)
                else:
                    lgv = rng.normal(0, 1.0)
                rows.append({"target": f"t{t}", "type": typ, "v_gen": float(np.exp(lgv)), "v_base": float(np.exp(lb)), "v_exp": float(np.exp(le))})
        return rows

    s1 = cohort_statistics(build(True), n_boot=1000)
    s2 = cohort_statistics(build(False), n_boot=1000)
    print("linked   : partial %.2f CI %s naive %.2f" % (s1["partial_spearman"], [round(x, 2) for x in s1["partial_spearman_ci95"]], s1["naive_spearman_of_ratios"]))
    print("unlinked : partial %.2f CI %s naive %.2f" % (s2["partial_spearman"], [round(x, 2) for x in s2["partial_spearman_ci95"]], s2["naive_spearman_of_ratios"]))
    assert s1["partial_spearman"] > 0.5 and s1["partial_spearman_ci95"][0] > 0
    assert abs(s2["partial_spearman"]) < 0.3 and s2["partial_spearman_ci95"][0] < 0 < s2["partial_spearman_ci95"][1]
    assert s2["naive_spearman_of_ratios"] > 0.4, "the shared-denominator artefact should be visible in the naive statistic"
    assert cohort_statistics(build(True)[:5])["testable"] is False

    # coordinate rows: wrapped torsions, pair variance, coverage
    Z = np.column_stack([rng.normal(179, 5, 100), rng.normal(10, 1, 100)])
    Z[:, 0] = ((Z[:, 0] + 180) % 360) - 180
    E = np.array([[-179.0, 9.0], [178.0, 12.0]])
    r = coordinate_rows("x", ["torsion[a]", "d[b]"], Z, np.array([True, False]), np.array([1.0, 1.0]), E)
    assert abs(r[0]["v_exp"] - (3.0 ** 2) / 4) < 1e-6, r[0]
    rm = coordinate_rows("x", ["torsion[a]", "d[b]"], Z, np.array([True, False]), np.array([1.0, 1.0]), E, centre="median")
    assert abs(rm[0]["v_exp"] - (3.0 ** 2) / 2) < 1e-6, rm[0]
    Y = rng.normal(5, 2, (30, 1)) + 40.0 * (np.arange(30)[:, None] > 26)
    a = coordinate_rows("x", ["d[c]"], Y, np.array([False]), np.array([1.0]), np.array([[5.0]] * 6 + [[9.0]]), centre="mean")[0]["v_gen"]
    assert abs(a - Y.var()) < 1e-9, "mean-centred generated variance must equal the 1/n variance"
    assert r[0]["v_gen"] < 50, "wrapped torsion variance must not be inflated by the +-180 seam"
    assert r[0]["coverage_central90"] == 1.0
    print("selftest passed")


def _mapped(models, input_seq):
    maps = [map_to_input(m, input_seq) for m in models]
    return maps, float(np.mean([len(m) / len(input_seq) for m in maps]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--out", type=Path, default=Path("experimental_variation_concordance.json"))
    ap.add_argument("--min-mapped", type=float, default=0.85,
                    help="minimum fraction of input residues that must map to the experimental structure")
    ap.add_argument("--centre", choices=("mean", "median"), default="mean",
                    help="mean: mean-centred 1/n variances (as in clearing.py); median: second moment about the median, divisor n - 1")
    ap.add_argument("--n-perturbations", type=int, default=300, help="perturbation attempts per target")
    ap.add_argument("--length-only-baseline", action="store_true",
                    help="accept perturbed vectors on length alone (default: require identical labels and stem identity)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.config and args.sample_root and args.bpmaps_root):
        ap.error("--config, --sample-root and --bpmaps-root are required")
    sys.path.insert(0, str(args.pipeline_dir))
    from collective_coords import extract_z
    from clearing import read_c1_coords, load_consensus_structure, noise_floor

    rows, per_target = [], {}
    for t in json.loads(args.config.read_text()):
        tid = t["id"]
        print(f"{tid} ({t['mode']}) ...")
        try:
            db = load_consensus_structure(tid, args.bpmaps_root / tid)
            fa = (args.bpmaps_root / tid / f"{tid}.fasta").read_text().split("\n")
            seq = "".join(l.strip() for l in fa if l and not l.startswith(">"))
            samples = sorted((args.sample_root / tid).glob("Sample_*.pdb"))
            zs = [labeled_z(read_c1_coords(f), db, extract_z) for f in samples]
            if t["mode"] == "pair":
                models = [read_models(t["apo_pdb"], t.get("apo_chain"))[0], read_models(t["holo_pdb"], t.get("holo_chain"))[0]]
                used = [t["apo_pdb"], t["holo_pdb"]]
                maps, frac = _mapped(models, seq)
            else:
                best = None
                for cand in t.get("pdb_candidates", [t.get("pdb")]):
                    try:
                        ms = read_models(cand, t.get("chain"))
                    except Exception as exc:  # candidate without usable RNA chain
                        print("   candidate", cand, "unusable:", exc); continue
                    mp, fr = _mapped(ms, seq)
                    print(f"   candidate {cand}: {fr:.2f} of input residues mapped")
                    if best is None or fr > best[1]:
                        best = (mp, fr, cand)
                if best is None:
                    raise ValueError("no usable experimental structure")
                maps, frac, used = best[0], best[1], [best[2]]
            if frac < args.min_mapped:
                per_target[tid] = {"testable": False, "reason": f"only {frac:.4f} of input residues mapped to the experimental structure (minimum {args.min_mapped})", "files": used}
                continue
            ez = [labeled_z(m, db, extract_z) for m in maps]
            common = [l for l in zs[0] if all(l in z for z in zs) and all(l in z for z in ez)]
            if len(common) < 2:
                per_target[tid] = {"testable": False, "reason": f"{len(common)} shared coordinates", "files": used}
                continue
            fv, fl, n_ok = noise_floor(read_c1_coords(samples[0]), db, n_perturbations=args.n_perturbations, correspondence=not args.length_only_baseline)
            if fv is None:
                per_target[tid] = {"testable": False, "reason": f"baseline not estimable ({n_ok}/{args.n_perturbations} accepted; at least 10 needed)", "files": used}
                continue
            vb = dict(zip(fl, fv))
            common = [l for l in common if l in vb]
            is_tors = np.array([l.startswith("torsion") for l in common])
            Z = np.array([[z[l] for l in common] for z in zs]); E = np.array([[z[l] for l in common] for z in ez])
            if t["mode"] != "pair" and len(E) < 5:
                per_target[tid] = {"testable": False, "reason": f"only {len(E)} experimental models", "files": used}
                continue
            r = coordinate_rows(tid, common, Z, is_tors, np.array([vb[l] for l in common]), E, centre=args.centre)
            rows.extend(r)
            per_target[tid] = {"testable": True, "files": used, "mapped_fraction": float(frac), "n_coordinates": len(common), "n_experimental": int(len(E)), "baseline_valid_perturbations": int(n_ok),
                               "mean_coverage_central90": float(np.mean([x["coverage_central90"] for x in r]))}
            print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in per_target[tid].items() if k != "files"})
        except Exception as exc:
            per_target[tid] = {"testable": False, "reason": f"{type(exc).__name__}: {exc}"}
            print("   skipped:", exc)
    cohort = cohort_statistics([{"target": r["target"], "type": r["type"], "v_gen": r["v_gen"], "v_base": r["v_base"], "v_exp": r["v_exp"]} for r in rows])
    out = {"centre": args.centre, "per_target": per_target, "cohort": cohort, "coordinates": rows}
    args.out.write_text(json.dumps(out, indent=2))
    print("cohort:", json.dumps(cohort))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
