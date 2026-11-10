"""Dependence of the perturbation baseline on the generated structure used as its coordinate substrate.

The baseline of clearing.py is computed from the base-pair map of a target and the C1'
coordinates of one generated sample, the first of the ensemble. This repeats that
estimation with several samples as the substrate, keeping the perturbation seed and the
attempt budget fixed, and reports

  - the baseline SD of each coordinate across substrates, as a range and a coefficient of
    variation,
  - the distance and angle clearing proportions obtained with each substrate, and
  - the coordinates whose clearing call changes relative to the first sample.

Usage:
  python3 baseline_substrate_sensitivity.py --sample-root <Predictions> --bpmaps-root <Inputs> \
      --pipeline-dir <dir with clearing.py and collective_coords.py> --out substrate.json
  python3 baseline_substrate_sensitivity.py --selftest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


def summarise(sds):
    """sds: (n_substrates, n_coordinates) baseline standard deviations."""
    lo, hi = sds.min(axis=0), sds.max(axis=0)
    mean = sds.mean(axis=0)
    cv = np.where(mean > 0, sds.std(axis=0) / np.where(mean > 0, mean, 1.0), np.nan)
    ratio = np.where(lo > 0, hi / np.where(lo > 0, lo, np.nan), np.inf)
    return lo, hi, cv, ratio


def run(args):
    sys.path.insert(0, str(args.pipeline_dir))
    from collective_coords import extract_z
    from clearing import read_c1_coords, load_consensus_structure, noise_floor

    targets = args.targets or sorted(p.name for p in args.sample_root.iterdir() if p.is_dir())
    out = {"n_perturbations": args.n_perturbations, "n_substrates": args.n_substrates, "targets": {}}
    for tid in targets:
        samples = sorted((args.sample_root / tid).glob("Sample_*.pdb"))
        if not samples:
            out["targets"][tid] = {"testable": False, "reason": "no Sample_*.pdb"}
            continue
        db = load_consensus_structure(tid, args.bpmaps_root / tid)
        coords = [read_c1_coords(f) for f in samples]
        z_arr = np.array([extract_z(c, db)[0] for c in coords])
        labels = extract_z(coords[0], db)[1]
        v_gen = z_arr.var(axis=0)
        used, sds, props, calls = [], [], [], []
        for k in range(min(args.n_substrates, len(samples))):
            fv, flabels, n_ok = noise_floor(coords[k], db, n_perturbations=args.n_perturbations)
            if fv is None:
                continue
            assert list(flabels) == list(labels), f"{tid}: coordinate labels differ for substrate {k}"
            used.append({"substrate": samples[k].name, "valid_perturbations": n_ok})
            sds.append(np.sqrt(fv))
            c = [bool(s > f) if f > 1e-6 else None for s, f in zip(v_gen, fv)]
            calls.append(c)
            d = [c[i] for i, l in enumerate(labels) if l.startswith("d[")]
            a = [c[i] for i, l in enumerate(labels) if l.startswith("theta[")]
            props.append({"distance_proportion": (sum(1 for x in d if x) / len(d)) if d else None,
                          "angle_proportion": (sum(1 for x in a if x) / len(a)) if a else None})
        if len(used) < 2:
            out["targets"][tid] = {"testable": False,
                                   "reason": f"baseline estimable for {len(used)} substrates"}
            print(f"{tid}: baseline estimable for {len(used)} substrates", flush=True)
            continue
        sds = np.array(sds)
        lo, hi, cv, ratio = summarise(sds)
        changed = sorted({labels[i] for c in calls[1:] for i in range(len(labels)) if c[i] != calls[0][i]})
        out["targets"][tid] = {
            "testable": True, "coordinate_labels": labels, "substrates": used,
            "baseline_sd_min": lo.tolist(), "baseline_sd_max": hi.tolist(),
            "baseline_sd_cv": [None if np.isnan(x) else float(x) for x in cv],
            "max_sd_ratio_across_substrates": float(np.nanmax(ratio)),
            "max_sd_cv_across_substrates": float(np.nanmax(cv)),
            "clearing_proportions_per_substrate": props,
            "coordinates_with_changed_call": changed,
        }
        print(f"{tid}: {len(used)} substrates; max baseline-SD ratio "
              f"{np.nanmax(ratio):.2f}, max CV {np.nanmax(cv):.2f}; "
              f"changed calls {len(changed)}", flush=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


def selftest():
    sds = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    lo, hi, cv, ratio = summarise(sds)
    assert list(lo) == [1.0, 2.0] and list(hi) == [3.0, 2.0], (lo, hi)
    assert abs(cv[0] - np.std([1, 2, 3]) / 2.0) < 1e-12 and cv[1] == 0.0, cv
    assert abs(float(np.nanmax(ratio)) - 3.0) < 1e-12, ratio
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--n-substrates", type=int, default=5)
    ap.add_argument("--n-perturbations", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("baseline_substrate_sensitivity.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.sample_root and args.bpmaps_root):
        ap.error("--sample-root and --bpmaps-root are required")
    run(args)


if __name__ == "__main__":
    main()
