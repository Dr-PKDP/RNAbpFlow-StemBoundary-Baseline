"""Control for selecting generated samples by backbone quality.

Retaining the --keep samples with the fewest O3'-P chain breaks changes some clearing calls
(sample_geometry_descriptives.py). A change could come from the selection on chain breaks or
from the smaller number of samples alone. This compares the fewest-breaks subset with
--n-draws random subsets of the same size, and with the subset of the most chain breaks,
for every target with an estimable baseline:

  observed_*          distance and angle clearing proportions, their difference (angle - distance)
                      and the number of changed calls for the fewest-breaks and most-breaks subsets
  random_*            the same quantities over the random subsets (mean, 2.5th and 97.5th percentile)
  mean_delta_*        the difference averaged over targets, for the fewest-breaks subset, the
                      most-breaks subset and the random subsets, with the fraction of random
                      subsets whose mean difference is at least as large as the fewest-breaks one

Usage:
  python3 subset_selection_control.py --sample-root <Predictions> --bpmaps-root <Inputs> \
      --pipeline-dir <dir with clearing.py and collective_coords.py> --out control.json
  python3 subset_selection_control.py --selftest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_structure_qc import read_residues, count_chain_breaks, count_clashes  # noqa: E402
from sample_geometry_descriptives import calls, class_props  # noqa: E402


def subset_stats(z_arr, idx, floor_var, labels, base_calls):
    sub = calls(z_arr[idx], floor_var)
    dp, ap = class_props(sub, labels)
    changed = sum(1 for a, b in zip(base_calls, sub) if a != b)
    return dp, ap, changed


def pct(a, q):
    return float(np.percentile(a, q))


def run(args):
    sys.path.insert(0, str(args.pipeline_dir))
    from collective_coords import extract_z
    from clearing import read_c1_coords, load_consensus_structure, noise_floor

    rng = np.random.default_rng(args.seed)
    targets = args.targets or sorted(p.name for p in args.sample_root.iterdir() if p.is_dir())
    out = {"keep": args.keep, "n_draws": args.n_draws, "seed": args.seed, "targets": {}}
    obs_fewest, obs_most, rand_delta = [], [], []
    for tid in targets:
        samples = sorted((args.sample_root / tid).glob("Sample_*.pdb"))
        if not samples:
            continue
        db = load_consensus_structure(tid, args.bpmaps_root / tid)
        floor_var, _, n_ok = noise_floor(read_c1_coords(samples[0]), db, n_perturbations=args.n_perturbations)
        if floor_var is None:
            out["targets"][tid] = {"testable": False, "reason": f"baseline not estimable ({n_ok} valid)"}
            continue
        z_arr = np.array([extract_z(read_c1_coords(f), db)[0] for f in samples])
        labels = extract_z(read_c1_coords(samples[0]), db)[1]
        base = calls(z_arr, floor_var)
        qual = []
        for f in samples:
            res = read_residues(f)
            qual.append((count_chain_breaks(res)[0], count_clashes(res)))
        order = sorted(range(len(samples)), key=lambda i: qual[i])
        fewest, most = sorted(order[:args.keep]), sorted(order[-args.keep:])
        f_dp, f_ap, f_ch = subset_stats(z_arr, fewest, floor_var, labels, base)
        m_dp, m_ap, m_ch = subset_stats(z_arr, most, floor_var, labels, base)
        draws = []
        for _ in range(args.n_draws):
            idx = np.sort(rng.choice(len(samples), size=args.keep, replace=False))
            draws.append(subset_stats(z_arr, idx, floor_var, labels, base))
        dd = np.array([(a - d) for d, a, _ in draws])
        ch = np.array([c for _, _, c in draws])
        b_dp, b_ap = class_props(base, labels)
        out["targets"][tid] = {
            "testable": True, "n_coordinates": len(labels),
            "all_samples": {"distance": b_dp, "angle": b_ap, "delta": b_ap - b_dp},
            "fewest_breaks": {"distance": f_dp, "angle": f_ap, "delta": f_ap - f_dp, "changed_calls": f_ch},
            "most_breaks": {"distance": m_dp, "angle": m_ap, "delta": m_ap - m_dp, "changed_calls": m_ch},
            "random": {"delta_mean": float(dd.mean()), "delta_p2.5": pct(dd, 2.5), "delta_p97.5": pct(dd, 97.5),
                       "changed_calls_mean": float(ch.mean()), "changed_calls_p97.5": pct(ch, 97.5),
                       "fraction_draws_changed_calls_ge_fewest": float(np.mean(ch >= f_ch))},
        }
        obs_fewest.append(f_ap - f_dp)
        obs_most.append(m_ap - m_dp)
        rand_delta.append(dd)
        print(f"{tid}: delta all {b_ap - b_dp:+.3f}, fewest {f_ap - f_dp:+.3f} (changed {f_ch}), "
              f"most {m_ap - m_dp:+.3f} (changed {m_ch}); random mean {dd.mean():+.3f} "
              f"[{pct(dd, 2.5):+.3f}, {pct(dd, 97.5):+.3f}], changed mean {ch.mean():.2f}", flush=True)
    if obs_fewest:
        rd = np.mean(np.array(rand_delta), axis=0)
        out["mean_delta_over_targets"] = {
            "n_targets": len(obs_fewest),
            "fewest_breaks": float(np.mean(obs_fewest)), "most_breaks": float(np.mean(obs_most)),
            "random_mean": float(rd.mean()), "random_p2.5": pct(rd, 2.5), "random_p97.5": pct(rd, 97.5),
            "fraction_random_ge_fewest": float(np.mean(rd >= np.mean(obs_fewest))),
            "fraction_random_le_most": float(np.mean(rd <= np.mean(obs_most))),
        }
        print("mean delta over targets:", json.dumps(out["mean_delta_over_targets"]))
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


def selftest():
    rng = np.random.default_rng(1)
    z = np.column_stack([rng.normal(0, 3.0, 40), rng.normal(0, 0.5, 40)])
    floor = np.array([1.0, 1.0])
    base = calls(z, floor)
    assert base == [True, False], base
    dp, ap, ch = subset_stats(z, np.arange(40), floor, ["d[a]", "theta[a]"], base)
    assert (dp, ap, ch) == (1.0, 0.0, 0), (dp, ap, ch)
    z2 = z.copy()
    z2[:20, 0] = 0.0
    dp, ap, ch = subset_stats(z2, np.arange(20), floor, ["d[a]", "theta[a]"], base)
    assert dp == 0.0 and ch == 1, (dp, ch)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--n-perturbations", type=int, default=300)
    ap.add_argument("--keep", type=int, default=50)
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("subset_selection_control.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.sample_root and args.bpmaps_root):
        ap.error("--sample-root and --bpmaps-root are required")
    run(args)


if __name__ == "__main__":
    main()
