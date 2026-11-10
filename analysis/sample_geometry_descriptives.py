"""Backbone and C1' linkage geometry of the generated samples, and clearing on graded subsets.

The coordinates of collective_coords.py are computed from C1' positions, so besides the
O3'(i)-P(i+1) linkage reported by sample_structure_qc.py this reports the consecutive
C1'(i)-C1'(i+1) distance, for every sample of a target:

  o3p_quantiles        5, 25, 50, 75 and 95th percentiles of the O3'-P distance over all linkages
  o3p_fraction_*       fraction of linkages within [1.2, 2.0] A, in (2.0, 3.0] A and above 3.0 A
  c1c1_quantiles       the same percentiles for the C1'-C1' distance
  c1c1_fraction_in_window   fraction of linkages within [4.0, 7.5] A
  samples_all_c1c1_in_window   samples in which every C1'-C1' linkage is within [4.0, 7.5] A

Clearing calls (clearing.py, same perturbation baseline as the main analysis) are then
recomputed on two graded subsets, which are descriptive and graded alternatives to the fixed
criteria of sample_structure_qc.py:

  fewest_breaks        the --keep samples with the fewest O3'-P chain breaks (ties: fewest clashes)
  c1c1_continuous      samples in which every C1'-C1' linkage is within [4.0, 7.5] A (>= --min-subset)

Usage:
  python3 sample_geometry_descriptives.py --sample-root <Predictions> --bpmaps-root <Inputs> \
      --pipeline-dir <dir with clearing.py and collective_coords.py> --out geometry.json
  python3 sample_geometry_descriptives.py --selftest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_structure_qc import read_residues, count_chain_breaks, count_clashes  # noqa: E402

O3P_WINDOW = (1.2, 2.0)
C1C1_WINDOW = (4.0, 7.5)
QUANTILES = (5, 25, 50, 75, 95)


def linkage_distances(residues, a, b):
    """Distances a(i)-b(i+1) over consecutive residues; NaN where an atom is missing."""
    out = []
    for (_, x), (_, y) in zip(residues, residues[1:]):
        out.append(float(np.linalg.norm(x[a] - y[b])) if a in x and b in y else np.nan)
    return np.array(out)


def describe(dist, window):
    ok = dist[~np.isnan(dist)]
    q = {f"p{p}": float(np.percentile(ok, p)) for p in QUANTILES} if ok.size else {}
    inside = float(np.mean((ok >= window[0]) & (ok <= window[1]))) if ok.size else None
    return q, inside, ok


def calls(z_arr, floor_var):
    v = z_arr.var(axis=0)
    return [bool(s > f) if f > 1e-6 else None for s, f in zip(v, floor_var)]


def class_props(cl, labels):
    d = [c for c, l in zip(cl, labels) if l.startswith("d[") and c is not None]
    t = [c for c, l in zip(cl, labels) if l.startswith("theta[") and c is not None]
    return (float(np.mean(d)) if d else None, float(np.mean(t)) if t else None)


def subset_entry(name, idx, z_arr, floor_var, labels, base_calls):
    sub = calls(z_arr[idx], floor_var)
    dp, ap = class_props(sub, labels)
    return {"subset": name, "n_samples": len(idx), "distance_proportion": dp, "angle_proportion": ap,
            "coordinates_with_changed_call": [l for l, a, b in zip(labels, base_calls, sub) if a != b]}


def run(args):
    sys.path.insert(0, str(args.pipeline_dir))
    from collective_coords import extract_z, parse_dotbracket
    from clearing import read_c1_coords, load_consensus_structure, noise_floor

    targets = args.targets or sorted(p.name for p in args.sample_root.iterdir() if p.is_dir())
    out = {"o3p_window_A": O3P_WINDOW, "c1c1_window_A": C1C1_WINDOW, "targets": {}}
    for tid in targets:
        samples = sorted((args.sample_root / tid).glob("Sample_*.pdb"))
        if not samples:
            out["targets"][tid] = {"reason": "no Sample_*.pdb"}
            continue
        o3p, c1c1, per = [], [], []
        for f in samples:
            res = read_residues(f)
            d1, d2 = linkage_distances(res, "O3'", "P"), linkage_distances(res, "C1'", "C1'")
            o3p.append(d1[~np.isnan(d1)])
            c1c1.append(d2[~np.isnan(d2)])
            ok = (~np.isnan(d2)) & (d2 >= C1C1_WINDOW[0]) & (d2 <= C1C1_WINDOW[1])
            per.append({"sample": f.name, "o3p_breaks": count_chain_breaks(res)[0], "clashes": count_clashes(res),
                        "c1c1_linkages": int(len(d2)), "c1c1_out_of_window": int(len(d2) - ok.sum()),
                        "c1c1_continuous": bool(ok.all())})
        o3p_all, c1c1_all = np.concatenate(o3p), np.concatenate(c1c1)
        qo, io, _ = describe(o3p_all, O3P_WINDOW)
        qc, ic, _ = describe(c1c1_all, C1C1_WINDOW)
        entry = {"n_samples": len(samples), "o3p_quantiles": qo,
                 "o3p_fraction_in_window": io,
                 "o3p_fraction_2_to_3": float(np.mean((o3p_all > 2.0) & (o3p_all <= 3.0))),
                 "o3p_fraction_above_3": float(np.mean(o3p_all > 3.0)),
                 "c1c1_quantiles": qc, "c1c1_fraction_in_window": ic,
                 "samples_all_c1c1_in_window": int(sum(p["c1c1_continuous"] for p in per)),
                 "per_sample": per}
        db = load_consensus_structure(tid, args.bpmaps_root / tid)
        z_arr = np.array([extract_z(read_c1_coords(f), db)[0] for f in samples])
        labels = extract_z(read_c1_coords(samples[0]), db)[1]
        floor_var, _, n_ok = noise_floor(read_c1_coords(samples[0]), db, n_perturbations=args.n_perturbations)
        if floor_var is None:
            entry["clearing"] = {"testable": False, "reason": f"baseline not estimable ({n_ok} valid)"}
        else:
            base = calls(z_arr, floor_var)
            dp, ap = class_props(base, labels)
            order = sorted(range(len(samples)), key=lambda i: (per[i]["o3p_breaks"], per[i]["clashes"]))
            subs = [{"subset": "all", "n_samples": len(samples), "distance_proportion": dp,
                     "angle_proportion": ap, "coordinates_with_changed_call": []},
                    subset_entry("fewest_breaks", sorted(order[:args.keep]), z_arr, floor_var, labels, base)]
            cont = [i for i, p in enumerate(per) if p["c1c1_continuous"]]
            if len(cont) >= args.min_subset:
                subs.append(subset_entry("c1c1_continuous", cont, z_arr, floor_var, labels, base))
            else:
                subs.append({"subset": "c1c1_continuous", "n_samples": len(cont),
                             "reason": f"fewer than {args.min_subset} samples"})
            entry["clearing"] = {"testable": True, "coordinate_labels": labels, "subsets": subs}
        out["targets"][tid] = entry
        print(f"{tid}: O3'-P median {qo.get('p50', float('nan')):.2f} A, {io:.2f} in window; "
              f"C1'-C1' median {qc.get('p50', float('nan')):.2f} A, {ic:.2f} in window; "
              f"{entry['samples_all_c1c1_in_window']}/{len(samples)} samples continuous", flush=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


def selftest():
    x = [("A", {"O3'": np.array([0.0, 0, 0]), "C1'": np.array([0.0, 0, 5.0])}),
         ("A", {"P": np.array([1.6, 0, 0]), "O3'": np.array([9.0, 0, 0]), "C1'": np.array([0.0, 0, 10.5])}),
         ("A", {"P": np.array([20.0, 0, 0]), "C1'": np.array([0.0, 0, 20.0])})]
    d = linkage_distances(x, "O3'", "P")
    assert abs(d[0] - 1.6) < 1e-9 and abs(d[1] - 11.0) < 1e-9, d
    c = linkage_distances(x, "C1'", "C1'")
    assert np.allclose(c, [5.5, 9.5]), c
    q, inside, ok = describe(c, C1C1_WINDOW)
    assert inside == 0.5 and q["p50"] == 7.5, (inside, q)
    assert np.isnan(linkage_distances([x[0], ("A", {})], "O3'", "P")[0])
    assert class_props([True, False, True, None], ["d[a]", "d[b]", "theta[a]", "theta[b]"]) == (0.5, 1.0)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--n-perturbations", type=int, default=300)
    ap.add_argument("--keep", type=int, default=50)
    ap.add_argument("--min-subset", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("sample_geometry_descriptives.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.sample_root and args.bpmaps_root):
        ap.error("--sample-root and --bpmaps-root are required")
    run(args)


if __name__ == "__main__":
    main()
