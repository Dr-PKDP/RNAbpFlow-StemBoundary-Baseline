"""Structural quality of the generated samples, and clearing recomputed on the passing subset.

For every Sample_*.pdb of a target this reports three fixed quality measures:

  chain_breaks      consecutive residues whose O3'(i)-P(i+1) distance leaves [1.2, 2.0] A,
                    or that lack one of the two atoms. If no residue of a model carries a P
                    atom, the C1'(i)-C1'(i+1) distance is used instead with the range
                    [4.0, 7.5] A, and "backbone_criterion" records which test was applied.
  clashes           heavy-atom pairs closer than 2.2 A whose residues are at least two
                    positions apart.
  bp_satisfied      fraction of the conditioning map's base pairs whose Watson-Crick edge
                    nitrogens (N1 for A and G, N3 for C and U) are within 3.5 A.

The thresholds are set in CRITERIA. Samples that meet all three enter the
filtered ensemble, for which the clearing calls of clearing.py are recomputed against the
same perturbation baseline, so a call that depends on malformed samples shows up as a
changed call.

Usage:
  python3 sample_structure_qc.py --sample-root <Predictions> --bpmaps-root <Inputs> \
      --pipeline-dir <dir with clearing.py and collective_coords.py> --out qc.json
  python3 sample_structure_qc.py --selftest
"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np

CRITERIA = {"max_chain_breaks": 0, "max_clashes": 0, "min_bp_satisfied": 0.80}
CLASH_A = 2.2
WC_ATOM = {"A": "N1", "G": "N1", "C": "N3", "U": "N3", "T": "N3"}


def read_residues(pdb_path: Path):
    """Residues in order of first appearance, as (resname, {atom_name: xyz})."""
    order, data = [], {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            element = line[76:78].strip().upper()
            atom = line[12:16].strip()
            if element == "H" or (not element and atom.startswith("H")):
                continue
            key = (line[21], line[22:26].strip(), line[26])
            if key not in data:
                order.append(key)
                data[key] = (line[17:20].strip(), {})
            data[key][1][atom] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return [data[k] for k in order]


def count_chain_breaks(residues):
    has_p = any("P" in atoms for _, atoms in residues)
    if has_p:
        lo, hi, a, b, name = 1.2, 2.0, "O3'", "P", "O3'-P"
    else:
        lo, hi, a, b, name = 4.0, 7.5, "C1'", "C1'", "C1'-C1'"
    breaks = 0
    for (_, x), (_, y) in zip(residues, residues[1:]):
        if a not in x or b not in y:
            breaks += 1
            continue
        if not lo <= float(np.linalg.norm(x[a] - y[b])) <= hi:
            breaks += 1
    return breaks, name


def count_clashes(residues, cutoff=CLASH_A, min_sep=2):
    pts, owner = [], []
    for i, (_, atoms) in enumerate(residues):
        for xyz in atoms.values():
            pts.append(xyz)
            owner.append(i)
    if not pts:
        return 0
    pts = np.asarray(pts, dtype=float)
    owner = np.asarray(owner)
    cells = {}
    keys = np.floor(pts / cutoff).astype(int)
    for n, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(n)
    seen = 0
    for n, (i, j, k) in enumerate(map(tuple, keys)):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for m in cells.get((i + di, j + dj, k + dk), ()):
                        if m <= n or abs(owner[m] - owner[n]) < min_sep:
                            continue
                        if float(np.linalg.norm(pts[m] - pts[n])) < cutoff:
                            seen += 1
    return seen


def bp_satisfied(residues, pairs, cutoff=3.5):
    if not pairs:
        return None, 0
    ok = 0
    for i, j in pairs:
        if i >= len(residues) or j >= len(residues):
            continue
        ra, aa = residues[i]
        rb, ab = residues[j]
        na, nb = WC_ATOM.get(ra[-1:].upper()), WC_ATOM.get(rb[-1:].upper())
        if na in aa and nb in ab and float(np.linalg.norm(aa[na] - ab[nb])) <= cutoff:
            ok += 1
    return ok / len(pairs), len(pairs)


def qc_sample(pdb_path: Path, pairs):
    residues = read_residues(pdb_path)
    breaks, criterion = count_chain_breaks(residues)
    frac, n_pairs = bp_satisfied(residues, pairs)
    return {"sample": pdb_path.name, "n_residues": len(residues), "chain_breaks": breaks,
            "backbone_criterion": criterion, "clashes": count_clashes(residues),
            "bp_satisfied": frac, "n_conditioning_pairs": n_pairs}


def passes(q, crit=None):
    c = crit or CRITERIA
    return (q["chain_breaks"] <= c["max_chain_breaks"]
            and q["clashes"] <= c["max_clashes"]
            and (q["bp_satisfied"] is None or q["bp_satisfied"] >= c["min_bp_satisfied"]))


def calls(z_arr, floor_var):
    v = z_arr.var(axis=0)
    return [bool(s > f) if f > 1e-6 else None for s, f in zip(v, floor_var)], v


def run(args):
    sys.path.insert(0, str(args.pipeline_dir))
    from collective_coords import extract_z, parse_dotbracket
    from clearing import read_c1_coords, load_consensus_structure, noise_floor

    targets = args.targets or sorted(p.name for p in args.sample_root.iterdir() if p.is_dir())
    crit = {"max_chain_breaks": args.max_chain_breaks, "max_clashes": args.max_clashes,
            "min_bp_satisfied": args.min_bp_satisfied}
    out = {"criteria": crit, "criteria_are_defaults": crit == CRITERIA,
           "clash_cutoff_angstrom": CLASH_A, "targets": {}}
    for tid in targets:
        sample_dir = args.sample_root / tid
        samples = sorted(sample_dir.glob("Sample_*.pdb"))
        if not samples:
            out["targets"][tid] = {"testable": False, "reason": "no Sample_*.pdb"}
            continue
        db = load_consensus_structure(tid, args.bpmaps_root / tid)
        pairs = parse_dotbracket(db)
        per = [qc_sample(f, pairs) for f in samples]
        keep = [i for i, q in enumerate(per) if passes(q, crit)]
        entry = {"n_samples": len(samples), "n_passing": len(keep),
                 "median_clashes": float(np.median([q["clashes"] for q in per])),
                 "max_chain_breaks": int(max(q["chain_breaks"] for q in per)),
                 "median_bp_satisfied": (None if per[0]["bp_satisfied"] is None
                                         else float(np.median([q["bp_satisfied"] for q in per]))),
                 "per_sample": per}
        z_arr = np.array([extract_z(read_c1_coords(f), db)[0] for f in samples])
        labels = extract_z(read_c1_coords(samples[0]), db)[1]
        floor_var, floor_labels, n_ok = noise_floor(read_c1_coords(samples[0]), db,
                                                    n_perturbations=args.n_perturbations)
        entry["baseline_valid_perturbations"] = n_ok
        if floor_var is None:
            entry["clearing"] = {"testable": False, "reason": f"baseline not estimable ({n_ok} valid)"}
        elif len(keep) < args.min_passing:
            entry["clearing"] = {"testable": False,
                                 "reason": f"{len(keep)} passing samples (need >= {args.min_passing})"}
        else:
            all_calls, v_all = calls(z_arr, floor_var)
            keep_calls, v_keep = calls(z_arr[keep], floor_var)
            changed = [labels[i] for i in range(len(labels)) if all_calls[i] != keep_calls[i]]
            entry["clearing"] = {
                "testable": True, "coordinate_labels": labels,
                "cleared_all_samples": int(sum(1 for c in all_calls if c)),
                "cleared_passing_samples": int(sum(1 for c in keep_calls if c)),
                "coordinates_with_changed_call": changed,
                "max_abs_log2_variance_change": float(np.max(np.abs(np.log2(
                    np.where(v_keep > 0, v_keep, np.nan) / np.where(v_all > 0, v_all, np.nan))))),
            }
        out["targets"][tid] = entry
        c = entry.get("clearing", {})
        print(f"{tid}: {len(keep)}/{len(samples)} samples pass; "
              f"cleared {c.get('cleared_all_samples')} -> {c.get('cleared_passing_samples')}; "
              f"changed calls {len(c.get('coordinates_with_changed_call', []))}", flush=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


def selftest():
    def atom(serial, name, resname, resseq, xyz, element):
        return (f"ATOM  {serial:>5d} {name:<4s}{resname:>4s} A{resseq:>4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.pdb"
        lines, serial = [], 1
        for i in range(3):
            base = np.array([0.0, 0.0, 6.0 * i])
            for nm, off, el in (("P", [0, 0, 0], "P"), ("O3'", [0, 0, 4.4], "O"),
                                ("C1'", [2.0, 0, 0], "C"), ("N1", [3.0, 0, 0], "N")):
                lines.append(atom(serial, nm, "A", i + 1, base + np.array(off, dtype=float), el))
                serial += 1
        p.write_text("".join(lines))
        res = read_residues(p)
        assert len(res) == 3 and res[0][0] == "A", res
        breaks, crit = count_chain_breaks(res)
        assert crit == "O3'-P" and breaks == 0, (crit, breaks)
        assert count_clashes(res) == 0, count_clashes(res)
        frac, n = bp_satisfied(res, [(0, 2)])
        assert n == 1 and frac == 0.0, (frac, n)
        near = [("A", {"N1": np.zeros(3)}), ("C", {"N3": np.array([3.0, 0.0, 0.0])}),
                ("G", {"N1": np.array([40.0, 0.0, 0.0])})]
        assert bp_satisfied(near, [(0, 1)])[0] == 1.0
        assert bp_satisfied(near, [(0, 2)])[0] == 0.0
        q = qc_sample(p, [(0, 2)])
        assert not passes(q) and passes({**q, "bp_satisfied": 1.0}), q
        clashy = res + [("A", {"C1'": res[0][1]["C1'"] + np.array([0.5, 0.0, 0.0])})]
        assert count_clashes(clashy) == 2, count_clashes(clashy)
        cl, _ = calls(np.array([[1.0, 5.0], [3.0, 5.0]]), np.array([0.5, 0.5]))
        assert cl == [True, False], cl
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--n-perturbations", type=int, default=300)
    ap.add_argument("--min-passing", type=int, default=10)
    ap.add_argument("--max-chain-breaks", type=int, default=CRITERIA["max_chain_breaks"])
    ap.add_argument("--max-clashes", type=int, default=CRITERIA["max_clashes"])
    ap.add_argument("--min-bp-satisfied", type=float, default=CRITERIA["min_bp_satisfied"])
    ap.add_argument("--out", type=Path, default=Path("sample_structure_qc.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.sample_root and args.bpmaps_root):
        ap.error("--sample-root and --bpmaps-root are required")
    run(args)


if __name__ == "__main__":
    main()
