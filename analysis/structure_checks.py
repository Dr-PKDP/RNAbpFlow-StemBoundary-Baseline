"""Structural checks of generated (or relaxed) samples, with the backbone geometry split by type.

For every Sample_*.pdb of every target directory under --sample-root this reports

  O3'(i)-P(i+1) linkage geometry, each linkage classed as
      compressed      below 1.2 A
      in range        1.2-2.0 A
      stretched       2.0-3.0 A
      extended        above 3.0 A (or an atom missing), counted as a chain break
  clashes             heavy-atom pairs closer than 2.2 A in residues at least two positions apart
  wc_n1n3             fraction of conditioning base pairs whose Watson-Crick edge nitrogens (N1 of A and G,
                      N3 of C and U) are within 3.5 A: the base-pair measure of sample_structure_qc.py
  hbond_pairs         fraction of conditioning base pairs with at least two canonical or wobble donor-acceptor
                      distances within 3.5 A (A-U: N1-N3, N6-O4; G-C: N1-N3, N2-O2, O6-N4; G-U: N1-O2, O6-N3)
  c1c1_out_of_window  consecutive C1'-C1' distances outside 4.0-7.5 A

and whether the sample meets the fixed criteria of sample_structure_qc.py (no O3'-P linkage outside 1.2-2.0 A,
no clash, wc_n1n3 >= 0.80). The per-target list of passing samples can be passed to ensemble_analyses.py.

Usage:
  python3 structure_checks.py --sample-root <dir> --bpmaps-root <Inputs> --pipeline-dir <pipeline> --out checks.json
  python3 structure_checks.py --selftest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_structure_qc import read_residues, count_clashes, bp_satisfied, CRITERIA  # noqa: E402

HBOND = {
    ("A", "U"): [("N1", "N3"), ("N6", "O4")],
    ("G", "C"): [("N1", "N3"), ("N2", "O2"), ("O6", "N4")],
    ("G", "U"): [("N1", "O2"), ("O6", "N3")],
}
HBOND_A = 3.5
LINK = (1.2, 2.0, 3.0)
C1C1 = (4.0, 7.5)


def base_letter(resname):
    r = resname.strip().upper()
    return {"ADE": "A", "CYT": "C", "GUA": "G", "URA": "U", "RA": "A", "RC": "C", "RG": "G", "RU": "U"}.get(r, r[-1:])


def pair_formed(res_a, res_b, cutoff=HBOND_A):
    """None when the pair is not canonical or wobble, else True when at least two H-bond distances are within cutoff."""
    (na, aa), (nb, ab) = res_a, res_b
    a, b = base_letter(na), base_letter(nb)
    if (a, b) in HBOND:
        spec, x, y = HBOND[(a, b)], aa, ab
    elif (b, a) in HBOND:
        spec, x, y = HBOND[(b, a)], ab, aa
    else:
        return None
    n = sum(1 for p, q in spec if p in x and q in y and float(np.linalg.norm(x[p] - y[q])) <= cutoff)
    return n >= 2


def hbond_fraction(residues, pairs):
    calls = [pair_formed(residues[i], residues[j]) for i, j in pairs if i < len(residues) and j < len(residues)]
    calls = [c for c in calls if c is not None]
    return (float(np.mean(calls)) if calls else None), len(calls)


def linkage_classes(residues):
    c = {"compressed": 0, "in_range": 0, "stretched": 0, "extended": 0}
    for (_, x), (_, y) in zip(residues, residues[1:]):
        if "O3'" not in x or "P" not in y:
            c["extended"] += 1
            continue
        d = float(np.linalg.norm(x["O3'"] - y["P"]))
        if d < LINK[0]:
            c["compressed"] += 1
        elif d <= LINK[1]:
            c["in_range"] += 1
        elif d <= LINK[2]:
            c["stretched"] += 1
        else:
            c["extended"] += 1
    return c


def c1c1_out(residues):
    n = 0
    for (_, x), (_, y) in zip(residues, residues[1:]):
        if "C1'" not in x or "C1'" not in y:
            n += 1
            continue
        d = float(np.linalg.norm(x["C1'"] - y["C1'"]))
        n += int(not C1C1[0] <= d <= C1C1[1])
    return n


def check_sample(path, pairs):
    res = read_residues(path)
    lc = linkage_classes(res)
    wc, n_pairs = bp_satisfied(res, pairs)
    hb, n_hb = hbond_fraction(res, pairs)
    cl = count_clashes(res)
    out_of_range = lc["compressed"] + lc["stretched"] + lc["extended"]
    passes = (out_of_range <= CRITERIA["max_chain_breaks"] and cl <= CRITERIA["max_clashes"]
              and (wc is None or wc >= CRITERIA["min_bp_satisfied"]))
    return {"sample": path.name, "n_residues": len(res), **lc, "out_of_range": out_of_range, "clashes": cl,
            "wc_n1n3": wc, "hbond_pairs": hb, "n_pairs_assessed": n_hb, "c1c1_out_of_window": c1c1_out(res),
            "passes_fixed_criteria": bool(passes)}


def summarise(per):
    def med(k):
        v = [p[k] for p in per if p[k] is not None]
        return float(np.median(v)) if v else None
    n_link = sum(p["compressed"] + p["in_range"] + p["stretched"] + p["extended"] for p in per)
    frac = lambda k: (sum(p[k] for p in per) / n_link) if n_link else None
    return {"n_samples": len(per), "n_passing": sum(p["passes_fixed_criteria"] for p in per),
            "passing": [p["sample"] for p in per if p["passes_fixed_criteria"]],
            "fraction_compressed": frac("compressed"), "fraction_in_range": frac("in_range"),
            "fraction_stretched": frac("stretched"), "fraction_extended": frac("extended"),
            "samples_without_extended": sum(1 for p in per if p["extended"] == 0),
            "samples_without_clash": sum(1 for p in per if p["clashes"] == 0),
            "samples_c1c1_continuous": sum(1 for p in per if p["c1c1_out_of_window"] == 0),
            "median_out_of_range": med("out_of_range"), "median_extended": med("extended"),
            "median_clashes": med("clashes"), "median_wc_n1n3": med("wc_n1n3"), "median_hbond_pairs": med("hbond_pairs")}


def run(args):
    sys.path.insert(0, str(args.pipeline_dir))
    from collective_coords import parse_dotbracket
    from clearing import load_consensus_structure
    targets = args.targets or sorted(p.name for p in args.sample_root.iterdir() if p.is_dir())
    out = {"sample_root": str(args.sample_root), "criteria": CRITERIA, "targets": {}}
    for tid in targets:
        samples = sorted((args.sample_root / tid).glob("Sample_*.pdb"))
        if not samples:
            out["targets"][tid] = {"n_samples": 0}
            print(f"{tid}: no samples", flush=True)
            continue
        try:
            pairs = parse_dotbracket(load_consensus_structure(tid, args.bpmaps_root / tid))
        except Exception as exc:  # no base-pair map for this directory
            print(f"{tid}: no base-pair map ({exc}); base-pair measures skipped", flush=True)
            pairs = []
        per = [check_sample(f, pairs) for f in samples]
        s = summarise(per)
        out["targets"][tid] = {**s, "per_sample": per}
        print(f"{tid}: {s['n_samples']} samples, {s['n_passing']} pass; linkages compressed/in-range/stretched/extended "
              f"{s['fraction_compressed']:.3f}/{s['fraction_in_range']:.3f}/{s['fraction_stretched']:.3f}/{s['fraction_extended']:.3f}; "
              f"median clashes {s['median_clashes']}; median wc {s['median_wc_n1n3']}, hbond {s['median_hbond_pairs']}", flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"Wrote {args.out}")


def selftest():
    A = {"N1": np.zeros(3), "N6": np.array([0.0, 2.9, 0.0]), "C1'": np.array([-1.0, 0, 0])}
    U = {"N3": np.array([2.9, 0, 0]), "O4": np.array([2.9, 2.9, 0]), "C1'": np.array([4.0, 0, 0])}
    assert pair_formed(("A", A), ("U", U)) is True
    assert pair_formed(("U", U), ("A", A)) is True
    U2 = {"N3": np.array([2.9, 0, 0]), "O4": np.array([6.0, 6.0, 0])}
    assert pair_formed(("A", A), ("U", U2)) is False
    assert pair_formed(("A", A), ("C", U)) is None
    G = {"N1": np.zeros(3), "O6": np.array([0.0, 2.8, 0])}
    Uw = {"O2": np.array([2.8, 0, 0]), "N3": np.array([2.8, 2.8, 0])}
    assert pair_formed(("G", G), ("U", Uw)) is True
    r = [("A", {"O3'": np.zeros(3), "C1'": np.array([0, 0, 0.0])}),
         ("A", {"P": np.array([1.6, 0, 0]), "O3'": np.array([10.0, 0, 0]), "C1'": np.array([5.5, 0, 0])}),
         ("A", {"P": np.array([12.5, 0, 0]), "O3'": np.array([13.0, 0, 0]), "C1'": np.array([11.0, 0, 0])}),
         ("A", {"P": np.array([13.5, 0, 0]), "C1'": np.array([20.0, 0, 0])})]
    lc = linkage_classes(r)
    assert lc == {"compressed": 1, "in_range": 1, "stretched": 1, "extended": 0}, lc
    assert c1c1_out(r) == 1, c1c1_out(r)
    assert base_letter("GUA") == "G" and base_letter("  U") == "U"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--out", type=Path, default=Path("structure_checks.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.sample_root and a.bpmaps_root):
        ap.error("--sample-root and --bpmaps-root are required")
    run(a)


if __name__ == "__main__":
    main()
