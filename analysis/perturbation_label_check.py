#!/usr/bin/env python3
"""Do the accepted perturbations preserve coordinate correspondence?

Question. clearing.py accepts a perturbed base-pair map when its coordinate vector has the same length as the
unperturbed one. Equal length does not guarantee that position j of the perturbed vector describes the same stem pair as
position j of the unperturbed vector, because stems are re-parsed and renumbered after the boundary shifts. This script
replays the accepted perturbations of clearing.py (same seed, same sequence of random draws) and reports, per target:
  attempted / accepted by length          as in clearing.py
  accepted with identical labels          the perturbed label list equals the unperturbed one
  baseline from label-matched vectors     recomputed variance, available when at least 10 vectors match
  clearing proportions                    distance and angle, from the length-matched baseline (as published) and
                                          from the label-matched baseline, with the number of coordinates whose
                                          clearing call differs
  torsion labels                          number of torsion coordinates in the unperturbed vector

Clearing follows clearing.py: ensemble variance (1/n, all samples) over baseline variance (1/n) greater than 1,
with the first generated sample as the coordinate source of the baseline.

Usage:
    python perturbation_label_check.py --targets t07 7kd1 --sample-root ../data/Predictions \\
        --bpmaps-root ../data/Inputs --out perturbation_label_check.json
    python perturbation_label_check.py --selftest
(--pipeline-dir selects another copy of collective_coords.py and clearing.py, for example the one that produced the tables.)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_pd = Path(__file__).resolve().parent.parent / "pipeline"
for i, a in enumerate(sys.argv):
    if a == "--pipeline-dir" and i + 1 < len(sys.argv):
        _pd = Path(sys.argv[i + 1])
sys.path.insert(0, str(_pd))
from collective_coords import extract_z  # noqa: E402
from clearing import perturb_dotbracket, read_c1_coords, load_consensus_structure, stem_correspondence  # noqa: E402

EPS = 1e-12


def replay(coords, db, n_perturbations=300, seed=0):
    rng = np.random.default_rng(seed)
    base_z, base_labels = extract_z(coords, db)
    by_length, matched, ident = [], [], []
    for _ in range(n_perturbations):
        pdb2 = perturb_dotbracket(db, rng)
        try:
            z2, labels2 = extract_z(coords, pdb2)
        except Exception:
            continue
        if len(z2) == len(base_z):
            by_length.append(z2)
            if list(labels2) == list(base_labels):
                matched.append(z2)
                if stem_correspondence(db, pdb2):
                    ident.append(z2)
    return base_labels, np.array(by_length), np.array(matched), np.array(ident)


def clearing(var_gen, var_base):
    return var_gen / (var_base + EPS) > 1.0


def check_target(name, db, sample_files, n_perturbations=300):
    coords = [read_c1_coords(f) for f in sample_files]
    z = []
    labels = None
    for c in coords:
        zi, labels = extract_z(c, db)
        z.append(zi)
    z = np.array(z)
    var_gen = z.var(axis=0)
    base_labels, zl, zm, zi = replay(coords[0], db, n_perturbations)
    is_dist = np.array([l.startswith("d[") for l in base_labels])
    is_ang = np.array([l.startswith("theta[") for l in base_labels])
    res = {"attempted": n_perturbations, "accepted_by_length": int(len(zl)), "accepted_with_identical_labels": int(len(zm)), "accepted_with_stem_identity": int(len(zi)),
           "n_distance": int(is_dist.sum()), "n_angle": int(is_ang.sum()),
           "n_torsion": int(sum(l.startswith("torsion") for l in base_labels))}
    out = {}
    for tag, arr in (("length_matched", zl), ("label_matched", zm), ("identity_matched", zi)):
        if len(arr) < 10:
            out[tag] = None
            continue
        c = clearing(var_gen, arr.var(axis=0))
        out[tag] = {"clear": c.tolist(),
                    "distance_proportion": float(c[is_dist].mean()) if is_dist.any() else None,
                    "angle_proportion": float(c[is_ang].mean()) if is_ang.any() else None}
    res.update(out)
    for tag, key in (("label_matched", "coordinates_with_changed_call"), ("identity_matched", "coordinates_with_changed_call_identity")):
        if out["length_matched"] and out[tag]:
            res[key] = int((np.array(out["length_matched"]["clear"]) != np.array(out[tag]["clear"])).sum())
    res["coordinate_labels"] = list(base_labels)
    return res


def selftest():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from validate_collective_coords import a_form_helix, rotation_about_y
    # two sibling helices of 10 base pairs: the label list is [d[sibling:stem0-stem1], theta[sibling:stem0-stem1]]
    c1, _ = a_form_helix(10)
    c2r, _ = a_form_helix(10, origin=np.array([30.0, 0.0, 0.0]), rot=rotation_about_y(50.0))
    coords = {**c1, **{k + 20: v for k, v in c2r.items()}}
    db = "(" * 10 + ")" * 10 + "(" * 10 + ")" * 10
    labels, zl, zm, zi = replay(coords, db, 100)
    assert len(zl) >= 10 and len(zi) <= len(zm) <= len(zl), (len(zl), len(zm), len(zi))
    assert zl.shape[1] == len(labels) == 2
    print("selftest passed:", len(zl), "accepted by length,", len(zm), "with identical labels,", len(zi), "with stem identity")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=None,
                    help="directory holding collective_coords.py and clearing.py to use (default ../pipeline)")
    ap.add_argument("--n-perturbations", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("perturbation_label_check.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.targets and a.sample_root and a.bpmaps_root):
        ap.error("--targets, --sample-root and --bpmaps-root are required")
    out = {}
    for t in a.targets:
        db = load_consensus_structure(t, a.bpmaps_root / t)
        files = sorted((a.sample_root / t).glob("Sample_*.pdb"))
        r = check_target(t, db, files, a.n_perturbations)
        out[t] = r
        lm, bm = r.get("length_matched"), r.get("label_matched")
        print(f"{t}: accepted by length {r['accepted_by_length']}, identical labels {r['accepted_with_identical_labels']}, stem identity {r['accepted_with_stem_identity']}, "
              f"torsions {r['n_torsion']}, changed calls {r.get('coordinates_with_changed_call', 'n/a')}; "
              f"distance/angle clearing length-matched "
              f"{(round(lm['distance_proportion'], 3), round(lm['angle_proportion'], 3)) if lm and lm['distance_proportion'] is not None and lm['angle_proportion'] is not None else 'n/a'} "
              f"label-matched "
              f"{(round(bm['distance_proportion'], 3), round(bm['angle_proportion'], 3)) if bm and bm['distance_proportion'] is not None and bm['angle_proportion'] is not None else 'n/a'}")
    a.out.write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
