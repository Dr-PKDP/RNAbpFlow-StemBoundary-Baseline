"""
displacement_direction_test.py -- direct test of whether RNAbpFlow samples
extend along the experimentally observed conformational displacement, in the
same collective-coordinate space (z = distances + angles) used by the
perturbation-baseline analysis.

Two target modes (set per target in the config JSON):

  "pair"  apo and holo (or free / bound) structures of the same construct.
          Delta_exp = z_holo - z_apo. Two statistics, each with a random-
          direction null:
            (1) variance enrichment  lam = e' C e, where e is the unit
                standardized Delta_exp and C the sample correlation matrix.
                lam > 1: more sample variance along the experimental
                direction than along a random direction (null mean = 1).
            (2) mean progress  mean_k[(z_k - z_apo)/s . e] / |Delta_exp/s|,
                the average fraction of the apo->holo displacement covered
                by the samples (0 = at apo, 1 = at holo).
  "nmr"   multi-model NMR ensemble. Direction = leading eigenvector of the
          standardized NMR-ensemble covariance; statistic = |cos| with the
          samples' leading eigenvector against a random-direction null.
          No apo/holo contrast is available for this mode.

Coordinates are divided by their perturbation-baseline standard deviation (the
same baseline used for the clearing analysis), which removes the Angstrom/degree
unit mismatch without discarding the between-coordinate differences in sample
variance that the enrichment statistic measures. (Standardizing by the sample SD
instead would force every axis-aligned direction to enrichment 1.) Coordinates
with baseline SD below 1e-6 are dropped. Torsion differences are wrapped to
[-180, 180]. Targets with fewer than 2 usable coordinates, or without an
estimable baseline, are reported as not testable.

Usage:
    python3 displacement_direction_test.py --config targets_displacement.json \
        --sample-root ../data/Predictions --bpmaps-root ../data/Inputs \
        --pipeline-dir ../pipeline --out displacement_report.json

Config entry, e.g.:
  {"id": "5e54", "mode": "pair",
   "apo_pdb": "refs/5E54.pdb", "apo_chain": "A",
   "holo_pdb": "refs/4TZX.pdb", "holo_chain": "A"}
  {"id": "5kh8", "mode": "nmr", "pdb": "refs/5KH8.pdb", "chain": "A"}

Run `python3 displacement_direction_test.py --selftest` first (synthetic
ground truth, no data needed).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AA3 = {"A": "A", "C": "C", "G": "G", "U": "U", "RA": "A", "RC": "C", "RG": "G", "RU": "U",
       "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U", "T": "U", "DA": "A", "DC": "C", "DG": "G"}


def wrap_deg(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def random_unit(n_draws, p, rng):
    v = rng.normal(size=(n_draws, p))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def diff(a, b, is_tors):
    d = np.asarray(a, float) - np.asarray(b, float)
    d[..., is_tors] = wrap_deg(d[..., is_tors])
    return d


def pair_statistics(Z, z_apo, z_holo, is_tors, scale, n_null=20000, seed=0):
    """Z: (K, p) sample coordinates; z_apo, z_holo: (p,); scale: (p,) baseline SD."""
    keep = np.asarray(scale) > 1e-6
    if keep.sum() < 2:
        return {"testable": False, "reason": "fewer than 2 coordinates with usable baseline SD"}
    Z, z_apo, z_holo, is_tors, s = Z[:, keep], z_apo[keep], z_holo[keep], is_tors[keep], np.asarray(scale)[keep]
    p = Z.shape[1]
    delta = diff(z_holo, z_apo, is_tors) / s
    nrm = np.linalg.norm(delta)
    if nrm < 1e-9:
        return {"testable": False, "reason": "zero experimental displacement"}
    e = delta / nrm
    dev = diff(Z, np.median(Z, axis=0), is_tors) / s
    C = np.cov(dev.T)
    mean_eig = np.trace(C) / p
    lam = float(e @ C @ e) / mean_eig
    rng = np.random.default_rng(seed)
    U = random_unit(n_null, p, rng)
    lam_null = np.einsum("ij,jk,ik->i", U, C, U) / mean_eig
    p_lam = float((np.sum(lam_null >= lam) + 1) / (n_null + 1))
    from_apo = diff(Z, z_apo, is_tors) / s
    prog_k = from_apo @ e / nrm
    prog = float(prog_k.mean())
    prog_null = (U @ from_apo.mean(axis=0)) / nrm
    p_prog = float((np.sum(prog_null >= prog) + 1) / (n_null + 1))
    return {"testable": True, "p_coords": int(p), "variance_enrichment": lam,
            "log2_enrichment": float(np.log2(lam)), "p_enrichment": p_lam,
            "mean_progress": prog, "frac_samples_toward_holo": float((prog_k > 0).mean()),
            "frac_samples_past_halfway": float((prog_k >= 0.5).mean()), "p_progress": p_prog,
            "K": int(len(Z)), "exp_displacement_norm_baseline_sd_units": float(nrm)}


def nmr_statistics(Z, Z_nmr, is_tors, scale, n_null=20000, seed=0):
    keep = np.asarray(scale) > 1e-6
    if keep.sum() < 2 or len(Z_nmr) < 3:
        return {"testable": False, "reason": "fewer than 2 usable coordinates or fewer than 3 NMR models"}
    Z, Z_nmr, is_tors, s = Z[:, keep], Z_nmr[:, keep], is_tors[keep], np.asarray(scale)[keep]
    p = Z.shape[1]
    a = diff(Z, np.median(Z, axis=0), is_tors) / s
    b = diff(Z_nmr, np.median(Z_nmr, axis=0), is_tors) / s
    va = np.linalg.eigh(np.cov(a.T))[1][:, -1]
    vb = np.linalg.eigh(np.cov(b.T))[1][:, -1]
    cos = float(abs(va @ vb))
    rng = np.random.default_rng(seed)
    null = np.abs(random_unit(n_null, p, rng) @ vb)
    return {"testable": True, "p_coords": int(p), "abs_cosine_leading_axes": cos,
            "p_alignment": float((np.sum(null >= cos) + 1) / (n_null + 1)),
            "K": int(len(Z)), "n_nmr_models": int(len(Z_nmr))}


def cohort_summary(per_target, n_boot=10000, seed=1):
    vals = [(t, r["log2_enrichment"], r["p_enrichment"]) for t, r in per_target.items()
            if r.get("testable") and "log2_enrichment" in r]
    if not vals:
        return {"n_targets": 0}
    x = np.array([v[1] for v in vals])
    rng = np.random.default_rng(seed)
    boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    npos = int((x > 0).sum())
    from math import comb
    n = len(x)
    sign_p = sum(comb(n, k) for k in range(npos, n + 1)) / 2 ** n
    return {"n_targets": n, "n_log2_enrichment_positive": npos, "mean_log2_enrichment": float(x.mean()),
            "bootstrap_95ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "one_sided_sign_test_p": float(sign_p),
            "note": "target-level inference; coordinates within a target are not independent"}


# ---- structure I/O and residue mapping (data-dependent; used by main only) ----
def read_models(pdb_path, chain=None):
    """Returns list of {res_index: (resname, C1' xyz)} per MODEL, residues numbered in order."""
    models, cur, seen = [], None, set()
    avail = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                cur, seen = [], set()
            elif line.startswith("ENDMDL"):
                models.append(cur); cur = None
            elif (line.startswith("ATOM") or line.startswith("HETATM")) and line[12:16].strip() == "C1'":
                avail[line[21]] = avail.get(line[21], 0) + 1
                if chain and line[21] != chain:
                    continue
                key = (line[21], line[22:26].strip(), line[26])
                if cur is None:
                    cur, seen = [], set()
                    models_single = True
                if key in seen:
                    continue
                seen.add(key)
                cur.append((AA3.get(line[17:20].strip().upper(), "N"),
                            np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    if cur:
        models.append(cur)
    if not models:
        raise ValueError(f"{pdb_path}: no C1' atoms for chain {chain!r}; C1' atom counts by chain: {avail or 'none (no nucleic-acid residues)'}")
    return models


def map_to_input(model, input_seq):
    """Global alignment of PDB residue sequence to the model input sequence; returns {input_idx: xyz}."""
    from Bio import Align
    al = Align.PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score = 2, -1
    al.open_gap_score, al.extend_gap_score = -2, -1
    al.end_gap_score = 0
    pdb_seq = "".join(r[0] for r in model)
    best = al.align(input_seq, pdb_seq)[0]
    mapping = {}
    for (a0, a1), (b0, b1) in zip(*best.aligned):
        for off in range(a1 - a0):
            if input_seq[a0 + off] == pdb_seq[b0 + off]:
                mapping[a0 + off] = model[b0 + off][1]
    return mapping


def labeled_z(coords, db, extract_z):
    z, labels = extract_z(coords, db)
    return dict(zip(labels, z))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--out", type=Path, default=Path("displacement_report.json"))
    ap.add_argument("--n-perturbations", type=int, default=300, help="perturbation attempts per target")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.config and args.sample_root and args.bpmaps_root):
        ap.error("--config, --sample-root and --bpmaps-root are required")
    sys.path.insert(0, str(args.pipeline_dir))
    from collective_coords import extract_z
    from clearing import read_c1_coords, load_consensus_structure, noise_floor

    cfg = json.loads(args.config.read_text())
    out = {}
    for t in cfg:
        tid = t["id"]
        print(f"{tid} ({t['mode']}) ...")
        db = load_consensus_structure(tid, args.bpmaps_root / tid)
        fa = (args.bpmaps_root / tid / f"{tid}.fasta").read_text().split("\n")
        input_seq = "".join(l.strip() for l in fa if l and not l.startswith(">"))
        samples = sorted((args.sample_root / tid).glob("Sample_*.pdb"))
        zs = [labeled_z(read_c1_coords(f), db, extract_z) for f in samples]
        common = [l for l in zs[0] if all(l in z for z in zs)]
        if t["mode"] == "pair":
            apo = labeled_z(map_to_input(read_models(t["apo_pdb"], t.get("apo_chain"))[0], input_seq), db, extract_z)
            holo = labeled_z(map_to_input(read_models(t["holo_pdb"], t.get("holo_chain"))[0], input_seq), db, extract_z)
            common = [l for l in common if l in apo and l in holo]
        else:
            nm = [labeled_z(map_to_input(m, input_seq), db, extract_z)
                  for m in read_models(t["pdb"], t.get("chain"))]
            common = [l for l in common if all(l in z for z in nm)]
        if len(common) < 2:
            out[tid] = {"testable": False, "reason": f"{len(common)} shared coordinates"}
            continue
        floor_var, floor_labels, n_ok = noise_floor(read_c1_coords(samples[0]), db, n_perturbations=args.n_perturbations)
        if floor_var is None:
            out[tid] = {"testable": False, "reason": f"perturbation baseline not estimable ({n_ok}/{args.n_perturbations} valid)"}
            continue
        fl = dict(zip(floor_labels, np.sqrt(floor_var)))
        scale = np.array([fl.get(l, 0.0) for l in common])
        is_tors = np.array([l.startswith("torsion") for l in common])
        Z = np.array([[z[l] for l in common] for z in zs])
        if t["mode"] == "pair":
            res = pair_statistics(Z, np.array([apo[l] for l in common]),
                                  np.array([holo[l] for l in common]), is_tors, scale)
        else:
            res = nmr_statistics(Z, np.array([[z[l] for l in common] for z in nm]), is_tors, scale)
        res["coordinates_used"] = common
        out[tid] = res
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in res.items() if k != "coordinates_used"})
    pairs = {k: v for k, v in out.items() if "variance_enrichment" in v}
    out["_cohort_pair_mode"] = cohort_summary(pairs)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


def selftest():
    rng = np.random.default_rng(0)
    K, p = 60, 6
    tors = np.zeros(p, bool)
    sc = np.ones(p)
    apo, holo = np.zeros(p), np.array([3., 0, 0, 0, 0, 0])
    A = rng.normal(size=(K, p)); A[:, 0] = rng.normal(1.5, 3.0, K)   # variance along experimental axis, toward holo
    B = rng.normal(size=(K, p)); B[:, 3] = rng.normal(0, 3.0, K)     # variance along an orthogonal axis, at apo
    ra, rb = pair_statistics(A, apo, holo, tors, sc), pair_statistics(B, apo, holo, tors, sc)
    print("A (aligned):    enrichment %.2f p=%.4f progress %.2f" % (ra["variance_enrichment"], ra["p_enrichment"], ra["mean_progress"]))
    print("B (orthogonal): enrichment %.2f p=%.4f progress %.2f" % (rb["variance_enrichment"], rb["p_enrichment"], rb["mean_progress"]))
    assert ra["variance_enrichment"] > 1.8 and ra["p_enrichment"] < 0.05 and ra["mean_progress"] > 0.3
    assert rb["variance_enrichment"] < 0.5 and rb["p_enrichment"] > 0.5
    d = diff(np.array([179.]), np.array([-179.]), np.array([True]))
    assert abs(d[0] + 2.0) < 1e-9
    ndir = np.array([1., 1, 0, 0, 0, 0]); ndir /= np.linalg.norm(ndir)
    N = rng.normal(size=(20, p)) * 0.3 + np.outer(rng.normal(size=20) * 3, ndir)
    S = rng.normal(size=(K, p)) * 0.3 + np.outer(rng.normal(size=K) * 3, ndir)
    rn = nmr_statistics(S, N, tors, sc)
    print("NMR aligned: |cos| %.2f p=%.4f" % (rn["abs_cosine_leading_axes"], rn["p_alignment"]))
    assert rn["abs_cosine_leading_axes"] > 0.9 and rn["p_alignment"] < 0.1
    assert pair_statistics(A[:, :1], apo[:1], holo[:1], tors[:1], sc[:1])["testable"] is False
    print("selftest passed")


if __name__ == "__main__":
    main()
