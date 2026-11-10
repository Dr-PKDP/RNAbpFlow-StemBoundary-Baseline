#!/usr/bin/env python3
"""Robustness analysis for the apo->holo displacement-direction test (pair targets only).

The primary test (displacement_direction_test.py) divides every coordinate by its
perturbation-baseline SD. That choice is arbitrary and it ignores a possible
systematic offset between the generated ensemble and both experimental states.
This script re-examines each pair target from angles that do not depend on it.

Per target it reports
  1. Scale sensitivity: variance enrichment along the experimental displacement
     under three coordinate scalings (baseline SD, sample SD, one pooled SD per
     coordinate type: distances in A, angles/torsions in degrees).
  2. Ensemble typicality (scale-free): Mahalanobis distance of the apo and holo
     coordinates from the ensemble, as a percentile of leave-one-out sample
     distances. A state the ensemble could plausibly produce has a percentile
     (p_typ) well above 0.05.
  3. Per-coordinate percentiles of apo and holo in the ensemble marginals, and
     the share (e_k^2) each coordinate contributes to the displacement direction.
  4. Leave-one-coordinate-out jackknife of enrichment and progress.
  5. Split-half (odd/even samples) replication.
  6. Bootstrap 95% intervals over samples for enrichment and progress.
  7. Offset diagnostics: progress of the ensemble centre along the displacement,
     its distance to apo and holo, and the fraction of its offset from apo that
     is orthogonal to the displacement.

Criteria (a target supports an
"ensemble extends along the apo->holo direction" statement only if all hold):
  c1 enrichment > 1 with p < 0.05 under all three scalings
  c2 lower bootstrap bound of enrichment (baseline scaling) > 1
  c3 enrichment > 1 after removing any single coordinate
  c4 enrichment > 1 in both split halves
  c5 holo typicality p_typ >= 0.05
  c6 ensemble-centre progress within [-0.5, 1.5] (no gross overshoot)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from displacement_direction_test import diff, random_unit, read_models, map_to_input, labeled_z  # noqa: E402


def circ_center(Z, is_tors):
    c = np.median(Z, axis=0).astype(float)
    for j in np.where(is_tors)[0]:
        r = np.deg2rad(Z[:, j])
        c[j] = np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    return c


def enrichment(dev, delta, scale, n_null, rng):
    D, d = dev / scale, delta / scale
    nrm = float(np.linalg.norm(d))
    if D.shape[1] < 2 or nrm < 1e-9:
        return None
    e = d / nrm
    C = np.cov(D.T)
    m = np.trace(C) / D.shape[1]
    lam = float(e @ C @ e) / m
    p = None
    if n_null:
        U = random_unit(n_null, D.shape[1], rng)
        null = np.einsum("ij,jk,ik->i", U, C, U) / m
        p = float((np.sum(null >= lam) + 1) / (n_null + 1))
    return lam, p, nrm


def mean_progress(from_apo, delta, scale):
    d = delta / scale
    nrm = np.linalg.norm(d)
    return float(((from_apo / scale) @ (d / nrm)).mean() / nrm)


def typicality(dev, ad, hd, scale):
    X, a, h = dev / scale, ad / scale, hd / scale
    K, p = X.shape

    def inv(M):
        S = np.cov(M.T)
        return np.linalg.inv(S + 1e-6 * np.trace(S) / p * np.eye(p))

    mu = X.mean(axis=0)
    Si = inv(X)
    d_apo = float(np.sqrt((a - mu) @ Si @ (a - mu)))
    d_holo = float(np.sqrt((h - mu) @ Si @ (h - mu)))
    loo = np.empty(K)
    for i in range(K):
        rest = np.delete(X, i, axis=0)
        m_i, S_i = rest.mean(axis=0), inv(rest)
        loo[i] = np.sqrt((X[i] - m_i) @ S_i @ (X[i] - m_i))
    pt = lambda d: float((np.sum(loo >= d) + 1) / (K + 1))
    return {"d_apo": d_apo, "d_holo": d_holo, "p_typ_apo": pt(d_apo), "p_typ_holo": pt(d_holo),
            "loo_median": float(np.median(loo)), "loo_p95": float(np.percentile(loo, 95))}


def robust_pair(Z, z_apo, z_holo, is_tors, scale, labels, n_null=20000, n_boot=2000, seed=0):
    keep = np.asarray(scale) > 1e-6
    if keep.sum() < 2:
        return {"testable": False, "reason": "fewer than 2 coordinates with usable baseline SD"}
    Z, z_apo, z_holo, is_tors = Z[:, keep], z_apo[keep], z_holo[keep], is_tors[keep]
    labels = [l for l, k in zip(labels, keep) if k]
    s_base = np.asarray(scale)[keep]
    K, p = Z.shape
    rng = np.random.default_rng(seed)

    center = circ_center(Z, is_tors)
    dev = diff(Z, center, is_tors)
    delta = diff(z_holo, z_apo, is_tors)
    ad, hd = diff(z_apo, center, is_tors), diff(z_holo, center, is_tors)
    from_apo = diff(Z, z_apo, is_tors)
    if np.linalg.norm(delta / s_base) < 1e-9:
        return {"testable": False, "reason": "zero experimental displacement"}

    s_samp = dev.std(axis=0, ddof=1)
    ang = np.array([not l.startswith("d[") for l in labels])
    s_block = np.empty(p)
    for mask in (ang, ~ang):
        if mask.any():
            s_block[mask] = np.sqrt(np.mean(s_samp[mask] ** 2))
    out = {"testable": True, "K": int(K), "p_coords": int(p), "coordinates_used": labels}

    # 1 scale sensitivity
    sens = {}
    for name, sc in (("baseline", s_base), ("sample_sd", s_samp), ("block_pooled", s_block)):
        r = enrichment(dev, delta, sc, n_null, rng) if sc.min() > 1e-9 else None
        sens[name] = None if r is None else {"enrichment": r[0], "log2": float(np.log2(r[0])), "p": r[1],
                                              "mean_progress": mean_progress(from_apo, delta, sc)}
    out["scale_sensitivity"] = sens

    # 2 typicality
    out["typicality"] = typicality(dev, ad, hd, s_base)

    # 3 per-coordinate table
    e = (delta / s_base) / np.linalg.norm(delta / s_base)
    tab = []
    for j, l in enumerate(labels):
        col = dev[:, j]
        pa, ph = float(np.mean(col <= ad[j])), float(np.mean(col <= hd[j]))
        tab.append({"coordinate": l, "apo_percentile": pa, "holo_percentile": ph,
                    "apo_in_central90": bool(0.05 <= pa <= 0.95), "holo_in_central90": bool(0.05 <= ph <= 0.95),
                    "displacement_baseline_sd": float(delta[j] / s_base[j]), "direction_share": float(e[j] ** 2)})
    out["per_coordinate"] = tab

    # 4 jackknife over coordinates
    jk = {}
    if p >= 3:
        for j, l in enumerate(labels):
            idx = np.arange(p) != j
            r = enrichment(dev[:, idx], delta[idx], s_base[idx], 0, rng)
            if r is not None:
                jk[l] = {"enrichment": r[0], "mean_progress": mean_progress(from_apo[:, idx], delta[idx], s_base[idx])}
    out["jackknife"] = jk

    # 5 split halves
    halves = {}
    for name, idx in (("even", np.arange(0, K, 2)), ("odd", np.arange(1, K, 2))):
        r = enrichment(dev[idx], delta, s_base, 0, rng)
        halves[name] = {"enrichment": r[0], "mean_progress": mean_progress(from_apo[idx], delta, s_base)}
    out["split_half"] = halves

    # 6 bootstrap over samples
    lam_b, prog_b = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(K, size=K)
        lam_b[b] = enrichment(dev[idx], delta, s_base, 0, rng)[0]
        prog_b[b] = mean_progress(from_apo[idx], delta, s_base)
    out["bootstrap"] = {"enrichment_ci95": [float(np.percentile(lam_b, 2.5)), float(np.percentile(lam_b, 97.5))],
                        "progress_ci95": [float(np.percentile(prog_b, 2.5)), float(np.percentile(prog_b, 97.5))]}

    # 7 offset diagnostics
    cA = -ad / s_base
    proj = float(cA @ e)
    out["offset"] = {"center_progress": proj / float(np.linalg.norm(delta / s_base)),
                     "center_to_apo_sd": float(np.linalg.norm(cA)),
                     "center_to_holo_sd": float(np.linalg.norm(hd / s_base)),
                     "orthogonal_fraction": float(np.linalg.norm(cA - proj * e) / max(np.linalg.norm(cA), 1e-12))}

    # criteria
    s_ok = all(v is not None and v["enrichment"] > 1 and v["p"] < 0.05 for v in sens.values())
    crit = {
        "c1_all_scalings": bool(s_ok),
        "c2_bootstrap_lower_gt1": bool(out["bootstrap"]["enrichment_ci95"][0] > 1),
        "c3_jackknife_gt1": bool(len(jk) > 0 and all(v["enrichment"] > 1 for v in jk.values())),
        "c4_split_halves_gt1": bool(all(v["enrichment"] > 1 for v in halves.values())),
        "c5_holo_typical": bool(out["typicality"]["p_typ_holo"] >= 0.05),
        "c6_no_gross_overshoot": bool(-0.5 <= out["offset"]["center_progress"] <= 1.5),
    }
    out["criteria"] = crit
    out["criteria_met"] = int(sum(crit.values()))
    return out


def selftest():
    rng = np.random.default_rng(1)
    K, p = 100, 6
    tors = np.zeros(p, bool)
    sc = np.ones(p)
    labs = ["d[x%d]" % i for i in range(p)]
    apo = np.zeros(p)
    holo = np.array([3., 3, 0, 0, 0, 0])

    # A: genuine case. Latent factor along (1,1) with centre half-way, holo and apo inside the ensemble.
    f = rng.normal(0, 2.2, K)
    A = rng.normal(0, 0.6, (K, p)) + np.outer(f + 0.5, [1, 1, 0, 0, 0, 0]) * 1.5
    ra = robust_pair(A, apo, holo, tors, sc, labs, n_null=5000, n_boot=300)
    print("A genuine   : met %d  enr %s  p_typ_holo %.3f  centre_prog %.2f" % (
        ra["criteria_met"], {k: round(v["enrichment"], 1) for k, v in ra["scale_sensitivity"].items()},
        ra["typicality"]["p_typ_holo"], ra["offset"]["center_progress"]))
    assert ra["criteria_met"] == 6, ra["criteria"]

    # B: variance orthogonal to the displacement, ensemble sitting at apo
    B = rng.normal(0, 1, (K, p)); B[:, 4] = rng.normal(0, 3, K)
    rb = robust_pair(B, apo, holo, tors, sc, labs, n_null=5000, n_boot=300)
    print("B orthogonal: met %d  enr baseline %.2f  p_typ_holo %.3f" % (
        rb["criteria_met"], rb["scale_sensitivity"]["baseline"]["enrichment"], rb["typicality"]["p_typ_holo"]))
    assert rb["criteria_met"] <= 2 and not rb["criteria"]["c1_all_scalings"] and not rb["criteria"]["c5_holo_typical"]

    # C: offset artefact. Variance along the displacement, but centre 5 displacement lengths past apo and off-axis.
    C = rng.normal(0, 0.5, (K, p)) + np.outer(rng.normal(0, 3, K), [1, 1, 0, 0, 0, 0]) / np.sqrt(2) \
        + np.array([15, 15, 6, 6, 0, 0])
    rc = robust_pair(C, apo, holo, tors, sc, labs, n_null=5000, n_boot=300)
    print("C offset    : met %d  centre_prog %.2f  orth %.2f  p_typ_holo %.3f" % (
        rc["criteria_met"], rc["offset"]["center_progress"], rc["offset"]["orthogonal_fraction"], rc["typicality"]["p_typ_holo"]))
    assert not rc["criteria"]["c6_no_gross_overshoot"] and not rc["criteria"]["c5_holo_typical"]

    # D: single-coordinate dependence. Huge variance on coordinate 0 only; holo has a small component elsewhere.
    D = rng.normal(0, 1, (K, p)); D[:, 0] = rng.normal(0, 8, K)
    holoD = np.array([6., 0, 0, 2, 0, 0])
    rd = robust_pair(D, apo, holoD, tors, sc, labs, n_null=5000, n_boot=300)
    jkmin = min(v["enrichment"] for v in rd["jackknife"].values())
    print("D single    : full enr %.2f  jackknife min %.2f  c3 %s" % (
        rd["scale_sensitivity"]["baseline"]["enrichment"], jkmin, rd["criteria"]["c3_jackknife_gt1"]))
    assert rd["scale_sensitivity"]["baseline"]["enrichment"] > 1.8 and jkmin < 1.3 and not rd["criteria"]["c3_jackknife_gt1"]

    # torsion circular centre
    cc = circ_center(np.array([[179.], [-179.], [178.], [-178.]]), np.array([True]))
    assert abs(abs(cc[0]) - 180) < 2, cc

    # unusable inputs
    assert robust_pair(A[:, :1], apo[:1], holo[:1], tors[:1], sc[:1], labs[:1])["testable"] is False
    assert robust_pair(A, apo, apo, tors, sc, labs)["testable"] is False
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--out", type=Path, default=Path("displacement_robustness.json"))
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

    out = {}
    for t in json.loads(args.config.read_text()):
        if t["mode"] != "pair":
            continue
        tid = t["id"]
        print(f"{tid} ...")
        db = load_consensus_structure(tid, args.bpmaps_root / tid)
        fa = (args.bpmaps_root / tid / f"{tid}.fasta").read_text().split("\n")
        input_seq = "".join(l.strip() for l in fa if l and not l.startswith(">"))
        samples = sorted((args.sample_root / tid).glob("Sample_*.pdb"))
        zs = [labeled_z(read_c1_coords(f), db, extract_z) for f in samples]
        apo = labeled_z(map_to_input(read_models(t["apo_pdb"], t.get("apo_chain"))[0], input_seq), db, extract_z)
        holo = labeled_z(map_to_input(read_models(t["holo_pdb"], t.get("holo_chain"))[0], input_seq), db, extract_z)
        common = [l for l in zs[0] if all(l in z for z in zs) and l in apo and l in holo]
        if len(common) < 2:
            out[tid] = {"testable": False, "reason": f"{len(common)} shared coordinates"}
            continue
        fv, fl, n_ok = noise_floor(read_c1_coords(samples[0]), db, n_perturbations=args.n_perturbations)
        if fv is None:
            out[tid] = {"testable": False, "reason": f"perturbation baseline not estimable ({n_ok}/{args.n_perturbations} valid)"}
            continue
        fmap = dict(zip(fl, np.sqrt(fv)))
        scale = np.array([fmap.get(l, 0.0) for l in common])
        is_tors = np.array([l.startswith("torsion") for l in common])
        Z = np.array([[z[l] for l in common] for z in zs])
        res = robust_pair(Z, np.array([apo[l] for l in common]), np.array([holo[l] for l in common]),
                          is_tors, scale, common)
        out[tid] = res
        if res.get("testable"):
            s = res["scale_sensitivity"]
            print("  enrichment:", {k: (None if v is None else round(v["enrichment"], 2)) for k, v in s.items()})
            print("  typicality:", {k: round(v, 3) for k, v in res["typicality"].items()})
            print("  offset    :", {k: round(v, 2) for k, v in res["offset"].items()})
            print("  criteria  :", res["criteria"], "->", res["criteria_met"], "of 6")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
