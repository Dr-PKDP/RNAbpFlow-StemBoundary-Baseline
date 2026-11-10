"""Clearing, target-level statistics, displacement-direction criteria and experimental concordance in one pass,
for one ensemble (original, relaxed, or a subset of samples) and one or more baseline substrates.

For every substrate (the generated sample whose C1' coordinates carry the perturbation baseline, §2.4) it computes

  clearing         per-coordinate calls, distance and angle clearing proportions and their difference per target;
                   the cohort mean, median, 95% bootstrap interval (10,000 resamples of targets, seed 0), exact sign
                   test, and the pooled McNemar test with Wilson intervals (compute_statistics.py)
  displacement     the six criteria of displacement_robustness.py for every apo/holo pair target, on all shared
                   coordinates and on the coordinates whose two stems are conserved in both experimental states
                   (at least half of the assessable predicted pairs of each stem formed in apo and in holo, with a
                   pair counted as formed when two canonical or wobble donor-acceptor distances are within 3.5 A)
  concordance      per-coordinate generated, baseline and experimental variances (experimental_variation_concordance.py)
                   and the partial Spearman statistic for all targets, NMR bundles, apo/holo pairs, and with the
                   mixed ground/excited-state FARFAR-NMR entry 8THV reported as its own reference type

Substrates: the primary one (the first sample in file order, as in all primary analyses) and, with --random-substrates N,
N further samples drawn without replacement with --seed; the same file names are used for every target. With
--perturbation-seeds N the whole analysis is repeated with the primary substrate and independently generated perturbation
sets (perturbation seeds 1..N; the primary analysis uses seed 0). With --sweep-substrates, clearing is also recomputed with
every sample of each target as the substrate, and the target-level mean difference is summarised over 10,000 random
combinations of one substrate per target.
--subset-checks <structure_checks.json> restricts each target to the samples that passed the fixed criteria
(targets with fewer than --min-subset are left out); the substrate is then the first passing sample.

Usage:
  python3 ensemble_analyses.py --sample-root <dir> --bpmaps-root <Inputs> --pipeline-dir <pipeline> \
      --pairs-config pairs.json --concordance-config concordance.json --targets ... [--random-substrates 20] --out out.json
  python3 ensemble_analyses.py --selftest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from displacement_direction_test import read_models, map_to_input, labeled_z, AA3  # noqa: E402
from displacement_robustness import robust_pair  # noqa: E402
from experimental_variation_concordance import coordinate_rows, cohort_statistics, _mapped  # noqa: E402
from compute_statistics import bootstrap_ci, pooled_mcnemar_and_wilson  # noqa: E402
from structure_checks import pair_formed  # noqa: E402

MIXED_STATE_ENTRIES = ("8THV",)


# ---------------------------------------------------------------- experimental structures, all heavy atoms
def read_full_model(pdb_path, chain=None):
    """First model, one chain: list of (base letter, {atom: xyz}) for residues with a C1' atom."""
    res, order = {}, []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ENDMDL") and order:
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if chain and line[21] != chain:
                continue
            key = (line[21], line[22:26].strip(), line[26])
            if key not in res:
                res[key] = (AA3.get(line[17:20].strip().upper(), "N"), {})
                order.append(key)
            name = line[12:16].strip().replace("*", "'")
            res[key][1].setdefault(name, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    return [res[k] for k in order if "C1'" in res[k][1]]


def map_full_to_input(model, input_seq):
    idx = map_to_input([(r[0], i) for i, r in enumerate(model)], input_seq)
    return {i: model[j] for i, j in idx.items()}


def stem_conservation(db, states, parse_dotbracket, find_stems, min_fraction=0.5):
    """states: list of {input_idx: (letter, atoms)}. Returns per-stem formed fractions and conserved stem ids."""
    stems = find_stems(parse_dotbracket(db))
    table, conserved = [], set()
    for s in stems:
        fr = []
        for st in states:
            calls = [pair_formed(st[i], st[j]) for i, j in s.pairs if i in st and j in st]
            calls = [c for c in calls if c is not None]
            fr.append(float(np.mean(calls)) if calls else None)
        ok = all(f is not None and f >= min_fraction for f in fr)
        table.append({"stem": s.id, "pairs": [list(p) for p in s.pairs], "formed_fraction": fr, "conserved": ok})
        if ok:
            conserved.add(s.id)
    return table, conserved


def label_stems(label):
    inner = label[label.index(":") + 1:label.index("]")]
    return {int(x.replace("stem", "")) for x in inner.split("-")}


# ---------------------------------------------------------------- statistics
def clearing_for_target(v_gen, labels, fv, fl):
    vb = dict(zip(fl, fv))
    calls, ratio = {}, {}
    for l, v in zip(labels, v_gen):
        b = vb.get(l)
        if b is None or b <= 1e-6:
            continue
        ratio[l] = float(v / b)
        calls[l] = bool(v > b)
    d = [calls[l] for l in calls if l.startswith("d[")]
    a = [calls[l] for l in calls if l.startswith("theta[")]
    dp = float(np.mean(d)) if d else None
    ap = float(np.mean(a)) if a else None
    pairs = [(l, "theta" + l[1:]) for l in calls if l.startswith("d[") and ("theta" + l[1:]) in calls]
    return {"calls": calls, "ratios": ratio, "distance": dp, "angle": ap,
            "delta": (ap - dp) if (dp is not None and ap is not None) else None,
            "matched_pairs": [(ratio[x], ratio[y]) for x, y in pairs]}


def cohort_clearing(per_target):
    from scipy.stats import binomtest
    deltas = {t: v["delta"] for t, v in per_target.items() if v.get("delta") is not None}
    if not deltas:
        return {"testable": False}
    vals = np.array(list(deltas.values()))
    nz = vals[vals != 0]
    rows = [(t, d, a) for t, v in per_target.items() if v.get("delta") is not None for d, a in v["matched_pairs"]]
    out = {"n_targets": len(vals), "mean_delta": float(vals.mean()), "median_delta": float(np.median(vals)),
           "bootstrap_ci95": list(bootstrap_ci(vals)), "n_nonzero": int(len(nz)), "n_positive": int((nz > 0).sum()),
           "sign_test_p": float(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else None,
           "deltas": deltas}
    if rows:
        m = pooled_mcnemar_and_wilson(rows)
        out["pooled"] = {"n": m["n"], "distance_count": m["distance_count"], "angle_count": m["angle_count"],
                         "mcnemar_p": float(m["mcnemar_p"])}
    return out


def displacement_summary(res):
    if not res.get("testable"):
        return res
    s = res["scale_sensitivity"]
    return {"testable": True, "coordinates_used": res["coordinates_used"], "criteria": res["criteria"],
            "criteria_met": res["criteria_met"],
            "enrichment": {k: (None if v is None else v["enrichment"]) for k, v in s.items()},
            "enrichment_p": {k: (None if v is None else v["p"]) for k, v in s.items()},
            "mean_progress_baseline": None if s["baseline"] is None else s["baseline"]["mean_progress"],
            "centre_progress": res["offset"]["center_progress"], "typicality": res["typicality"],
            "bootstrap": res["bootstrap"],
            "top_direction_share": max(res["per_coordinate"], key=lambda r: r["direction_share"])}


# ---------------------------------------------------------------- main loop
def run(args):
    sys.path.insert(0, str(args.pipeline_dir))
    from collective_coords import extract_z, parse_dotbracket, find_stems
    from clearing import read_c1_coords, load_consensus_structure, noise_floor

    pair_cfg = {t["id"]: t for t in json.loads(args.pairs_config.read_text()) if t["mode"] == "pair"} if args.pairs_config else {}
    conc_cfg = {t["id"]: t for t in json.loads(args.concordance_config.read_text())} if args.concordance_config else {}
    subset = None
    if args.subset_checks:
        chk = json.loads(args.subset_checks.read_text())["targets"]
        subset = {t: set(v.get("passing", [])) for t, v in chk.items()}

    data, skipped = {}, {}
    for tid in args.targets:
        samples = sorted((args.sample_root / tid).glob("Sample_*.pdb"))
        if subset is not None:
            samples = [f for f in samples if f.name in subset.get(tid, set())]
            if len(samples) < args.min_subset:
                skipped[tid] = f"{len(samples)} samples in the subset (minimum {args.min_subset})"
                continue
        if not samples:
            skipped[tid] = "no samples"
            continue
        db = load_consensus_structure(tid, args.bpmaps_root / tid)
        fa = (args.bpmaps_root / tid / f"{tid}.fasta").read_text().split("\n")
        seq = "".join(l.strip() for l in fa if l and not l.startswith(">"))
        coords = {f.name: read_c1_coords(f) for f in samples}
        zs = [labeled_z(c, db, extract_z) for c in coords.values()]
        labels = [l for l in zs[0] if all(l in z for z in zs)]
        if not labels:
            skipped[tid] = "no inter-helix coordinate"
            continue
        Z = np.array([[z[l] for l in labels] for z in zs])
        d = {"db": db, "seq": seq, "names": [f.name for f in samples], "coords": coords, "labels": labels, "Z": Z,
             "v_gen": Z.var(axis=0)}
        if tid in pair_cfg:
            t = pair_cfg[tid]
            apo = labeled_z(map_to_input(read_models(t["apo_pdb"], t.get("apo_chain"))[0], seq), db, extract_z)
            holo = labeled_z(map_to_input(read_models(t["holo_pdb"], t.get("holo_chain"))[0], seq), db, extract_z)
            states = [map_full_to_input(read_full_model(t["apo_pdb"], t.get("apo_chain")), seq),
                      map_full_to_input(read_full_model(t["holo_pdb"], t.get("holo_chain")), seq)]
            table, cons = stem_conservation(db, states, parse_dotbracket, find_stems)
            d["pair"] = {"apo": apo, "holo": holo, "stems": table, "conserved": sorted(cons)}
        if tid in conc_cfg:
            t = conc_cfg[tid]
            if t["mode"] == "pair":
                models = [read_models(t["apo_pdb"], t.get("apo_chain"))[0], read_models(t["holo_pdb"], t.get("holo_chain"))[0]]
                maps, frac = _mapped(models, seq)
                used = [t["apo_pdb"], t["holo_pdb"]]
            else:
                best = None
                for cand in t.get("pdb_candidates", [t.get("pdb")]):
                    try:
                        mp, fr = _mapped(read_models(cand, t.get("chain")), seq)
                    except Exception:
                        continue
                    if best is None or fr > best[1]:
                        best = (mp, fr, cand)
                maps, frac, used = best[0], best[1], [best[2]]
            ez = [labeled_z(m, db, extract_z) for m in maps]
            if frac >= args.min_mapped and (t["mode"] == "pair" or len(ez) >= 5):
                stype = t["mode"]
                if any(Path(u).name.upper().startswith(MIXED_STATE_ENTRIES) for u in used):
                    stype = "mixed_state"
                d["conc"] = {"ez": ez, "type": stype, "files": [Path(u).name for u in used], "mapped": frac}
            else:
                d["conc_skip"] = f"mapped {frac:.3f}, {len(ez)} models"
        data[tid] = d
        print(f"{tid}: {len(samples)} samples, {len(labels)} coordinates"
              + (f", conserved stems {d['pair']['conserved']}" if "pair" in d else ""), flush=True)

    names0 = sorted(set.intersection(*[set(d["names"]) for d in data.values()])) if data else []
    primary = [None]
    extra = []
    if args.random_substrates and subset is None:
        first = data[next(iter(data))]["names"][0]
        pool = [n for n in names0 if n != first]
        rng = np.random.default_rng(args.seed)
        extra = list(rng.choice(pool, size=min(args.random_substrates, len(pool)), replace=False))
    out = {"sample_root": str(args.sample_root), "subset": bool(subset), "n_perturbations": args.n_perturbations,
           "skipped": skipped, "random_substrates": extra, "seed": args.seed,
           "stems": {t: d["pair"]["stems"] for t, d in data.items() if "pair" in d},
           "conserved_stems": {t: d["pair"]["conserved"] for t, d in data.items() if "pair" in d},
           "concordance_inputs": {t: {"type": d["conc"]["type"], "files": d["conc"]["files"], "mapped": d["conc"]["mapped"]}
                                  for t, d in data.items() if "conc" in d},
           "substrates": {}}
    jobs = [("primary", None, 0)] + [(x, x, 0) for x in extra] + [(f"seed_{k}", None, k) for k in range(1, args.perturbation_seeds + 1)]
    out["perturbation_seeds"] = list(range(1, args.perturbation_seeds + 1))
    for key, sub, pseed in jobs:
        per_t, disp, disp_c, rows = {}, {}, {}, []
        for tid, d in data.items():
            name = d["names"][0] if sub is None else sub
            fv, fl, n_ok = noise_floor(d["coords"][name], d["db"], n_perturbations=args.n_perturbations, seed=pseed)
            if fv is None:
                per_t[tid] = {"testable": False, "reason": f"baseline not estimable ({n_ok} accepted)", "substrate": name}
                continue
            c = clearing_for_target(d["v_gen"], d["labels"], fv, fl)
            per_t[tid] = {"testable": True, "substrate": name, "accepted": int(n_ok), **c}
            fmap = dict(zip(fl, np.sqrt(fv)))
            if "pair" in d:
                P = d["pair"]
                for target_dict, keep in ((disp, None), (disp_c, set(P["conserved"]))):
                    common = [l for l in d["labels"] if l in P["apo"] and l in P["holo"]
                              and (keep is None or label_stems(l) <= keep)]
                    if len(common) < 2:
                        target_dict[tid] = {"testable": False, "reason": f"{len(common)} coordinates"}
                        continue
                    j = [d["labels"].index(l) for l in common]
                    scale = np.array([fmap.get(l, 0.0) for l in common])
                    res = robust_pair(d["Z"][:, j], np.array([P["apo"][l] for l in common]),
                                      np.array([P["holo"][l] for l in common]),
                                      np.array([l.startswith("torsion") for l in common]), scale, common)
                    target_dict[tid] = displacement_summary(res)
            if "conc" in d:
                C = d["conc"]
                vb = dict(zip(fl, fv))
                common = [l for l in d["labels"] if all(l in z for z in C["ez"]) and l in vb]
                if len(common) >= 2:
                    j = [d["labels"].index(l) for l in common]
                    E = np.array([[z[l] for l in common] for z in C["ez"]])
                    r = coordinate_rows(tid, common, d["Z"][:, j], np.array([l.startswith("torsion") for l in common]),
                                        np.array([vb[l] for l in common]), E, centre="mean")
                    for x in r:
                        x["reference_type"] = C["type"]
                    rows.extend(r)
        strata = {"all": None, "all_without_mixed_state": ("nmr", "pair"), "nmr": ("nmr",), "pair": ("pair",),
                  "mixed_state": ("mixed_state",)}
        conc = {}
        for sname, types in strata.items():
            rr = [r for r in rows if types is None or r["reference_type"] in types]
            st = cohort_statistics([{k: r[k] for k in ("target", "type", "v_gen", "v_base", "v_exp")} for r in rr])
            conc[sname] = st
        out["substrates"][key] = {"clearing": per_t, "cohort": cohort_clearing({t: v for t, v in per_t.items() if v.get("testable")}),
                                  "displacement": disp, "displacement_conserved_stems": disp_c,
                                  "concordance": conc, "concordance_rows": rows}
        co = out["substrates"][key]["cohort"]
        print(f"[{key}] mean delta {co.get('mean_delta')}, sign p {co.get('sign_test_p')}; displacement criteria "
              f"{ {t: v.get('criteria_met') for t, v in disp.items()} }, conserved-stem "
              f"{ {t: v.get('criteria_met', v.get('reason')) for t, v in disp_c.items()} }; concordance all "
              f"{conc['all'].get('partial_spearman')}, without 8THV {conc['all_without_mixed_state'].get('partial_spearman')}", flush=True)
    if args.sweep_substrates:
        # clearing with every sample of each target (in a subset run: every passing sample) as the substrate
        sweep = {}
        for tid, d in data.items():
            rec = []
            for name in d["names"]:
                fv, fl, n_ok = noise_floor(d["coords"][name], d["db"], n_perturbations=args.n_perturbations)
                if fv is None:
                    rec.append({"substrate": name, "testable": False})
                    continue
                c = clearing_for_target(d["v_gen"], d["labels"], fv, fl)
                rec.append({"substrate": name, "testable": True, "distance": c["distance"], "angle": c["angle"], "delta": c["delta"],
                            "calls": c["calls"]})
            sweep[tid] = rec
            dl = [r["delta"] for r in rec if r.get("testable") and r["delta"] is not None]
            print(f"sweep {tid}: {len(rec)} substrates, delta {min(dl) if dl else None} to {max(dl) if dl else None}", flush=True)
        rng = np.random.default_rng(args.seed)
        from scipy.stats import binomtest
        pools = {t: [r["delta"] for r in v if r.get("testable") and r["delta"] is not None] for t, v in sweep.items()}
        pools = {t: v for t, v in pools.items() if v}
        means, ps = [], []
        for _ in range(10000):
            dd = np.array([v[rng.integers(len(v))] for v in pools.values()])
            means.append(dd.mean())
            nz = dd[dd != 0]
            ps.append(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0)
        means, ps = np.array(means), np.array(ps)
        out["substrate_sweep"] = {"per_target": sweep, "n_targets": len(pools),
                                  "combined": {"n_draws": 10000, "mean_delta_mean": float(means.mean()),
                                               "mean_delta_p2.5": float(np.percentile(means, 2.5)),
                                               "mean_delta_p97.5": float(np.percentile(means, 97.5)),
                                               "fraction_sign_p_below_0.05": float(np.mean(ps < 0.05)),
                                               "min_sign_p": float(ps.min())}}
        print("substrate sweep:", json.dumps(out["substrate_sweep"]["combined"]), flush=True)
    args.out.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"Wrote {args.out}")


def selftest():
    labels = ["d[sibling:stem0-stem1]", "theta[sibling:stem0-stem1]", "d[pseudoknot:stem2-stem1]", "theta[pseudoknot:stem2-stem1]"]
    c = clearing_for_target(np.array([4.0, 1.0, 0.5, 9.0]), labels, [1.0, 2.0, 1.0, 1.0], labels)
    assert c["distance"] == 0.5 and c["angle"] == 0.5 and c["delta"] == 0.0 and len(c["matched_pairs"]) == 2, c
    assert label_stems("theta[pseudoknot:stem2-stem1]") == {1, 2}
    co = cohort_clearing({"a": {"delta": 0.5, "matched_pairs": [(2, 2)]}, "b": {"delta": 0.0, "matched_pairs": [(0.5, 2)]},
                          "c": {"delta": 0.25, "matched_pairs": [(2, 0.5)]}})
    assert abs(co["mean_delta"] - 0.25) < 1e-12 and co["n_positive"] == 2 and abs(co["sign_test_p"] - 0.5) < 1e-12, co
    G = {"N1": np.zeros(3), "N2": np.array([0, 2.9, 0.0]), "O6": np.array([0, -2.9, 0.0]), "C1'": np.zeros(3)}
    C = {"N3": np.array([2.9, 0, 0]), "O2": np.array([2.9, 2.9, 0]), "N4": np.array([2.9, -2.9, 0]), "C1'": np.zeros(3)}
    far = {k: v + 20 for k, v in C.items()}
    st_ok = {0: ("G", G), 5: ("C", C), 1: ("G", G), 4: ("C", C)}
    st_melt = {0: ("G", G), 5: ("C", far), 1: ("G", G), 4: ("C", far)}
    class S:
        def __init__(s, i, p): s.id, s.pairs = i, p
    tab, cons = stem_conservation("((..))", [st_ok, st_ok], lambda db: [(0, 5), (1, 4)], lambda p: [S(0, p)])
    assert cons == {0}, tab
    tab, cons = stem_conservation("((..))", [st_ok, st_melt], lambda db: [(0, 5), (1, 4)], lambda p: [S(0, p)])
    assert cons == set() and tab[0]["formed_fraction"] == [1.0, 0.0], tab
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--bpmaps-root", type=Path)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--pairs-config", type=Path)
    ap.add_argument("--concordance-config", type=Path)
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--n-perturbations", type=int, default=300)
    ap.add_argument("--random-substrates", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturbation-seeds", type=int, default=0,
                    help="also repeat everything with the primary substrate and perturbation seeds 1..N")
    ap.add_argument("--sweep-substrates", action="store_true",
                    help="also compute clearing with every sample (every passing sample with --subset-checks) as the substrate")
    ap.add_argument("--subset-checks", type=Path)
    ap.add_argument("--min-subset", type=int, default=10)
    ap.add_argument("--min-mapped", type=float, default=0.85)
    ap.add_argument("--out", type=Path, default=Path("ensemble_analyses.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.sample_root and a.bpmaps_root and a.targets):
        ap.error("--sample-root, --bpmaps-root and --targets are required")
    run(a)


if __name__ == "__main__":
    main()
