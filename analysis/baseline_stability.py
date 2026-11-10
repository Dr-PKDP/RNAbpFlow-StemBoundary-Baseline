#!/usr/bin/env python3
"""Stability of the perturbation baseline for one target.

For 7KD1 the baseline SD of theta[parent-child:stem2-stem3] rests on few accepted
perturbations, and the variance enrichment along the apo->holo displacement
(9.55) is carried by that angle. This script repeats the baseline estimate over
independent perturbation seeds and, for pair targets, recomputes the
enrichment under each, to show how much the result depends on the baseline draw.

Output per seed: accepted perturbations (n_ok), baseline SD of the key coordinate,
and enrichment along the displacement (baseline scaling).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--target", default="7kd1")
    ap.add_argument("--coordinate", default="theta[parent-child:stem2-stem3]")
    ap.add_argument("--sample-root", type=Path, required=True)
    ap.add_argument("--bpmaps-root", type=Path, required=True)
    ap.add_argument("--pipeline-dir", type=Path, default=Path("../pipeline"))
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-perturbations", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("baseline_stability.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(a.pipeline_dir)); sys.path.insert(0, str(Path(__file__).parent))
    from collective_coords import extract_z
    from clearing import read_c1_coords, load_consensus_structure, noise_floor
    from displacement_direction_test import (labeled_z, read_models, map_to_input, pair_statistics)
    t = next(x for x in json.loads(a.config.read_text()) if x["id"] == a.target)
    tid = a.target
    db = load_consensus_structure(tid, a.bpmaps_root / tid)
    fa = (a.bpmaps_root / tid / f"{tid}.fasta").read_text().split("\n")
    seq = "".join(l.strip() for l in fa if l and not l.startswith(">"))
    samples = sorted((a.sample_root / tid).glob("Sample_*.pdb"))
    zs = [labeled_z(read_c1_coords(f), db, extract_z) for f in samples]
    apo = labeled_z(map_to_input(read_models(t["apo_pdb"], t.get("apo_chain"))[0], seq), db, extract_z)
    holo = labeled_z(map_to_input(read_models(t["holo_pdb"], t.get("holo_chain"))[0], seq), db, extract_z)
    common = [l for l in zs[0] if all(l in z for z in zs) and l in apo and l in holo]
    is_tors = np.array([l.startswith("torsion") for l in common])
    Z = np.array([[z[l] for l in common] for z in zs])
    za = np.array([apo[l] for l in common]); zh = np.array([holo[l] for l in common])
    k = common.index(a.coordinate)
    v_gen = float(np.var(Z[:, k]))
    v_exp = float((za[k] - zh[k]) ** 2 / 2.0)
    rows = []
    for seed in range(a.n_seeds):
        fv, fl, n_ok = noise_floor(read_c1_coords(samples[0]), db, n_perturbations=a.n_perturbations, seed=seed)
        if fv is None:
            rows.append({"seed": seed, "n_ok": n_ok, "estimable": False}); continue
        fd = dict(zip(fl, np.sqrt(fv)))
        scale = np.array([fd.get(l, 0.0) for l in common])
        res = pair_statistics(Z, za, zh, is_tors, scale)
        rows.append({"seed": seed, "n_ok": int(n_ok), "estimable": True,
                     "baseline_sd_key": float(scale[k]),
                     "enrichment": float(res["variance_enrichment"]),
                     "gen_over_base_key": (v_gen / float(scale[k]) ** 2) if scale[k] > 0 else None,
                     "exp_over_base_key": (v_exp / float(scale[k]) ** 2) if scale[k] > 0 else None})
        print(rows[-1])
    ok = [r for r in rows if r["estimable"]]
    if ok:
        e = np.array([r["enrichment"] for r in ok]); s = np.array([r["baseline_sd_key"] for r in ok])
        print(f"estimable {len(ok)}/{len(rows)}; n_ok range {min(r['n_ok'] for r in ok)}-{max(r['n_ok'] for r in ok)}")
        print(f"key-coordinate baseline SD: median {np.median(s):.3f}, range {s.min():.3f}-{s.max():.3f}")
        print(f"enrichment: median {np.median(e):.2f}, range {e.min():.2f}-{e.max():.2f}, share > 1: {(e>1).mean():.2f}")
    g = [r["gen_over_base_key"] for r in ok if r["gen_over_base_key"] is not None]
    if g:
        e2 = [r["exp_over_base_key"] for r in ok if r["exp_over_base_key"] is not None]
        print(f"key coordinate generated/baseline variance: median {np.median(g):.1f}, range {min(g):.1f}-{max(g):.1f}; "
              f"share > 1: {np.mean(np.array(g) > 1):.2f} (draws with baseline SD > 0: {len(g)}/{len(ok)})")
        print(f"key coordinate experimental/baseline variance: median {np.median(e2):.1f}, range {min(e2):.1f}-{max(e2):.1f}")
    a.out.write_text(json.dumps(rows, indent=2)); print("Wrote", a.out)

if __name__ == "__main__":
    main()
