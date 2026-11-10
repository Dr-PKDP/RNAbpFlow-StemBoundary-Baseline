"""Per-target clearing analysis.

For one or more targets: collective coordinates of every generated sample, the convergence curve over sample count,
the bracket-type-preserving, stem-corresponding perturbation baseline (noise_floor) and the clearing indicator for each coordinate.
Also provides the helpers imported by the analysis scripts (read_c1_coords, load_consensus_structure, noise_floor).

Usage:
    python clearing.py --targets <id> [<id> ...] --sample-root ../data/Predictions --bpmaps-root ../data/Inputs \
        --n-perturbations 300 --out report.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from collective_coords import (extract_z, parse_dotbracket, find_stems, Stem,
                                   nested_pairs_only, mark_pseudoknot_stems)


def read_c1_coords(pdb_path: Path) -> dict:
    coords = {}
    seen_res_keys = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "C1'":
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26]
            res_key = (chain, resseq, icode)
            if res_key in seen_res_keys:
                continue
            seen_res_keys.append(res_key)
            idx = len(seen_res_keys) - 1
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            coords[idx] = np.array([x, y, z])
    return coords


def load_consensus_structure(target_name: str, structure_dir: Path,
                              prob_threshold: float = 0.5) -> str:
    """Derives one dot-bracket secondary structure for a target.

    If IPknot returned at least one crossing pair (pseudoknot_pairs.json), its complete prediction (nested and
    crossing components) is used. Splicing crossing pairs into the nested three-tool consensus is not done: for 4RZD
    the consensus pairs the crossing region in a nested-only arrangement, and a merge that gives the consensus priority
    accepted none of the seven crossing pairs. Otherwise the structure is the consensus of map1-3.npy
    (MXfold2, SimFold, CONTRAfold), thresholded at prob_threshold.
    """
    pk_path = structure_dir / "pseudoknot_pairs.json"
    if pk_path.exists():
        pk_data = json.loads(pk_path.read_text())
        crossing = pk_data.get("crossing_pairs", [])
        if crossing:
            db = pk_data["ipknot_dotbracket"]
            print(f"  [{target_name}] pseudoknot ({len(crossing)} crossing pairs): "
                  f"using the IPknot prediction", file=sys.stderr)
            return db

    maps = []
    for i in (1, 2, 3):
        p = structure_dir / f"map{i}.npy"
        if p.exists():
            maps.append(np.load(p))
    if not maps:
        raise FileNotFoundError(
            f"no map1/2/3.npy found in {structure_dir} for {target_name!r}")
    L = maps[0].shape[0]
    for m in maps:
        if m.shape != (L, L):
            raise ValueError(f"map shape mismatch for {target_name!r}: {m.shape} vs {(L, L)}")

    mean_prob = np.mean(maps, axis=0)
    mean_prob = (mean_prob + mean_prob.T) / 2.0

    candidates = []
    for i in range(L):
        for j in range(i + 1, L):
            p = mean_prob[i, j]
            if p >= prob_threshold:
                candidates.append((p, i, j))
    candidates.sort(reverse=True)

    used = set()
    nested_pairs = []
    for p, i, j in candidates:
        if i in used or j in used:
            continue
        nested_pairs.append((i, j))
        used.add(i)
        used.add(j)

    db = ["."] * L
    for i, j in nested_pairs:
        db[i] = "("
        db[j] = ")"
    return "".join(db)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_aligned = (R @ Pc.T).T + Q.mean(0)
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1))))
    return rmsd, R, Q.mean(0) - R @ P.mean(0)


def raw_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


def perturb_dotbracket(db: str, rng: np.random.Generator, max_shift=1):
    pairs = parse_dotbracket(db)
    stems = find_stems(pairs)
    # Pseudoknot stems keep their bracket type when pairs are written back.
    nested_set = set(nested_pairs_only(pairs))
    mark_pseudoknot_stems(stems, nested_set)
    n = len(db)

    kept = []  # list of (i, j, is_pk) so the correct bracket type can be written below
    for stem in stems:
        stem_pairs = list(stem.pairs)
        is_pk = stem.is_pseudoknot
        if len(stem_pairs) == 1:
            i, j = stem_pairs[0]
            kept.append((i, j, is_pk))
            continue
        outer_i, outer_j = stem_pairs[0]
        inner_i, inner_j = stem_pairs[-1]

        shift_outer = int(rng.integers(-max_shift, max_shift + 1))
        new_outer_i = np.clip(outer_i + shift_outer, 0, inner_i - 1)
        new_outer_j = np.clip(outer_j - shift_outer, inner_j + 1, n - 1)

        shift_inner = int(rng.integers(-max_shift, max_shift + 1))
        new_inner_i = np.clip(inner_i + shift_inner, new_outer_i + 1, n - 1)
        new_inner_j = np.clip(inner_j - shift_inner, 0, new_outer_j - 1)
        if new_inner_i >= new_inner_j:
            new_inner_i, new_inner_j = inner_i, inner_j

        middle = stem_pairs[1:-1]
        kept.append((int(new_outer_i), int(new_outer_j), is_pk))
        kept.extend((i, j, is_pk) for (i, j) in middle)
        kept.append((int(new_inner_i), int(new_inner_j), is_pk))

    new_db = ["."] * n
    used_positions = set()
    for (i, j, is_pk) in kept:
        if i in used_positions or j in used_positions or i >= j:
            continue
        new_db[i] = "[" if is_pk else "("
        new_db[j] = "]" if is_pk else ")"
        used_positions.add(i)
        used_positions.add(j)
    return "".join(new_db)


def stem_correspondence(db: str, db2: str, tol: int = 1) -> bool:
    """True when every stem of the perturbed map is the same stem as in the unperturbed map.

    Identity is decided from residue-pair membership, not from the coordinate label: the two maps must have the same
    number of stems, each stem k must have the same bracket class, and the outermost and innermost base pair of stem k
    must lie within `tol` positions of those of stem k in the unperturbed map (the boundary shift of
    perturb_dotbracket is at most 1). A split, merge, loss or reordering of stems therefore fails the test."""
    try:
        p1, p2 = parse_dotbracket(db), parse_dotbracket(db2)
    except Exception:
        return False
    s1, s2 = find_stems(p1), find_stems(p2)
    if len(s1) != len(s2):
        return False
    n1, n2 = set(nested_pairs_only(p1)), set(nested_pairs_only(p2))
    mark_pseudoknot_stems(s1, n1); mark_pseudoknot_stems(s2, n2)
    for a, b in zip(s1, s2):
        if a.is_pseudoknot != b.is_pseudoknot:
            return False
        for (x, y) in ((a.pairs[0], b.pairs[0]), (a.pairs[-1], b.pairs[-1])):
            if abs(x[0] - y[0]) > tol or abs(x[1] - y[1]) > tol:
                return False
    return True


def perturbed_z(coords: dict, db: str, n_perturbations=300, seed=0, correspondence=True):
    """Collective-coordinate vectors of the accepted perturbations (array, n_accepted x n_dims) and the coordinate labels.

    A perturbation is accepted when the vector can be computed and has the length of the unperturbed vector. With
    correspondence=True (the default) it must also have the labels of the unperturbed vector and satisfy
    stem_correspondence, so that position j of every accepted vector describes the same stem relationship as position j
    of the unperturbed vector. correspondence=False gives the length-only (equal-length) rule."""
    rng = np.random.default_rng(seed)
    zs = []
    base_z, base_labels = extract_z(coords, db)
    for _ in range(n_perturbations):
        pdb2 = perturb_dotbracket(db, rng)
        try:
            z2, labels2 = extract_z(coords, pdb2)
        except Exception:
            continue
        if len(z2) != len(base_z):
            continue
        if correspondence and not (list(labels2) == list(base_labels) and stem_correspondence(db, pdb2)):
            continue
        zs.append(z2)
    return np.array(zs), base_labels


def noise_floor(coords: dict, db: str, n_perturbations=300, seed=0, correspondence=True):
    zs, base_labels = perturbed_z(coords, db, n_perturbations, seed, correspondence)
    if len(zs) < 10:
        return None, base_labels, len(zs)
    return zs.var(axis=0), base_labels, len(zs)


def k_star(convergence: dict, suffix: str = "", threshold: float = 0.10, exclude_full_pool: bool = True):
    """Smallest tested K with CV < threshold whose next tested count also has CV < threshold (Eq. 2).

    With exclude_full_pool, the largest tested count (the whole pool, where the CV is zero by construction) is not
    accepted as the second condition, so a K whose only successor is the full pool is not returned."""
    ks = sorted(convergence)
    cv = {k: convergence[k][f"std_frob_norm{suffix}"] / convergence[k][f"mean_frob_norm{suffix}"]
          for k in ks if f"std_frob_norm{suffix}" in convergence[k] and convergence[k][f"mean_frob_norm{suffix}"] > 0}
    for a, b in zip(ks[:-1], ks[1:]):
        if a in cv and b in cv and cv[a] < threshold and cv[b] < threshold and not (exclude_full_pool and b == ks[-1]):
            return a
    return None


def run_target(target_name: str, sample_dir: Path, dotbracket: str, out: dict,
                n_perturbations: int = 300, correspondence: bool = True):
    sample_files = sorted(sample_dir.glob("Sample_*.pdb"))
    if not sample_files:
        raise FileNotFoundError(f"no Sample_*.pdb files found in {sample_dir}")
    K = len(sample_files)

    all_coords = []
    z_list = []
    labels = None
    for f in sample_files:
        coords = read_c1_coords(f)
        if len(coords) != len(dotbracket):
            print(f"  WARNING [{target_name}/{f.name}]: {len(coords)} C1' atoms read, "
                  f"{len(dotbracket)} positions in structure -- indexing may be off",
                  file=sys.stderr)
        all_coords.append(coords)
        z, labels = extract_z(coords, dotbracket)
        z_list.append(z)

    z_arr = np.array(z_list)
    n_dist = sum(1 for l in labels if l.startswith("d["))
    n_ang = sum(1 for l in labels if l.startswith("theta["))

    keys_common = sorted(set.intersection(*[set(c.keys()) for c in all_coords]))
    coord_mats = [np.array([c[k] for k in keys_common]) for c in all_coords]
    raw_rmsds, sup_rmsds = [], []
    for a in range(K):
        for b in range(a + 1, K):
            raw_rmsds.append(raw_rmsd(coord_mats[a], coord_mats[b]))
            r, _, _ = kabsch_rmsd(coord_mats[a], coord_mats[b])
            sup_rmsds.append(r)
    raw_mean, sup_mean = float(np.mean(raw_rmsds)), float(np.mean(sup_rmsds))

    z_var_dist = z_arr[:, :n_dist].var(axis=0) if n_dist else np.array([])
    z_var_ang = z_arr[:, n_dist:].var(axis=0) if n_ang else np.array([])

    floor_var, floor_labels, n_ok = noise_floor(all_coords[0], dotbracket,
                                                  n_perturbations=n_perturbations, correspondence=correspondence)
    floor_dist = floor_var[:n_dist] if floor_var is not None and n_dist else None
    floor_ang = floor_var[n_dist:] if floor_var is not None and n_ang else None

    convergence = {}
    ks_to_check = [k for k in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100) if k <= K]
    rng = np.random.default_rng(42)
    sd_all = z_arr.std(axis=0)
    sd_all[sd_all < 1e-9] = 1.0
    z_std = z_arr / sd_all  # unit-free coordinates: every column has SD 1 over the full ensemble
    blocks = {"all": slice(0, z_arr.shape[1]), "distance": slice(0, n_dist), "angle": slice(n_dist, z_arr.shape[1])}
    for k in ks_to_check:
        norms = {b: [] for b in ("all", "distance", "angle", "standardised")}
        for _ in range(50):
            idx = rng.choice(K, size=k, replace=False)
            norms["all"].append(np.linalg.norm(np.atleast_2d(np.cov(z_arr[idx].T)), "fro"))
            if n_dist:
                norms["distance"].append(np.linalg.norm(np.atleast_2d(np.cov(z_arr[idx][:, blocks["distance"]].T)), "fro"))
            if n_ang:
                norms["angle"].append(np.linalg.norm(np.atleast_2d(np.cov(z_arr[idx][:, blocks["angle"]].T)), "fro"))
            norms["standardised"].append(np.linalg.norm(np.atleast_2d(np.cov(z_std[idx].T)), "fro"))
        entry = {"mean_frob_norm": float(np.mean(norms["all"])), "std_frob_norm": float(np.std(norms["all"]))}
        for b in ("distance", "angle", "standardised"):
            if norms[b]:
                entry[f"mean_frob_norm_{b}"] = float(np.mean(norms[b]))
                entry[f"std_frob_norm_{b}"] = float(np.std(norms[b]))
        convergence[k] = entry
    out_k_star = {}
    for b, suffix in (("all", ""), ("distance", "_distance"), ("angle", "_angle"), ("standardised", "_standardised")):
        out_k_star[b] = k_star(convergence, suffix, exclude_full_pool=True)

    out[target_name] = {
        "K": K,
        "n_distance_coords": n_dist,
        "n_angle_coords": n_ang,
        "raw_rmsd_mean": raw_mean,
        "superposed_rmsd_mean": sup_mean,
        "global_rigid_body_fraction": 1.0 - (sup_mean / raw_mean) if raw_mean > 0 else None,
        "z_variance_distance_block": z_var_dist.tolist(),
        "z_variance_angle_block": z_var_ang.tolist(),
        "noise_floor_distance_block": floor_dist.tolist() if floor_dist is not None else None,
        "noise_floor_angle_block": floor_ang.tolist() if floor_ang is not None else None,
        "noise_floor_n_valid_perturbations": n_ok,
        "noise_floor_n_perturbations_attempted": n_perturbations,
        "signal_to_floor_ratio_distance": (
            [(round(s / f, 2) if f > 1e-6 else "floor near-zero, ratio not meaningful")
             for s, f in zip(z_var_dist, floor_dist)]
            if floor_dist is not None and n_dist else
            (f"not computed -- only {n_ok} valid perturbations survived "
             f"(need >= 10)" if n_dist else None)
        ),
        "signal_to_floor_ratio_angle": (
            [(round(s / f, 2) if f > 1e-6 else "floor near-zero, ratio not meaningful")
             for s, f in zip(z_var_ang, floor_ang)]
            if floor_ang is not None and n_ang else
            (f"not computed -- only {n_ok} valid perturbations survived "
             f"(need >= 10)" if n_ang else None)
        ),
        "k_convergence": convergence,
        "k_star_excluding_full_pool": out_k_star,
        "coordinate_labels": labels,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--sample-root", type=Path, required=True)
    ap.add_argument("--bpmaps-root", type=Path, default=None)
    ap.add_argument("--dotbracket-file", type=Path, default=None)
    ap.add_argument("--n-perturbations", type=int, default=300,
                     help="perturbation attempts for the noise floor; raise it for a target "
                          "with too few accepted perturbations at the default (no new "
                          "inference is needed, since only the base-pair map is "
                          "re-perturbed against the generated sample coordinates)")
    ap.add_argument("--length-only-baseline", action="store_true",
                     help="accept perturbed vectors on length alone (default: also require identical labels and "
                          "stem identity)")
    ap.add_argument("--out", type=Path, default=Path("clearing_report.json"))
    args = ap.parse_args()

    if args.dotbracket_file:
        db_map = json.loads(args.dotbracket_file.read_text())
    else:
        db_map = None

    if not args.dotbracket_file and not args.bpmaps_root:
        ap.error("--bpmaps-root is required unless --dotbracket-file is given")

    out = {}
    for t in args.targets:
        db = db_map[t] if db_map else load_consensus_structure(t, args.bpmaps_root / t)
        print(f"Running {t} ...")
        run_target(t, args.sample_root / t, db, out, n_perturbations=args.n_perturbations,
                   correspondence=not args.length_only_baseline)
        r = out[t]
        print(f"  K={r['K']}  raw/superposed RMSD = {r['raw_rmsd_mean']:.2f} / "
              f"{r['superposed_rmsd_mean']:.2f} A  "
              f"(global rigid-body fraction = {r['global_rigid_body_fraction']:.2f})")
        print(f"  valid perturbations: {r['noise_floor_n_valid_perturbations']}/"
              f"{r['noise_floor_n_perturbations_attempted']} attempted")
        if r["signal_to_floor_ratio_distance"]:
            print(f"  distance-block signal/floor: {r['signal_to_floor_ratio_distance']}")
        if r["signal_to_floor_ratio_angle"]:
            print(f"  angle-block signal/floor: {r['signal_to_floor_ratio_angle']}")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
