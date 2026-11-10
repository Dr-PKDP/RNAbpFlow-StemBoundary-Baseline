"""Cluster structure of the generated ensemble across a range of dendrogram cut heights.

Reports the number of clusters and their populations at several cut heights (hierarchical clustering of the
collective-coordinate vector z) and the within-cluster spread at the 50% cut. No single threshold is selected. Reuses
read_c1_coords and extract_z from the main pipeline.

Usage:
    python target_multimodality_diagnostic.py <target_name> <sample_dir> <dotbracket_string>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from clearing import read_c1_coords
from collective_coords import extract_z


def load_z_matrix(sample_dir: Path, dotbracket: str):
    sample_files = sorted(sample_dir.glob("Sample_*.pdb"))
    if not sample_files:
        raise FileNotFoundError(f"no Sample_*.pdb files found in {sample_dir}")
    z_list, labels = [], None
    for f in sample_files:
        coords = read_c1_coords(f)
        z, labels = extract_z(coords, dotbracket)
        z_list.append(z)
    return np.array(z_list), labels, [f.name for f in sample_files]


def cluster_at_heights(z_arr: np.ndarray, heights_as_fraction_of_max: tuple = (0.3, 0.4, 0.5, 0.6, 0.7)):
    """Average-linkage hierarchical clustering on pairwise Euclidean distance
    in z-space. Reports cluster structure at
    several cut heights (as a fraction of the dendrogram's max linkage
    distance) rather than committing to one threshold."""
    dist_matrix = pdist(z_arr, metric="euclidean")
    Z = linkage(dist_matrix, method="average")
    max_height = Z[-1, 2]

    results = {}
    for frac in heights_as_fraction_of_max:
        cut_height = frac * max_height
        cluster_ids = fcluster(Z, t=cut_height, criterion="distance")
        n_clusters = len(set(cluster_ids))
        populations = sorted(
            (cluster_ids == cid).sum() for cid in set(cluster_ids)
        )[::-1]
        results[frac] = {
            "cut_height": float(cut_height),
            "n_clusters": n_clusters,
            "populations": populations,
        }
    return results, Z


def within_cluster_covariance_norms(z_arr: np.ndarray, cluster_ids: np.ndarray):
    """For each cluster with enough members to compute a covariance,
    report its Frobenius norm -- tests whether clusters individually
    converge tightly even if the pooled ensemble does not."""
    norms = {}
    for cid in sorted(set(cluster_ids)):
        members = z_arr[cluster_ids == cid]
        if len(members) < 3:
            norms[int(cid)] = {"n": len(members), "frob_norm": None}
            continue
        cov = np.cov(members.T)
        norms[int(cid)] = {"n": len(members), "frob_norm": float(np.linalg.norm(cov, "fro"))}
    return norms


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    target_name, sample_dir, dotbracket = sys.argv[1], Path(sys.argv[2]), sys.argv[3]

    z_arr, labels, sample_names = load_z_matrix(sample_dir, dotbracket)
    print(f"{target_name}: {z_arr.shape[0]} samples, {z_arr.shape[1]}-dim z-vector "
          f"({sum(1 for l in labels if l.startswith('d['))} distance, "
          f"{sum(1 for l in labels if l.startswith('theta['))} angle/torsion dims)")

    print("\n"
          "cluster count and population at several cut heights:")
    results, Z = cluster_at_heights(z_arr)
    for frac, r in results.items():
        pop_str = ", ".join(str(p) for p in r["populations"][:6])
        more = "..." if len(r["populations"]) > 6 else ""
        print(f"  cut at {frac:.0%} of max linkage distance ({r['cut_height']:.2f}): "
              f"{r['n_clusters']} clusters, populations [{pop_str}{more}]")

    # within-cluster spread at the 50% cut height
    mid_cut = results[0.5]["cut_height"]
    cluster_ids = fcluster(Z, t=mid_cut, criterion="distance")
    print(f"\nAt the 50%-height cut ({results[0.5]['n_clusters']} clusters), "
          f"per-cluster covariance Frobenius norm:")
    norms = within_cluster_covariance_norms(z_arr, cluster_ids)
    for cid, info in norms.items():
        if info["frob_norm"] is None:
            print(f"  cluster {cid}: n={info['n']} (too small for covariance)")
        else:
            print(f"  cluster {cid}: n={info['n']}  frob_norm={info['frob_norm']:.2f}")

    pooled_cov = np.cov(z_arr.T)
    pooled_norm = float(np.linalg.norm(pooled_cov, "fro"))
    print(f"\npooled (all {z_arr.shape[0]} samples) frob_norm={pooled_norm:.2f}")
    print("Two things distinguish real mode-mixing from a hierarchical clustering "
          "artifact on genuinely continuous spread -- validated against synthetic "
          "test cases before this script was used on real data:")
    print("  1. STABILITY ACROSS CUT HEIGHTS. A cut through a real gap between modes "
          "gives the same cluster count and membership across a wide range of cut "
          "heights. A cut through a continuum shifts count and membership as the "
          "height changes, since there is no real gap to lock onto.")
    print("  2. MAGNITUDE OF THE PER-CLUSTER NORM GAP. Real separate modes tend to "
          "differ by orders of magnitude in internal spread (a tight mode next to a "
          "diffuse one, or a rare outlier cluster next to the main body). Arbitrary "
          "slices through one continuum tend to have per-cluster norms within roughly "
          "the same order of magnitude as each other.")
    print("If the cluster count above is stable across most or all of the five cut "
          "heights AND per-cluster norms differ by a large factor, that is real "
          "evidence for mode-mixing (Scenario B). If cluster count keeps changing "
          "across heights and per-cluster norms are all similar in magnitude, that "
          "points to genuine broad unimodal spread (Scenario A) instead.")


if __name__ == "__main__":
    main()
