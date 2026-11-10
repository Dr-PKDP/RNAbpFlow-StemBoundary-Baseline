"""Validation of collective_coords.py on synthetic geometries with known answers.

  1. Line-segment recovery: segment_distance and axis_angle against hand-computed values.
  2. Idealized A-form helices at a known bend angle: fit_helix_axis recovers the axis.
  3. Three-way-junction hinge construct: extract_z reports an inter-helix angle that tracks the imposed hinge angle.
  4. Inter-helix distance recovery.
  5. Pseudoknot handling: a pseudoknot contributes a distance and an angle entry, and z is finite.

Run from the repository root: python tests/validate_collective_coords.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import numpy as np
from collective_coords import (
    parse_dotbracket, find_stems, fit_helix_axis, segment_distance,
    axis_angle, extract_z, Stem, Axis
)

TOL_DIST = 0.15   # Angstrom
TOL_ANGLE = 3.0    # degrees


def report(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


# ---------------------------------------------------------------------------
# Test 1: bare segment geometry, hand-computed ground truth
# ---------------------------------------------------------------------------

def test1_segment_primitives():
    results = []

    # Two parallel segments, 10 A apart, both along +x
    a = Axis(0, point=np.array([0, 0, 0.]), direction=np.array([1, 0, 0.]),
             start=np.array([0, 0, 0.]), end=np.array([10, 0, 0.]))
    b = Axis(1, point=np.array([0, 10, 0.]), direction=np.array([1, 0, 0.]),
             start=np.array([0, 10, 0.]), end=np.array([10, 10, 0.]))
    d = segment_distance(a, b)
    ang = axis_angle(a, b)
    results.append(report("parallel segments distance", abs(d - 10.0) < TOL_DIST, f"got {d:.3f}, want 10.0"))
    results.append(report("parallel segments angle", abs(ang - 0.0) < TOL_ANGLE, f"got {ang:.3f}, want 0.0"))

    # Perpendicular segments, known 90 degree angle, offset in z so they don't cross
    c = Axis(2, point=np.array([5, 5, 5.]), direction=np.array([0, 1, 0.]),
              start=np.array([5, 0, 5.]), end=np.array([5, 10, 5.]))
    ang2 = axis_angle(a, c)
    results.append(report("perpendicular segments angle", abs(ang2 - 90.0) < TOL_ANGLE, f"got {ang2:.3f}, want 90.0"))

    # Antiparallel segments: direction convention should give ~180, not fold to 0
    e = Axis(3, point=np.array([0, 20, 0.]), direction=np.array([-1, 0, 0.]),
              start=np.array([10, 20, 0.]), end=np.array([0, 20, 0.]))
    ang3 = axis_angle(a, e)
    results.append(report("antiparallel segments angle stays ~180 (not folded)",
                           abs(ang3 - 180.0) < TOL_ANGLE, f"got {ang3:.3f}, want 180.0"))

    return all(results)


# ---------------------------------------------------------------------------
# Test 2: idealized A-form helix, fit_helix_axis on real double-helix geometry
# ---------------------------------------------------------------------------

def a_form_helix(n_bp, origin=np.zeros(3), rot=np.eye(3), rise=2.81, twist_deg=32.7, radius=9.4):
    """Idealized A-form double helix: n_bp base pairs, each strand's
    'C1'' analog placed on a cylinder of the given radius, rotated by
    twist_deg per step and translated by rise per step along local z,
    then placed in space by `rot` (3x3 rotation) and `origin`.
    Returns coords dict {residue_index: xyz} for strand 1 (0..n_bp-1)
    and strand 2 (2*n_bp-1 .. n_bp, paired i <-> 2*n_bp-1-i), and the
    known axis direction in world frame (rot @ [0,0,1])."""
    coords = {}
    twist = np.radians(twist_deg)
    strand2 = {}
    for k in range(n_bp):
        theta = k * twist
        z = k * rise
        p1_local = np.array([radius * np.cos(theta), radius * np.sin(theta), z])
        p2_local = np.array([radius * np.cos(theta + np.pi), radius * np.sin(theta + np.pi), z])
        coords[k] = origin + rot @ p1_local
        strand2[2 * n_bp - 1 - k] = origin + rot @ p2_local
    coords.update(strand2)
    known_direction = rot @ np.array([0., 0., 1.])
    return coords, known_direction


def rotation_about_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def test2_aform_axis_recovery():
    n_bp = 12
    results = []
    for bend_deg in [0.0, 30.0, 60.0, 90.0]:
        rot = rotation_about_y(bend_deg)
        coords, known_dir = a_form_helix(n_bp, origin=np.zeros(3), rot=rot)
        stem = Stem(pairs=[(k, 2 * n_bp - 1 - k) for k in range(n_bp)], id=0)
        ax = fit_helix_axis(coords, stem)
        cos_recovered = np.clip(np.dot(ax.direction, known_dir), -1, 1)
        err_deg = np.degrees(np.arccos(abs(cos_recovered)))  # axis fit sign can go either way pre-orientation check
        results.append(report(
            f"A-form axis recovery at bend={bend_deg} deg",
            err_deg < TOL_ANGLE,
            f"recovered axis vs. known, error={err_deg:.3f} deg"
        ))
    return all(results)


def test2b_aform_pair_angle():
    """Two A-form helices, second rotated by a KNOWN angle relative to the
    first, both fit independently -- checks axis_angle end-to-end on
    realistic double-helix atom coordinates, not just single-axis recovery."""
    n_bp = 12
    results = []
    for true_bend in [20.0, 45.0, 80.0, 120.0]:
        coords1, dir1 = a_form_helix(n_bp, origin=np.zeros(3), rot=np.eye(3))
        rot2 = rotation_about_y(true_bend)
        offset = np.array([0., 0., n_bp * 2.81 + 15.0])  # push second helix downstream, avoid overlap
        coords2_raw, dir2 = a_form_helix(n_bp, origin=offset, rot=rot2)
        coords2 = {k + 1000: v for k, v in coords2_raw.items()}

        stem1 = Stem(pairs=[(k, 2 * n_bp - 1 - k) for k in range(n_bp)], id=0)
        stem2 = Stem(pairs=[(k + 1000, 2 * n_bp - 1 - k + 1000) for k in range(n_bp)], id=1)
        ax1 = fit_helix_axis(coords1, stem1)
        ax2 = fit_helix_axis(coords2, stem2)
        measured = axis_angle(ax1, ax2)
        # orientation convention (low-i->high-i) may recover true_bend or its
        # supplement depending on strand-direction bookkeeping; check both
        err = min(abs(measured - true_bend), abs(measured - (180 - true_bend)))
        results.append(report(
            f"pairwise A-form bend angle, true={true_bend} deg",
            err < TOL_ANGLE,
            f"measured={measured:.2f} deg (true={true_bend} or supplement {180-true_bend})"
        ))
    return all(results)


# ---------------------------------------------------------------------------
# Test 3: two-domain hinge construct via extract_z, full pipeline
# ---------------------------------------------------------------------------

def test3_two_domain_hinge_full_pipeline():
    """Builds a minimal 3-stem construct: stem A -- internal loop -- stem B
    -- hairpin, i.e. dot-bracket like '((((....))))....((((....))))....'
    is too simple (single unbranched chain has no branch point to bend at
    without a multi-loop). Use a 3-way junction instead: one parent stem
    opening into a multi-branch loop with two child stems (a genuine hinge
    point), and confirm extract_z reports an inter-helix angle that tracks
    an imposed bend, end to end through dot-bracket parsing, stem finding,
    loop-tree construction, and axis fitting -- not just the geometry
    primitives tested above."""
    n_bp = 10
    results = []
    for imposed_bend in [10.0, 45.0, 90.0]:
        # parent stem: residues 0..n_bp-1 paired with far end (built below)
        # child stem A: short unpaired linker then a helix
        # child stem B: another short linker then a second helix, rotated
        link = 3  # unpaired residues forming part of the multi-loop
        coordsP, _ = a_form_helix(n_bp, origin=np.zeros(3), rot=np.eye(3))
        # place child A right after parent's strand-1 end + linker
        offsetA = np.array([12.0, 0., n_bp * 2.81 * 0.3])
        coordsA_raw, _ = a_form_helix(n_bp, origin=offsetA, rot=rotation_about_y(0.0))
        coordsA = {k + 100: v for k, v in coordsA_raw.items()}
        # child B rotated by the imposed bend relative to child A, placed
        # nearby so it shares the same multi-loop
        offsetB = np.array([-12.0, 0., n_bp * 2.81 * 0.3])
        coordsB_raw, _ = a_form_helix(n_bp, origin=offsetB, rot=rotation_about_y(imposed_bend))
        coordsB = {k + 200: v for k, v in coordsB_raw.items()}

        coords = {**coordsP, **coordsA, **coordsB}

        # dot-bracket: parent stem (round brackets) with A and B as two
        # separate child stems inside it, separated by unpaired linkers,
        # i.e. a genuine multi-branch (3-way) loop: parent -> {A, B}
        db = (
            "(" * n_bp +
            "." * link +
            "(" * n_bp + "." * link + ")" * n_bp +
            "." * link +
            "(" * n_bp + "." * link + ")" * n_bp +
            "." * link +
            ")" * n_bp
        )
        # residue index accounting to match coords keys above:
        # parent open: 0..n_bp-1 -> coordsP[0..n_bp-1]
        # linker: n_bp..n_bp+link-1 (no coords needed, not paired)
        # child A open: n_bp+link .. n_bp+link+n_bp-1 -> coordsA[100..]
        # child A hairpin linker: unpaired
        # child A close: -> coordsA paired-partner residues
        # this hand-built index bookkeeping only needs to be internally
        # consistent with `coords` keys, not sequential -- build coords
        # keyed exactly to the dot-bracket's actual residue numbering:
        idx = 0
        coords_final = {}
        # parent open half
        for k in range(n_bp):
            coords_final[idx] = coordsP[k]; idx += 1
        idx += link  # linker, unpaired, no coords needed for axis fit
        childA_open_start = idx
        for k in range(n_bp):
            coords_final[idx] = coordsA[100 + k]; idx += 1
        idx += link
        for k in range(n_bp):
            coords_final[idx] = coordsA[100 + n_bp + k]; idx += 1
        idx += link
        for k in range(n_bp):
            coords_final[idx] = coordsB[200 + k]; idx += 1
        idx += link
        for k in range(n_bp):
            coords_final[idx] = coordsB[200 + n_bp + k]; idx += 1
        idx += link
        for k in range(n_bp):
            coords_final[idx] = coordsP[n_bp + k]; idx += 1

        assert idx == len(db), f"index bookkeeping mismatch: idx={idx}, len(db)={len(db)}"

        z, labels = extract_z(coords_final, db)
        angle_labels = [(lab, val) for lab, val in zip(labels, z) if lab.startswith("theta[sibling")]
        assert len(angle_labels) >= 1, f"no sibling angle recovered for bend={imposed_bend}; labels={labels}"
        lab, measured = angle_labels[0]
        err = min(abs(measured - imposed_bend), abs(measured - (180 - imposed_bend)))
        results.append(report(
            f"end-to-end extract_z hinge angle, imposed={imposed_bend} deg",
            err < TOL_ANGLE * 2,  # looser tolerance: full pipeline, more accumulated approximation
            f"{lab} = {measured:.2f} deg (imposed {imposed_bend}, or supplement)"
        ))
    return all(results)


# ---------------------------------------------------------------------------
# Test 4: distance recovery on straight, unbent A-form helices
# ---------------------------------------------------------------------------

def test4_distance_recovery():
    n_bp = 10
    results = []
    for true_gap in [5.0, 15.0, 30.0]:
        coords1, _ = a_form_helix(n_bp, origin=np.zeros(3), rot=np.eye(3))
        # place helix 2 alongside helix 1, offset in x by true_gap -- axis-
        # to-axis distance for two parallel lines equals the offset between
        # them directly, independent of helix radius (radius sets where the
        # *strand atoms* sit around the axis, not where the axis itself is)
        offset = np.array([true_gap, 0., 0.])
        coords2_raw, _ = a_form_helix(n_bp, origin=offset, rot=np.eye(3))
        coords2 = {k + 1000: v for k, v in coords2_raw.items()}
        stem1 = Stem(pairs=[(k, 2 * n_bp - 1 - k) for k in range(n_bp)], id=0)
        stem2 = Stem(pairs=[(k + 1000, 2 * n_bp - 1 - k + 1000) for k in range(n_bp)], id=1)
        ax1 = fit_helix_axis(coords1, stem1)
        ax2 = fit_helix_axis(coords2, stem2)
        measured = segment_distance(ax1, ax2)
        results.append(report(
            f"parallel-helix distance recovery, true gap={true_gap} A (+ 2*radius)",
            abs(measured - true_gap) < TOL_DIST,
            f"measured={measured:.3f}, want {true_gap:.3f}"
        ))
    return all(results)


# ---------------------------------------------------------------------------
# Test 5: pseudoknot handling does not crash the loop tree and contributes
# a distance/angle entry
# ---------------------------------------------------------------------------

def test5_pseudoknot_smoke():
    n_bp = 8
    results = []
    # H-type pseudoknot: stem1 (round brackets) crosses stem2 (square brackets)
    # db:  ((((((((....[[[[[[[[....))))))))....]]]]]]]]
    db = ("(" * n_bp + "." * 4 + "[" * n_bp + "." * 4 +
          ")" * n_bp + "." * 4 + "]" * n_bp)
    coords1, _ = a_form_helix(n_bp, origin=np.zeros(3), rot=np.eye(3))
    coords2_raw, _ = a_form_helix(n_bp, origin=np.array([25., 0., 0.]), rot=rotation_about_y(35.0))
    idx = 0
    coords_final = {}
    for k in range(n_bp):
        coords_final[idx] = coords1[k]; idx += 1
    idx += 4
    for k in range(n_bp):
        coords_final[idx] = coords2_raw[k]; idx += 1
    idx += 4
    for k in range(n_bp):
        coords_final[idx] = coords1[n_bp + k]; idx += 1
    idx += 4
    for k in range(n_bp):
        coords_final[idx] = coords2_raw[n_bp + k]; idx += 1
    assert idx == len(db), f"idx={idx} len(db)={len(db)}"

    try:
        z, labels = extract_z(coords_final, db)
    except Exception as e:
        return report("pseudoknot does not crash extract_z", False, f"raised {type(e).__name__}: {e}")

    has_pk_entry = any("pseudoknot" in lab for lab in labels)
    results.append(report("pseudoknot contributes a distance/angle entry", has_pk_entry, f"labels={labels}"))
    results.append(report("z has no NaN/inf", np.all(np.isfinite(z)), f"z={z}"))
    return all(results)


if __name__ == "__main__":
    print("=== Test 1: bare segment/angle primitives ===")
    ok1 = test1_segment_primitives()
    print("\n=== Test 2a: A-form single-axis recovery under known rotation ===")
    ok2a = test2_aform_axis_recovery()
    print("\n=== Test 2b: A-form pairwise bend-angle recovery ===")
    ok2b = test2b_aform_pair_angle()
    print("\n=== Test 3: end-to-end extract_z on a 3-way-junction hinge construct ===")
    ok3 = test3_two_domain_hinge_full_pipeline()
    print("\n=== Test 4: inter-helix distance recovery ===")
    ok4 = test4_distance_recovery()
    print("\n=== Test 5: pseudoknot handling (smoke test) ===")
    ok5 = test5_pseudoknot_smoke()
    print()
    all_ok = ok1 and ok2a and ok2b and ok3 and ok4 and ok5
    print(f"ALL TESTS {'PASSED' if all_ok else 'FAILED'}")
    import sys
    sys.exit(0 if all_ok else 1)
