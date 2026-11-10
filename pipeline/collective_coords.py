"""
Collective-coordinate (z) extraction.

Turns (3D coordinates, secondary structure) into the whole-target vector
z = [d_1..d_p, theta_1..theta_q] of inter-helix-axis distances and angles.

Definitions:

  - Stems come from ANY base pair, including pseudoknot brackets.
  - The loop tree (hairpin / internal / multi-branch loops) is built from
    the nested, non-crossing pairs only.
  - Distance coordinates (d): one per structurally adjacent helix pair --
    parent/child in the loop tree, siblings under the same multi-branch
    loop, and any pseudoknot-forming pair.
  - Angle coordinates (theta): the bend angle at every one of those same
    adjacent pairs, plus one torsion (dihedral) angle for every
    multi-branch loop with 3+ children.
  - Helix axis direction is oriented 5'->3' along the first strand.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
for lo in "abcdefghijklmnopqrstuvwxyz":
    OPEN_TO_CLOSE[lo] = lo.upper()
CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}


def parse_dotbracket(structure: str):
    stacks = {op: [] for op in OPEN_TO_CLOSE}
    pairs = []
    for idx, ch in enumerate(structure):
        if ch == ".":
            continue
        if ch in OPEN_TO_CLOSE:
            stacks[ch].append(idx)
        elif ch in CLOSE_TO_OPEN:
            op = CLOSE_TO_OPEN[ch]
            if not stacks[op]:
                raise ValueError(f"unmatched '{ch}' at position {idx}")
            i = stacks[op].pop()
            pairs.append((i, idx))
    for op, stack in stacks.items():
        if stack:
            raise ValueError(f"unmatched '{op}' at position(s) {stack}")
    return sorted(pairs)


def nested_pairs_only(pairs):
    pairs_sorted = sorted(pairs, key=lambda p: (p[0], -p[1]))
    kept = []
    for i, j in pairs_sorted:
        crosses = any((i < ki < j < kj) or (ki < i < kj < j) for ki, kj in kept)
        if not crosses:
            kept.append((i, j))
    return sorted(kept)


@dataclass
class Stem:
    pairs: list
    id: int = -1
    is_pseudoknot: bool = False

    @property
    def i_range(self):
        return (self.pairs[0][0], self.pairs[-1][0])

    @property
    def j_range(self):
        return (self.pairs[-1][1], self.pairs[0][1])


def find_stems(pairs):
    pair_set = set(pairs)
    consumed = set()
    stems = []
    for (i, j) in sorted(pairs):
        if (i, j) in consumed:
            continue
        run = [(i, j)]
        consumed.add((i, j))
        ii, jj = i, j
        while (ii + 1, jj - 1) in pair_set and (ii + 1, jj - 1) not in consumed:
            ii, jj = ii + 1, jj - 1
            run.append((ii, jj))
            consumed.add((ii, jj))
        stems.append(Stem(pairs=run))
    for k, s in enumerate(stems):
        s.id = k
    return stems


def mark_pseudoknot_stems(stems, nested_pair_set):
    for s in stems:
        s.is_pseudoknot = not any(p in nested_pair_set for p in s.pairs)


@dataclass
class LoopNode:
    parent_stem: 'Stem | None'
    child_stems: list = field(default_factory=list)


def build_loop_tree(stems, nested_pairs, seq_len):
    nested_set = set(nested_pairs)
    nested_stems = [s for s in stems if all(p in nested_set for p in s.pairs)]

    def contains(outer, inner):
        oi, oj = outer.pairs[0]
        ii, ij = inner.pairs[0]
        return oi < ii and ij < oj

    def is_immediate_child(outer, inner, all_stems):
        if not contains(outer, inner):
            return False
        for mid in all_stems:
            if mid is outer or mid is inner:
                continue
            if contains(outer, mid) and contains(mid, inner):
                return False
        return True

    loops = []
    top_level = [s for s in nested_stems
                 if not any(contains(o, s) for o in nested_stems if o is not s)]
    loops.append(LoopNode(parent_stem=None, child_stems=top_level))
    for outer in nested_stems:
        children = [s for s in nested_stems if is_immediate_child(outer, s, nested_stems)]
        loops.append(LoopNode(parent_stem=outer, child_stems=children))
    return loops


@dataclass
class Axis:
    stem_id: int
    point: np.ndarray
    direction: np.ndarray
    start: np.ndarray
    end: np.ndarray


def fit_helix_axis(coords: dict, stem: Stem):
    if len(stem.pairs) < 2:
        return None
    midpoints = []
    for (i, j) in stem.pairs:
        if i not in coords or j not in coords:
            return None
        midpoints.append((coords[i] + coords[j]) / 2.0)
    midpoints = np.array(midpoints)
    centroid = midpoints.mean(axis=0)
    centered = midpoints - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if np.dot(midpoints[-1] - midpoints[0], direction) < 0:
        direction = -direction
    direction = direction / np.linalg.norm(direction)
    proj_first = centroid + np.dot(midpoints[0] - centroid, direction) * direction
    proj_last = centroid + np.dot(midpoints[-1] - centroid, direction) * direction
    return Axis(stem_id=stem.id, point=centroid, direction=direction,
                start=proj_first, end=proj_last)


def segment_distance(a: Axis, b: Axis) -> float:
    p1, q1, p2, q2 = a.start, a.end, b.start, b.end
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    A, E = np.dot(d1, d1), np.dot(d2, d2)
    F = np.dot(d2, r)
    if A <= 1e-12 and E <= 1e-12:
        return float(np.linalg.norm(p1 - p2))
    if A <= 1e-12:
        t = np.clip(F / E, 0.0, 1.0)
        s = 0.0
    else:
        C = np.dot(d1, r)
        if E <= 1e-12:
            s = np.clip(-C / A, 0.0, 1.0)
            t = 0.0
        else:
            B = np.dot(d1, d2)
            denom = A * E - B * B
            s = np.clip((B * F - C * E) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
            t = (B * s + F) / E
            if t < 0.0:
                t = 0.0
                s = np.clip(-C / A, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((B - C) / A, 0.0, 1.0)
    closest1 = p1 + s * d1
    closest2 = p2 + t * d2
    return float(np.linalg.norm(closest1 - closest2))


def axis_angle(a: Axis, b: Axis) -> float:
    cos_t = np.clip(np.dot(a.direction, b.direction), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def junction_torsion(axes_at_loop: list, loop_center: np.ndarray):
    if len(axes_at_loop) < 3:
        return None
    pts = np.array([ax.point for ax in axes_at_loop])
    hub = pts.mean(axis=0)
    b0 = axes_at_loop[0].point - hub
    b1 = axes_at_loop[1].point - hub
    axis_vec = hub - loop_center
    n = np.linalg.norm(axis_vec)
    if n < 1e-9:
        axis_vec = np.cross(b0, b1)
        n = np.linalg.norm(axis_vec)
        if n < 1e-9:
            return None
    axis_vec = axis_vec / n
    b0p = b0 - np.dot(b0, axis_vec) * axis_vec
    b1p = b1 - np.dot(b1, axis_vec) * axis_vec
    if np.linalg.norm(b0p) < 1e-9 or np.linalg.norm(b1p) < 1e-9:
        return None
    b0p /= np.linalg.norm(b0p)
    b1p /= np.linalg.norm(b1p)
    cos_t = np.clip(np.dot(b0p, b1p), -1.0, 1.0)
    sin_t = np.dot(np.cross(b0p, b1p), axis_vec)
    return float(np.degrees(np.arctan2(sin_t, cos_t)))


def extract_z(coords: dict, dotbracket: str):
    pairs = parse_dotbracket(dotbracket)
    nested = nested_pairs_only(pairs)
    nested_set = set(nested)
    stems = find_stems(pairs)
    mark_pseudoknot_stems(stems, nested_set)
    axes = {}
    for s in stems:
        ax = fit_helix_axis(coords, s)
        if ax is not None:
            axes[s.id] = ax

    loops = build_loop_tree(stems, nested, len(dotbracket))

    d_vals, d_labels = [], []
    theta_vals, theta_labels = [], []
    seen_pairs = set()

    def add_distance_angle(sa, sb, tag):
        key = tuple(sorted((sa.id, sb.id)))
        if key in seen_pairs:
            return
        if sa.id not in axes or sb.id not in axes:
            return
        seen_pairs.add(key)
        ax_a, ax_b = axes[sa.id], axes[sb.id]
        d_vals.append(segment_distance(ax_a, ax_b))
        d_labels.append(f"d[{tag}:stem{sa.id}-stem{sb.id}]")
        theta_vals.append(axis_angle(ax_a, ax_b))
        theta_labels.append(f"theta[{tag}:stem{sa.id}-stem{sb.id}]")

    for loop in loops:
        members = ([loop.parent_stem] if loop.parent_stem else []) + loop.child_stems
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                sa, sb = members[a], members[b]
                if loop.parent_stem is not None and (sa is loop.parent_stem or sb is loop.parent_stem):
                    tag = "parent-child"
                else:
                    tag = "sibling"
                add_distance_angle(sa, sb, tag)
        if len(loop.child_stems) >= 3:
            branch_axes = [axes[s.id] for s in loop.child_stems if s.id in axes]
            if len(branch_axes) >= 3:
                loop_center = (loop.parent_stem and axes.get(loop.parent_stem.id) and
                               axes[loop.parent_stem.id].point)
                if loop_center is None:
                    loop_center = np.mean([ax.point for ax in branch_axes], axis=0)
                tors = junction_torsion(branch_axes, loop_center)
                if tors is not None:
                    theta_vals.append(tors)
                    theta_labels.append(
                        f"torsion[loop@stem{loop.parent_stem.id if loop.parent_stem else 'ext'}]")

    def stems_cross(sa: Stem, sb: Stem) -> bool:
        for (i, j) in sa.pairs:
            for (ki, kj) in sb.pairs:
                if (i < ki < j < kj) or (ki < i < kj < j):
                    return True
        return False

    pk_stems = [s for s in stems if s.is_pseudoknot]
    for pk in pk_stems:
        for other in stems:
            if other is pk:
                continue
            if stems_cross(pk, other):
                add_distance_angle(pk, other, "pseudoknot")

    z = np.array(d_vals + theta_vals, dtype=float)
    labels = d_labels + theta_labels
    return z, labels
