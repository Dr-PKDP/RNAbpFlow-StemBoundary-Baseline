"""Restrained energy minimization of generated RNA samples (AMBER RNA.OL3, GBn2 implicit solvent, OpenMM).

Each Sample_*.pdb is cleaned (water, ions and hydrogens removed; residue and atom names standardised; the 5'
phosphate of the first residue of each chain removed, since the 5'-terminal residue templates carry none),
completed with PDBFixer when heavy atoms are missing, protonated, and minimized in two stages with harmonic
positional restraints on every heavy atom present in the input:

  stage 1  bonded terms, restraints and a soft bounded repulsion between non-bonded atoms closer than 2.5 A,
           so that bond lengths and angles are restored and overlapping atoms separated before the full
           non-bonded terms are evaluated
  stage 2  the full force field with GBn2 implicit solvent and the same restraints

The restraint constant is set with --k (kcal mol^-1 A^-2). The relaxed
heavy-atom structure is written to --out-root/<target>/<same file name>, in the input residue order, and
relax_log.json records, per sample, success, the C1' and heavy-atom RMSD from the input (no superposition), and
the stage-2 energies before and after minimization. A sample whose final potential energy is positive is
recorded as not relaxed and is not written.

Usage:
  python3 relax_samples.py --sample-root <Predictions> --out-root <Predictions_relaxed> [--targets ...] [--k 5]
  python3 relax_samples.py --selftest <a heavy-atom RNA PDB>
"""
from __future__ import annotations
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np

RES = {"A": "A", "C": "C", "G": "G", "U": "U", "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U",
       "RA": "A", "RC": "C", "RG": "G", "RU": "U", "RA5": "A", "RC5": "C", "RG5": "G", "RU5": "U",
       "RA3": "A", "RC3": "C", "RG3": "G", "RU3": "U", "A5": "A", "C5": "C", "G5": "G", "U5": "U",
       "A3": "A", "C3": "C", "G3": "G", "U3": "U"}
ATOM = {"O1P": "OP1", "O2P": "OP2", "O3P": "OP3", "C5M": "C7", "O1'": "O4'", "O2*": "O2'", "O3*": "O3'",
        "O4*": "O4'", "O5*": "O5'", "C1*": "C1'", "C2*": "C2'", "C3*": "C3'", "C4*": "C4'", "C5*": "C5'"}
FIVE_PRIME_PHOSPHATE = {"P", "OP1", "OP2", "OP3"}


def clean_pdb(text):
    """Heavy-atom RNA residues only, standard names, one chain id per chain, 5' phosphate removed."""
    rows, order = [], []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        rn = line[17:21].strip().upper()
        if rn not in RES:
            continue
        name = line[12:16].strip()
        el = line[76:78].strip().upper()
        if el == "H" or (not el and name.lstrip("0123456789").startswith("H")):
            continue
        name = ATOM.get(name, name)
        ch = line[21] if line[21].strip() else "A"
        key = (ch, line[22:27])
        if key not in order:
            order.append(key)
        rows.append((key, RES[rn], name, line[30:54]))
    first = {}
    for k in order:
        first.setdefault(k[0], k)
    out, serial, seen = [], 1, set()
    resnum = {k: i + 1 for i, k in enumerate(order)}
    for key, rn, name, xyz in rows:
        if key == first[key[0]] and name in FIVE_PRIME_PHOSPHATE:
            continue
        if (key, name) in seen:
            continue
        seen.add((key, name))
        el = name[0]
        out.append(f"ATOM  {serial:5d} {name:<4s} {rn:>3s} {key[0]}{resnum[key]:4d}    {xyz}  1.00  0.00          {el:>2s}")
        serial += 1
    out.append("END")
    return "\n".join(out) + "\n", order


def heavy_positions(topology, positions):
    """{(residue index, atom name): xyz in nm} for heavy atoms."""
    d = {}
    for a in topology.atoms():
        if a.element is not None and a.element.symbol != "H":
            p = positions[a.index]
            d[(a.residue.index, a.name)] = np.array([p.x, p.y, p.z])
    return d


def relax_one(pdb_text, k_kcal, tolerance, max_iter, platform_name, ff_files):
    import openmm as mm
    from openmm import app, unit
    clean, _ = clean_pdb(pdb_text)
    pdb = app.PDBFile(io.StringIO(clean))
    top, pos = pdb.topology, pdb.positions
    input_heavy = heavy_positions(top, pos.value_in_unit(unit.nanometer))
    try:
        from pdbfixer import PDBFixer
        fx = PDBFixer(pdbfile=io.StringIO(clean))
        fx.missingResidues = {}
        fx.findMissingAtoms()
        fx.missingTerminals = {}
        if fx.missingAtoms:
            fx.addMissingAtoms()
        top, pos = fx.topology, fx.positions
    except ImportError:
        pass
    ff = app.ForceField(*ff_files)
    mod = app.Modeller(top, pos)
    mod.addHydrogens(ff)
    top, pos = mod.topology, mod.positions

    bonds = [(b[0].index, b[1].index) for b in top.bonds()]

    def build(full):
        system = ff.createSystem(top, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.5 * unit.nanometer, constraints=None)
        if not full:
            for i in reversed(range(system.getNumForces())):
                f = system.getForce(i)
                if not isinstance(f, (mm.HarmonicBondForce, mm.HarmonicAngleForce, mm.PeriodicTorsionForce)):
                    system.removeForce(i)
            # soft, bounded repulsion between non-bonded atoms closer than 2.5 A, so that overlapping atoms
            # separate before the full Lennard-Jones term is evaluated
            soft = mm.CustomNonbondedForce("c*step(r0-r)*(r0-r)^2")
            soft.addGlobalParameter("c", 2.0e4)
            soft.addGlobalParameter("r0", 0.25)
            soft.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
            soft.setCutoffDistance(0.5)
            for _ in range(system.getNumParticles()):
                soft.addParticle([])
            soft.createExclusionsFromBonds(bonds, 3)
            system.addForce(soft)
        k = k_kcal * 4.184 * 100.0  # kJ mol^-1 nm^-2
        rest = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        rest.addGlobalParameter("k", k)
        for p in ("x0", "y0", "z0"):
            rest.addPerParticleParameter(p)
        ref = pos.value_in_unit(unit.nanometer)
        n = 0
        for a in top.atoms():
            if (a.residue.index, a.name) in input_heavy:
                rest.addParticle(a.index, list(ref[a.index]))
                n += 1
        system.addForce(rest)
        return system, n

    def context(system, positions):
        integ = mm.VerletIntegrator(0.001)
        plat = mm.Platform.getPlatformByName(platform_name) if platform_name else None
        props = {"Threads": "1"} if platform_name == "CPU" else {}
        ctx = mm.Context(system, integ, plat, props) if plat else mm.Context(system, integ)
        ctx.setPositions(positions)
        return ctx, integ

    s1, n_rest = build(False)
    c1, i1 = context(s1, pos)
    mm.LocalEnergyMinimizer.minimize(c1, tolerance, max_iter)
    pos1 = c1.getState(getPositions=True).getPositions()
    del c1, i1
    s2, _ = build(True)
    c2, i2 = context(s2, pos)
    e0 = c2.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    c2.setPositions(pos1)
    mm.LocalEnergyMinimizer.minimize(c2, tolerance, max_iter)
    st = c2.getState(getPositions=True, getEnergy=True)
    e1 = st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    final = st.getPositions()
    if not np.all(np.isfinite(np.array(final.value_in_unit(unit.nanometer)))) or not np.isfinite(e1):
        raise FloatingPointError("non-finite energy or coordinates after minimization")
    if e1 > 0:
        raise RuntimeError(f"minimization left a positive potential energy ({e1:.3g} kcal/mol)")
    out_heavy = heavy_positions(top, final.value_in_unit(unit.nanometer))
    common = [k for k in input_heavy if k in out_heavy]
    dev = np.array([np.linalg.norm(out_heavy[k] - input_heavy[k]) for k in common]) * 10.0
    c1dev = np.array([np.linalg.norm(out_heavy[k] - input_heavy[k]) for k in common if k[1] == "C1'"]) * 10.0
    # heavy-atom output in input residue order
    buf = io.StringIO()
    keep = [a for a in top.atoms() if a.element is not None and a.element.symbol != "H"]
    sub = app.Modeller(top, final)
    sub.delete([a for a in top.atoms() if a.element is not None and a.element.symbol == "H"])
    app.PDBFile.writeFile(sub.topology, sub.positions, buf, keepIds=True)
    return buf.getvalue(), {"ok": True, "n_restrained": n_rest, "n_heavy_out": len(keep),
                            "heavy_rmsd_A": float(np.sqrt(np.mean(dev ** 2))),
                            "c1_rmsd_A": float(np.sqrt(np.mean(c1dev ** 2))) if len(c1dev) else None,
                            "c1_max_shift_A": float(c1dev.max()) if len(c1dev) else None,
                            "energy_before_kcal": float(e0), "energy_after_kcal": float(e1)}


def pick_platform(requested):
    import openmm as mm
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    if requested:
        return requested
    for p in ("CUDA", "OpenCL", "CPU"):
        if p in names:
            return p
    return None


def _task(job):
    f, out_path, k, tol, it, platform, ff_files = job
    try:
        txt, rec = relax_one(Path(f).read_text(), k, tol, it, platform, ff_files)
        Path(out_path).write_text(txt)
    except Exception as exc:
        rec = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"sample": Path(f).name, **rec}


def run(args):
    platform = pick_platform(args.platform)
    print(f"OpenMM platform: {platform}; workers: {args.workers}", flush=True)
    ff_files = ["amber14-all.xml", "implicit/gbn2.xml"]
    targets = args.targets or sorted(p.name for p in args.sample_root.iterdir() if p.is_dir())
    log = {"k_kcal_per_mol_A2": args.k, "tolerance_kJ_per_mol_nm": args.tolerance, "max_iterations": args.max_iterations,
           "force_field": ff_files, "platform": platform, "targets": {}}
    from concurrent.futures import ProcessPoolExecutor
    for tid in targets:
        samples = sorted((args.sample_root / tid).glob("Sample_*.pdb"))
        od = args.out_root / tid
        od.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(od / f.name), args.k, args.tolerance, args.max_iterations, platform, ff_files) for f in samples]
        t0 = time.time()
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                recs = list(ex.map(_task, jobs))
        else:
            recs = [_task(j) for j in jobs]
        ok = [r for r in recs if r["ok"]]
        log["targets"][tid] = {"n_samples": len(samples), "n_relaxed": len(ok),
                               "median_c1_rmsd_A": float(np.median([r["c1_rmsd_A"] for r in ok])) if ok else None,
                               "max_c1_shift_A": float(max(r["c1_max_shift_A"] for r in ok)) if ok else None,
                               "median_heavy_rmsd_A": float(np.median([r["heavy_rmsd_A"] for r in ok])) if ok else None,
                               "errors": sorted({r["error"] for r in recs if not r["ok"]})[:5], "per_sample": recs}
        g = log["targets"][tid]
        print(f"{tid}: {g['n_relaxed']}/{g['n_samples']} relaxed in {time.time() - t0:.0f} s; median C1' RMSD "
              f"{g['median_c1_rmsd_A']}, max C1' shift {g['max_c1_shift_A']}; errors {g['errors']}", flush=True)
        args.log.write_text(json.dumps(log, indent=1))
    print(f"Wrote {args.log}")


def selftest(pdb_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from structure_checks import linkage_classes
    from sample_structure_qc import read_residues, count_clashes
    clean, order = clean_pdb(Path(pdb_path).read_text())
    rng = np.random.default_rng(3)
    lines = []
    for ln in clean.splitlines():
        if ln.startswith("ATOM"):
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]) + rng.normal(0, 0.35, 3)
            ln = ln[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + ln[54:]
        lines.append(ln)
    noisy = "\n".join(lines) + "\n"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p0 = Path(d) / "in.pdb"; p0.write_text(noisy)
        r0 = read_residues(p0)
        txt, rec = relax_one(noisy, 5.0, 10.0, 0, pick_platform(None), ["amber14-all.xml", "implicit/gbn2.xml"])
        p1 = Path(d) / "out.pdb"; p1.write_text(txt)
        r1 = read_residues(p1)
    lc0, lc1 = linkage_classes(r0), linkage_classes(r1)
    print("before:", lc0, "clashes", count_clashes(r0))
    print("after: ", lc1, "clashes", count_clashes(r1), rec)
    assert len(r1) == len(r0), (len(r1), len(r0))
    assert lc1["in_range"] == len(r1) - 1, lc1
    assert count_clashes(r1) == 0
    assert rec["c1_rmsd_A"] < 1.0
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-root", type=Path)
    ap.add_argument("--out-root", type=Path)
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--k", type=float, default=5.0, help="restraint constant, kcal mol^-1 A^-2 (default 5)")
    ap.add_argument("--tolerance", type=float, default=10.0, help="minimizer tolerance, kJ mol^-1 nm^-1")
    ap.add_argument("--max-iterations", type=int, default=5000, help="per stage; 0: until converged")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--platform", default=None)
    ap.add_argument("--log", type=Path, default=Path("relax_log.json"))
    ap.add_argument("--selftest", metavar="PDB")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.selftest)
    if not (a.sample_root and a.out_root):
        ap.error("--sample-root and --out-root are required")
    run(a)


if __name__ == "__main__":
    main()
