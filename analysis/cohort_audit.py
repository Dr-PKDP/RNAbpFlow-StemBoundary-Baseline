"""Training-exposure audit of target structures against the RNA3DB 2024-04-26 split.

Criteria (per target chain):
  (a) exact deposition: any chain of the PDB entry is a training or test chain.
  (b) sequence: the target sequence equals, or is contained in, a training chain.
  (c) family: Rfam family of the target chain (best significant Infernal hit, inc '!')
      and the number of training chains with a significant hit to the same family.
  (d) sub-threshold homologs: training chains whose hit to the same family has inc '?'
      (E-value above the Rfam gathering threshold but within the cmscan reporting limit).

Inputs: rna3db_2024-04-26_chains.tsv (chain, split and sequence columns are detected
from the values) and rna3db_2024-04-26_full_cmscan.tbl (Infernal --tblout).

  python cohort_audit.py --chains-tsv CHAINS --cmscan TBL --cohort-csv ../data/cohort/cohort_audit.csv --recheck
  python cohort_audit.py --chains-tsv CHAINS --cmscan TBL --target 1P5M:A:GGCUGU... --out audit_1P5M.csv
  python cohort_audit.py --selftest
"""
import argparse, csv, re, sys, collections

CHAIN_RE = re.compile(r"^[0-9][0-9a-z]{3}_\w+$", re.I)
SEQ_RE = re.compile(r"^[ACGUTNacgutn]{4,}$")


def read_cmscan(path):
    hits = collections.defaultdict(list)          # chain -> [(family, name, evalue, inc)]
    fam_chains = collections.defaultdict(lambda: {"!": set(), "?": set()})
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split(None, 17)
            fam, name, chain, ev, inc = f[1], f[0], f[2], float(f[15]), f[16]
            hits[chain].append((fam, name, ev, inc))
            if inc in ("!", "?"):
                fam_chains[fam][inc].add(chain)
    return hits, fam_chains


def read_chains(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    header, body = rows[0], rows[1:]
    ncol = len(header)
    col = {"chain": None, "split": None, "seq": None}
    for j in range(ncol):
        vals = [r[j] for r in body[:200] if len(r) > j]
        if not vals:
            continue
        if col["chain"] is None and sum(bool(CHAIN_RE.match(v)) for v in vals) > 0.9 * len(vals):
            col["chain"] = j
        elif col["split"] is None and {v.lower() for v in vals} <= {"train", "test", "train_set", "test_set", "valid", "valid_set"}:
            col["split"] = j
        elif col["seq"] is None and sum(bool(SEQ_RE.match(v)) for v in vals) > 0.9 * len(vals):
            col["seq"] = j
    if None in col.values():
        sys.exit(f"could not detect columns in {path}: header={header} detected={col}")
    chains = {}
    for r in body:
        split = "train" if r[col["split"]].lower().startswith("train") else "test"
        chains[r[col["chain"]]] = (split, r[col["seq"]].upper().replace("T", "U"))
    return chains


def audit(pdb, chain, seq, chains, hits, fam_chains, family=None):
    pdb_l = pdb.lower()
    entry = {c: v[0] for c, v in chains.items() if c.split("_")[0].lower() == pdb_l}
    status = "absent from RNA3DB" if not entry else "train split" if "train" in entry.values() else "test split only"
    seq = seq.upper().replace("T", "U")
    train = {c: s for c, (sp, s) in chains.items() if sp == "train"}
    exact = sorted(c for c, s in train.items() if s == seq)
    embedded = sorted(c for c, s in train.items() if seq in s and s != seq)
    own = sorted((h for h in hits.get(f"{pdb_l}_{chain}", []) if h[3] == "!"), key=lambda h: h[2])
    if family is None and own:
        family = own[0][0]
    name = next((h[1] for h in own if h[0] == family), "")
    tr = sorted(c for c in fam_chains[family]["!"] if c in train) if family else []
    te = sorted(c for c in fam_chains[family]["!"] if chains.get(c, ("",))[0] == "test") if family else []
    sub = sorted(c for c in fam_chains[family]["?"] if c in train) if family else []
    return {"pdb_id": pdb.upper(), "chain": chain, "rna3db_2024-04-26_status": status,
            "train_chains_identical_sequence": ";".join(exact) or "none",
            "train_chains_containing_sequence": ";".join(embedded) or "none",
            "rfam_family": f"{name} ({family})" if family else "no significant hit",
            "best_hit_evalue": own[0][2] if own else "",
            "training_chains_same_family": len(tr), "training_chains_subthreshold_same_family": len(sub),
            "test_chains_same_family": len(te), "training_chains_same_family_ids": ";".join(tr),
            "training_chains_subthreshold_ids": ";".join(sub)}


def recheck(cohort_csv, chains, fam_chains):
    bad = 0
    with open(cohort_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            m = re.search(r"RF\d{5}", r["rfam_family"])
            n = len([c for c in fam_chains[m.group(0)]["!"] if chains.get(c, ("",))[0] == "train"])
            status = "absent from RNA3DB" if not any(c.split("_")[0].lower() == r["pdb_id"].lower() for c in chains) else "in RNA3DB"
            sub = len([c for c in fam_chains[m.group(0)]["?"] if chains.get(c, ("",))[0] == "train"])
            ok = n == int(r["training_chains_same_family"])
            bad += not ok
            print(f"{r['pdb_id']:5s} {m.group(0)}  recorded={r['training_chains_same_family']:>3s}  recomputed={n:>3d}  subthreshold={sub:>3d}  {'ok' if ok else 'MISMATCH'}   status: {r['rna3db_2024-04-26_status']} / {status}")
    print("mismatches:", bad)


def selftest():
    import tempfile, os
    d = tempfile.mkdtemp()
    tsv, tbl = os.path.join(d, "c.tsv"), os.path.join(d, "t.tbl")
    open(tsv, "w").write("chain_id\tsplit\tsequence\n1aaa_A\ttrain_set\tGGGAAACCC\n1bbb_B\ttrain_set\tUUGGGAAACCCUU\n2ccc_A\ttest_set\tGGGAAACCC\n")
    line = "{n:20s} {a:9s} {q:20s} -          cm        1       10        1       10      +    no    1 0.50   0.0   30.0   {e:>9s} {i} desc\n"
    open(tbl, "w").write("#hdr\n" + line.format(n="Fam", a="RF99999", q="1aaa_A", e="1e-9", i="!") + line.format(n="Fam", a="RF99999", q="1bbb_B", e="2.0", i="?")
                         + line.format(n="Fam", a="RF99999", q="2ccc_A", e="1e-9", i="!") + line.format(n="Fam", a="RF99999", q="9zzz_A", e="1e-9", i="!"))
    chains, (hits, fam) = read_chains(tsv), read_cmscan(tbl)
    r = audit("9ZZZ", "A", "GGGAAACCC", chains, hits, fam)
    assert r["rna3db_2024-04-26_status"] == "absent from RNA3DB"
    assert r["train_chains_identical_sequence"] == "1aaa_A" and r["train_chains_containing_sequence"] == "1bbb_B"
    assert r["rfam_family"] == "Fam (RF99999)" and r["training_chains_same_family"] == 1
    assert r["training_chains_subthreshold_same_family"] == 1 and r["test_chains_same_family"] == 1
    assert audit("2CCC", "A", "GGGAAACCC", chains, hits, fam)["rna3db_2024-04-26_status"] == "test split only"
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains-tsv"); ap.add_argument("--cmscan")
    ap.add_argument("--target", action="append", default=[], help="PDBID:CHAIN:SEQUENCE[:RFxxxxx]")
    ap.add_argument("--cohort-csv"); ap.add_argument("--recheck", action="store_true")
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit()
    chains = read_chains(a.chains_tsv)
    print(f"{sum(v[0] == 'train' for v in chains.values())} training and {sum(v[0] == 'test' for v in chains.values())} test chains read")
    hits, fam = read_cmscan(a.cmscan)
    if a.recheck:
        recheck(a.cohort_csv, chains, fam)
    rows = []
    for t in a.target:
        p = t.split(":")
        rows.append(audit(p[0], p[1], p[2], chains, hits, fam, p[3] if len(p) > 3 else None))
    for r in rows:
        print("\n".join(f"  {k}: {v}" for k, v in r.items()), "\n")
    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
