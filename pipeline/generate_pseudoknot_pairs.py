"""Extract the base pairs that cross a target's nested consensus structure, using IPknot.

IPknot's nested pairs are not used; only the crossing pairs are kept, because the MXfold2/SimFold/CONTRAfold consensus
cannot represent them.

Usage:
    python generate_pseudoknot_pairs.py <target_id> <sequence> <output_dir> <ipknot_binary_path>

Writes <output_dir>/pseudoknot_pairs.json -- a list of [i, j] 0-indexed
pairs, or an empty list if IPknot found no crossing pairs for this
sequence (a real, expected outcome for ligand-induced pseudoknots like
2MIY's, not a failure of this script).
"""
import json
import subprocess
import sys
from pathlib import Path


def parse_ipknot_dotbracket(structure: str):
    """Returns (nested_pairs, crossing_pairs) as lists of (i, j) 0-indexed
    tuples. IPknot marks crossing pairs with non-'()' bracket characters
    ([], {}, etc.) by design -- that convention is what this relies on,
    not an assumption this script invents."""
    open_map = {"(": ")", "[": "]", "{": "}", "<": ">"}
    close_map = {v: k for k, v in open_map.items()}
    stacks = {op: [] for op in open_map}
    nested, crossing = [], []
    for idx, ch in enumerate(structure):
        if ch == ".":
            continue
        if ch in open_map:
            stacks[ch].append(idx)
        elif ch in close_map:
            op = close_map[ch]
            if not stacks[op]:
                raise ValueError(f"unmatched '{ch}' at position {idx} in IPknot output")
            i = stacks[op].pop()
            (nested if op == "(" else crossing).append((i, idx))
    for op, stack in stacks.items():
        if stack:
            raise ValueError(f"unmatched '{op}' in IPknot output, position(s) {stack}")
    return sorted(nested), sorted(crossing)


def run_ipknot(ipknot_bin: Path, sequence: str, target_id: str, workdir: Path) -> str:
    fasta_path = workdir / f"{target_id}_ipknot_input.fa"
    fasta_path.write_text(f">{target_id}\n{sequence}\n")
    result = subprocess.run(
        [str(ipknot_bin), str(fasta_path)],
        capture_output=True, text=True, check=True,
    )
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    # output format: >header, sequence, dot-bracket -- the third non-empty line
    if len(lines) < 3:
        raise RuntimeError(f"unexpected IPknot output for {target_id}:\n{result.stdout}")
    return lines[2].strip()


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    target_id, sequence, output_dir, ipknot_bin = sys.argv[1:5]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ipknot_bin = Path(ipknot_bin)

    db = run_ipknot(ipknot_bin, sequence, target_id, output_dir)
    nested, crossing = parse_ipknot_dotbracket(db)

    print(f"{target_id}: IPknot dot-bracket: {db}")
    print(f"  {len(nested)} nested pairs, {len(crossing)} crossing pairs")
    if not crossing:
        print(f"  No pseudoknot found for {target_id} -- either this sequence "
              f"has no ligand-independent crossing structure (a real, expected "
              f"result, e.g. a ligand-induced pseudoknot), or a different "
              f"threshold/model needs trying manually. Not treated as an error.")

    out_path = output_dir / "pseudoknot_pairs.json"
    out_path.write_text(json.dumps({
        "target_id": target_id,
        "ipknot_dotbracket": db,
        "crossing_pairs": crossing,
    }, indent=2))
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
