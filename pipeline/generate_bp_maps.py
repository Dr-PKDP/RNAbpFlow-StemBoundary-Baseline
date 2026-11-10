"""Generate the three base-pair maps (map1.npy, map2.npy, map3.npy) that RNAbpFlow takes as input.

Maps come from MXfold2, SimFold and CONTRAfold; CONTRAfold is used in place of ContextFold.
Tool locations are read from the environment variable BPMAPS_ROOT (default: ./bpmaps).

Usage:
    python generate_bp_maps.py <target_id> <sequence> <output_dir>
"""
import os
import subprocess
import sys
import re
from pathlib import Path

import numpy as np

# Tool installation paths
BPMAPS_ROOT = Path(os.environ.get("BPMAPS_ROOT", "bpmaps"))
MXFOLD2_VENV_PYTHON = BPMAPS_ROOT / "venv" / "bin" / "python"
MXFOLD2_BIN = BPMAPS_ROOT / "venv" / "bin" / "mxfold2"
SIMFOLD_BIN = BPMAPS_ROOT / "MultiRNAFold" / "simfold"
CONTRAFOLD_BIN = BPMAPS_ROOT / "contrafold_env" / "shared" / "contrafold" / "contrafold"


def dotbracket_to_matrix(structure: str) -> np.ndarray:
    """Parse dot-bracket notation into an LxL symmetric binary matrix."""
    L = len(structure)
    matrix = np.zeros((L, L), dtype=np.float32)
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if not stack:
                raise ValueError(f"Unmatched ')' at position {i} in structure: {structure}")
            j = stack.pop()
            matrix[i, j] = 1.0
            matrix[j, i] = 1.0
        elif char == '.':
            continue
        else:
            raise ValueError(f"Unexpected character '{char}' at position {i}: '{char}'")
    if stack:
        raise ValueError(f"Unmatched '(' at position(s) {stack}")
    return matrix


def run_mxfold2(target_id: str, sequence: str, workdir: Path) -> str:
    """Returns the dot-bracket structure string. MXfold2 output format,
    confirmed output format:
        SEQUENCE
        STRUCTURE (ENERGY)
    (no header line echoed back)."""
    fasta_path = workdir / f"{target_id}_mxfold2_input.fa"
    fasta_path.write_text(f">{target_id}\n{sequence}\n")
    result = subprocess.run(
        [str(MXFOLD2_BIN), "predict", str(fasta_path)],
        capture_output=True, text=True, check=True,
    )
    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    struct_line = lines[-1]
    structure = struct_line.split()[0]
    if len(structure) != len(sequence):
        raise ValueError(
            f"MXfold2 structure length {len(structure)} != sequence length {len(sequence)}. "
            f"Raw output:\n{result.stdout}"
        )
    return structure
def run_simfold(target_id: str, sequence: str, workdir: Path) -> str:
    """Returns the dot-bracket structure string. SimFold output format,
    confirmed output format:
        Seq: SEQUENCE
        MFE: STRUCTURE  ENERGY
    Run from the MultiRNAFold directory since it expects params/ as a
    relative path from its own working directory."""
    result = subprocess.run(
        [str(SIMFOLD_BIN), "-s", sequence],
        capture_output=True, text=True, check=True,
        cwd=str(SIMFOLD_BIN.parent),
    )
    mfe_line = next(l for l in result.stdout.split("\n") if l.startswith("MFE:"))
    # "MFE: <structure>  <energy>"
    structure = mfe_line[len("MFE:"):].strip().split()[0]
    if len(structure) != len(sequence):
        raise ValueError(
            f"SimFold structure length {len(structure)} != sequence length {len(sequence)}. "
            f"Raw output:\n{result.stdout}"
        )
    return structure


def run_contrafold(target_id: str, sequence: str, workdir: Path) -> str:
    """Returns the dot-bracket structure string. CONTRAfold output format,
    confirmed output format:
        >target_id
        SEQUENCE
        >structure
        STRUCTURE
    Requires a FASTA-style '>' header; a non-'>' header character
    produces an error."""
    seq_path = workdir / f"{target_id}_contrafold_input.seq"
    seq_path.write_text(f">{target_id}\n{sequence}\n")
    result = subprocess.run(
        [str(CONTRAFOLD_BIN), "predict", str(seq_path)],
        capture_output=True, text=True, check=True,
    )
    lines = result.stdout.strip().split("\n")
    # structure is the line immediately after the ">structure" marker
    idx = next(i for i, l in enumerate(lines) if l.strip() == ">structure")
    structure = lines[idx + 1].strip()
    if len(structure) != len(sequence):
        raise ValueError(
            f"CONTRAfold structure length {len(structure)} != sequence length {len(sequence)}. "
            f"Raw output:\n{result.stdout}"
        )
    return structure
def generate_maps(target_id: str, sequence: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = sequence.strip().upper().replace("T", "U")  # RNA convention

    predictors = [
        ("MXfold2", run_mxfold2),
        ("SimFold", run_simfold),
        ("CONTRAfold", run_contrafold),
    ]

    structures = {}
    for name, fn in predictors:
        print(f"Running {name}...")
        structure = fn(target_id, sequence, output_dir)
        structures[name] = structure
        print(f"  {name}: {structure}")

    for i, (name, structure) in enumerate(structures.items(), start=1):
        matrix = dotbracket_to_matrix(structure)
        out_path = output_dir / f"map{i}.npy"
        np.save(out_path, matrix)
        print(f"Wrote {out_path} ({name}, {int(matrix.sum() / 2)} base pairs, shape {matrix.shape})")

    # Write the FASTA file RNAbpFlow itself expects, matching the format
    # confirmed from the shipped example targets (single header + sequence)
    fasta_path = output_dir / f"{target_id}.fasta"
    fasta_path.write_text(f">{target_id}\n{sequence}\n")
    print(f"Wrote {fasta_path}")

    # Sanity check: agreement between predictors is informative, not
    # required -- print it, don't enforce it. Real methodological
    # diversity across three different modeling paradigms means some
    # disagreement is expected and desired (see plan Sec. 5.5's later use
    # of this same idea for attribution).
    n_common = sum(
        1 for i in range(len(sequence))
        for j in range(i + 1, len(sequence))
        if all(dotbracket_to_matrix(s)[i, j] == 1 for s in structures.values())
    )
    print(f"\nBase pairs agreed upon by all three predictors: {n_common}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    target_id, sequence, output_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    generate_maps(target_id, sequence, output_dir)
