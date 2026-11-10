# Stem-boundary perturbation baseline for RNAbpFlow structure ensembles

Code, derived data and the audited target list for *A Stem-Boundary Perturbation Baseline for Evaluating Inter-Helix Variation in RNAbpFlow Structure Ensembles* (Pramanik and Zhao).

The study asks whether the variance of inter-helix collective coordinates (distances, angles, junction torsions) in
RNAbpFlow ensembles exceeds the variance produced by a bracket-type-preserving, stem-corresponding perturbation of the base-pair map, and
whether coordinates that exceed this baseline also vary in experimental structures of the same RNA.

## Layout

```
pipeline/
  generate_bp_maps.py            MXfold2, SimFold and CONTRAfold base-pair maps (map1-3.npy) for one target
  generate_pseudoknot_pairs.py   IPknot crossing pairs for one target
  run_inference.py               RNAbpFlow sampling with dropout off
  configs/inference.yaml         sampler configuration (default 50 samples and 50 timesteps; a count in list.txt overrides the sample count)
  collective_coords.py           helix axes and the collective-coordinate vector z
  clearing.py                    consensus structure, perturbation baseline (noise_floor), clearing ratios
analysis/
  compute_statistics.py          target-level and pooled angle-versus-distance statistics
  displacement_direction_test.py apo/holo displacement-direction test
  displacement_robustness.py     scaling, typicality, jackknife, split-half and offset checks for that test
  baseline_stability.py          repeats the baseline of one target over independent perturbation seeds
  perturbation_resampling.py     stability of the baseline variance under subsampling of accepted perturbations
  cohort_audit.py                RNA3DB training-exposure audit (split membership, sequence containment, Rfam family counts)
  family_exposure_check.py       Spearman association between Rfam-family training exposure and clearing across targets
  experimental_variation_concordance.py   generated versus experimental variance (--centre mean|median)
  perturbation_label_check.py    whether accepted perturbations keep the stem-pair labels and stem identity of the unperturbed vector
  target_multimodality_diagnostic.py      cluster structure of one target's ensemble
  sample_structure_qc.py         fixed structural-quality criteria, and clearing recomputed on the samples meeting them
  structure_checks.py            O3'-P linkage classes, clashes, and two base-pairing measures per sample
  sample_geometry_descriptives.py   O3'-P and consecutive C1'-C1' distance distributions, with clearing on two graded subsets
  subset_selection_control.py    clearing on the fewest- and most-chain-break subsets against random subsets of equal size
  baseline_substrate_sensitivity.py   the baseline re-estimated from several generated samples as the coordinate substrate
  relax_samples.py               restrained minimization of every sample (OpenMM, AMBER RNA.OL3, GBn2)
  ensemble_analyses.py           clearing, target-level statistics, displacement criteria and concordance for one ensemble
  config/                        target lists for the displacement, concordance and apo/holo-pair analyses
data/
  clearing_signal_ratios.csv     signal-to-baseline ratios of the 24 coordinate pairs (8 targets) behind Tables 5 and 6
  cohort/cohort_audit.csv        structures of the cohort targets with RNA3DB status, Rfam family and training-chain counts
  cohort/target_sequences.csv    model-input sequence of each of the 11 generated targets, by prediction directory name
tests/validate_collective_coords.py   synthetic-geometry validation of the coordinate extraction
environment/                     conda environments: rnabpflow.yml, analysis.yml, ipknot.yml
```

## Environments

| Environment | Used for |
|---|---|
| `rnabpflow` | `run_inference.py` (RNAbpFlow source and checkpoint from its own repository) |
| `ipknot` | `generate_pseudoknot_pairs.py` (IPknot binary) |
| `analysis` | everything in `analysis/`, `pipeline/clearing.py`, `pipeline/collective_coords.py`, `tests/` |

`generate_bp_maps.py` reads the locations of the MXfold2, SimFold and CONTRAfold installations from `BPMAPS_ROOT`.
`relax_samples.py` additionally needs `openmm` and `pdbfixer`.

## Target identifiers

`data/cohort/cohort_audit.csv` and the config files use cohort identifiers; the manuscript uses PDB ids.

| Identifier | Structures |
|---|---|
| `T01` | 5E54 (apo), 4TZX (holo) |
| `T02`, `T03`, `T04`, `T05` | 2L1V, 2MIY, 3Q50, 4RZD |
| `T06` | 8THV |
| `T07` | 8F4O (apo), 2GDI (holo) |
| `T09` | 5DDO (apo), 5DDP (holo) |
| `T10` | 7KD1 (apo), 4LVV (holo) |
| `T11` | 5KH8 |
| `T15`, `T17`, `T19` | 3IRW, 8FW4, 6OD9 |
| `T20` | 1P5M |

Five generated targets keep a lowercase cohort identifier in the prediction directories and the config files: `t02`
is 2L1V, `t04` is 3Q50, `t06` is 8THV, `t07` is 8F4O and `t09` is 5DDO. Of the remaining six directories, `hcv_iia`
holds 1P5M and each other is named by the PDB id of its target. The directory `1y26` holds the samples of the
backbone comparison reported in the supplement and is not part of the cohort. `data/cohort/target_sequences.csv` lists the
model-input sequence of each of the 11 generated targets under its directory name and PDB id.

## Reference structures

Experimental coordinates are not redistributed. Download the following entries from the RCSB PDB as PDB-format files
into `analysis/refs/`, named `<PDB id>.pdb`: 5E54, 4TZX, 8F4O, 2GDI, 7KD1, 4LVV, 2MIY, 4RZD, 5KH8, 1P5M, 2L1V, 8THV.

## Reproducing the analysis

Per target `<id>` (directory name) with sequence `<seq>`, both taken from `data/cohort/target_sequences.csv`:

```bash
cd pipeline
python generate_bp_maps.py <id> <seq> ../data/Inputs/<id>
python generate_pseudoknot_pairs.py <id> <seq> ../data/Inputs/<id> <ipknot-binary>
# list <id> and its sample count in data/Inputs/list.txt ("<id> 100"; the analysed ensembles have 100 samples per
# target and the count in list.txt overrides samples_per_sequence in the YAML), then, from the RNAbpFlow source directory:
python run_inference.py --config configs/inference.yaml --output-dir <path>/Predictions
python clearing.py --targets <id> --sample-root ../data/Predictions --bpmaps-root ../data/Inputs \
    --n-perturbations 300 --out <id>_report.json
```

Experimental comparisons, run from `analysis/`:

```bash
python displacement_direction_test.py --config config/displacement.json \
    --sample-root ../data/Predictions --bpmaps-root ../data/Inputs --out displacement_report.json
python displacement_robustness.py --config config/displacement.json \
    --sample-root ../data/Predictions --bpmaps-root ../data/Inputs --out displacement_robustness.json
python experimental_variation_concordance.py --config config/concordance.json \
    --sample-root ../data/Predictions --bpmaps-root ../data/Inputs --min-mapped 0.85 \
    --out experimental_variation_concordance.json
python baseline_stability.py --config config/displacement.json --target 7kd1 \
    --sample-root ../data/Predictions --bpmaps-root ../data/Inputs
python perturbation_resampling.py --targets t07 7kd1 \
    --sample-root ../data/Predictions --bpmaps-root ../data/Inputs
python perturbation_label_check.py --targets t07 7kd1 --sample-root ../data/Predictions \
    --bpmaps-root ../data/Inputs --out perturbation_label_check.json
```

Baselines in the manuscript use 300 attempted perturbations per target (`--n-perturbations 300`). With the
stem-correspondence rule 7KD1 has 6 accepted perturbations at 300 attempts and 41 at 3,000, and 5DDO has none at
either budget; the 3,000-attempt runs (`--n-perturbations 3000`) are the sensitivity analysis reported in the
manuscript.

## Baseline acceptance rule

The perturbation baseline accepts a perturbed base-pair map only when its coordinate vector has the labels of the
unperturbed vector and every stem keeps its identity (same bracket class, outer and inner pair within one position of
the original). `--length-only-baseline` (in `clearing.py` and `experimental_variation_concordance.py`) restores the
length-only (equal-length) rule.

## Structural quality, relaxation and ensemble analyses

`sample_structure_qc.py` reports the structural quality of every generated sample (chain breaks from the O3'-P
distance, heavy-atom clashes below 2.2 A, and the fraction of the conditioning map's base pairs whose Watson-Crick
nitrogens are within 3.5 A) and recomputes the clearing calls on the samples that meet the fixed criteria, so a call
that depends on malformed samples appears as a changed call. `structure_checks.py` classifies every O3'-P linkage as
compressed (< 1.2 A), in range, stretched (2.0-3.0 A) or extended (> 3.0 A), and reports clashes, the Watson-Crick
N1-N3 proximity measure and a hydrogen-bond pairing measure (two canonical or wobble donor-acceptor distances within
3.5 A) for the conditioning base pairs.

`sample_geometry_descriptives.py` reports the distribution of the O3'-P and consecutive C1'-C1' distances of the
generated samples and recomputes the clearing calls on two graded subsets (the samples with the fewest chain breaks,
and the samples with a continuous C1' trace). `subset_selection_control.py` compares the clearing proportions of the
samples with the fewest chain breaks and of those with the most chain breaks with those of random subsets of the same
size. `baseline_substrate_sensitivity.py` re-estimates the perturbation baseline from several generated samples as the
coordinate substrate and reports the spread of the baseline SD and any clearing call that changes with it.

`relax_samples.py` minimizes every sample with AMBER RNA.OL3 and GBn2 implicit solvent in OpenMM, with harmonic
restraints (5 kcal mol^-1 A^-2) on the input heavy atoms, after a bonded-only stage with a soft repulsion.

`ensemble_analyses.py` computes clearing, the target-level statistics, the six displacement criteria (on all shared
coordinates and on coordinates whose stems are paired in both experimental states) and the concordance statistics by
reference type for one ensemble, for the primary baseline substrate and for randomly drawn substrates
(`--random-substrates 20 --seed 0`), for independently generated perturbation sets (`--perturbation-seeds 20`), or for
the samples passing the fixed criteria (`--subset-checks`, with `--sweep-substrates` to use every passing sample in
turn as the substrate). `config/pairs.json` lists the apo/holo pairs used by it.

The scripts in `analysis/` accept `--selftest`, which runs their checks on synthetic inputs and exits, except
`compute_statistics.py`, `baseline_stability.py` and `target_multimodality_diagnostic.py`.

## Statistics

```bash
python analysis/compute_statistics.py
```

reads `data/clearing_signal_ratios.csv` and reports the target-level mean difference between angle and distance
clearing (0.066; bootstrap 95% interval 0.00 to 0.17), the exact sign test (p = 0.50 on the two non-zero targets), the pooled McNemar test (11/24 distance
and 14/24 angle coordinates clearing; p = 0.38) and the leave-largest-target-out comparison (5/16 and 7/16; p = 0.63).

## Training-exposure audit

`analysis/cohort_audit.py` evaluates a target against the RNA3DB 2024-04-26 split using two inputs: a tab-separated list of
the split's chains (chain id, split, sequence; 1,204 training and 516 test chains) and the Infernal table
`2024-04-26-full-cmscan.tbl` from the RNA3DB `incremental-update` release. Per target it reports whether any chain of the entry is in
the split, whether the sequence is identical to or contained in a training chain, the Rfam family of the target chain's best
significant hit, and the number of training chains with a significant (`!`) or sub-threshold (`?`) hit to that family.

```bash
cd analysis
python cohort_audit.py --selftest
python cohort_audit.py --chains-tsv <chains.tsv> --cmscan <cmscan.tbl> --cohort-csv ../data/cohort/cohort_audit.csv --recheck
python cohort_audit.py --chains-tsv <chains.tsv> --cmscan <cmscan.tbl> --target 1P5M:A:<sequence> --out audit_1P5M.csv
```

`--recheck` recomputes the family counts of every row in `cohort_audit.csv`.

`family_exposure_check.py` relates each analyzable target's family-level exposure to its clearing proportion (Spearman, permutation p-value):

```bash
python family_exposure_check.py --ratios ../data/clearing_signal_ratios.csv --audit ../data/cohort/cohort_audit.csv
```

## Notes

- RNA3DB training-exposure checks use the `train_set` split only.
- Pseudoknot detection is run before generation for every target.

## Citation

See `CITATION.cff`.
