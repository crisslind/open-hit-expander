# Open Hit Expander + Boltz-2

Research workflow for RDKit-based analogue enumeration, Boltz-2 protein–ligand cofolding / predicted potency, PLIP interactions, and offline compound assessment. **Predictions are hypotheses, not measured potency or validated binding modes.**

## Installation

Requirements: Python 3.10–3.13, RDKit and the Python dependencies declared in `pyproject.toml`; configure the `boltz` executable separately for inference and PLIP for fresh interaction analysis.

```bash
python -m pip install -e '.[notebook]'
# Development
python -m pip install -e '.[dev]'
```

Use `Open_Hit_Expander_boltz2_v070.ipynb` from the repository root. Avoid merging or replacing your existing project files without first backing up your current working repository.

## Existing run (no Boltz-2 inference)

In Section 1 of the notebook:

```python
RUN_MODE = "reload"
SAVED_TIMESTAMP = "YOUR_EXISTING_TIMESTAMP"
OUTPUT_DIRECTORY = Path("hit_expander_output")
```

Execute Section **1 → 6.7 → 7** for 3D structures and median-vs-best comparison. Execute **9 → 10 → 11** for previously saved PLIP results and the final assessment. Section 8 is configured not to rerun PLIP on reload; change `RUN_PLIP` explicitly only when needed. **Do not execute Section 6.5** on reload; this is the expensive inference step.

On reload, saved Boltz JSON/structure results are reparsed and the analysis tables are refreshed, but no new cofolding takes place. The notebook visualizer displays an existing replicate whose predicted pIC50 is closest to the median; an optional overlay shows the replicate with highest predicted pIC50. Neither is an averaged coordinate structure.

## New run

Set `RUN_MODE = "new"`, configure the ligand enumeration and protein target parameters, and follow the notebook from top to bottom. The notebook includes demonstration inputs and a protein example—**replace them with your intended system**. Boltz-2 inference is performed only in Section 6.5. `N_REPLICATES` controls independent runs; inspect the command and sampling preset before launching large batches.

## Outputs

Run-specific files are written under `hit_expander_output/boltz2_cofold/<timestamp>/`. Key outputs:

- `replicate_predictions.csv`: potency from independent replicate predictions
- `hit_expander_boltz2_all_results.csv`: median statistics, representative structure and highest-pIC50 pose metadata
- `interaction_analysis/`: saved PLIP status, details, fingerprint matrix and heatmap
- `compound_assessment/compound_summary.csv`: sortable export with potency and interaction metrics
- `compound_assessment/compound_assessment.html`: standalone offline summary

The report treats PLIP status `ok` and `success` as successful; unavailable PLIP analyses remain missing rather than silently becoming zero contacts. The Jaccard contact similarity uses residue–interaction-type features, not 3D pose similarity. Assessment labels depend on configurable effect/variability cutoffs and have not been calibrated to experimental uncertainty. Pose RMSD is not currently calculated.

## Checks

```bash
python -m compileall -q src
python -m pytest -q tests
```

See `tests/` for targeted metadata and PLIP regression checks. **End-to-end Boltz-2 inference, interactive viewer rendering, and full scientific validation are not covered by these tests.**
